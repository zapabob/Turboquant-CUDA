from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import stable_hash
from turboquant.triality_fixture_gguf import read_fixture_gguf
from turboquant.triality_contract import (
    TRIALITY_FIXTURE_MANIFEST_VERSION,
    TRIALITY_GGUF_SCHEMA_V2,
    build_triality_payload,
    expected_modalities,
    payload_json_dumps,
    validate_triality_metadata,
    validate_triality_payload,
)
from turboquant.triality_schema_v2 import (
    TRIALITY_CONSENSUS_VIEWS,
    build_triality_v2_tensors,
    rotation_tensor_name,
    tensor_sha256,
    validate_rotation_tensor,
)


def resolve_bundle_path(bundle_dir: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest path {name} must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"manifest path {name} must be relative")
    resolved_bundle = bundle_dir.resolve()
    resolved = (resolved_bundle / relative).resolve()
    if resolved != resolved_bundle and resolved_bundle not in resolved.parents:
        raise ValueError(f"manifest path {name} escapes the fixture bundle")
    return resolved


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_equal(expected: object, actual: object) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(expected, float(actual), rel_tol=1.0e-6, abs_tol=1.0e-7)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            metadata_equal(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Triality fixture bundle.")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    bundle_dir = manifest_path.parent

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    payload_path = resolve_bundle_path(bundle_dir, paths["payload"], name="payload")
    metadata_path = resolve_bundle_path(bundle_dir, paths["metadata"], name="metadata")
    metrics_path = resolve_bundle_path(
        bundle_dir, paths["offline_metrics"], name="offline_metrics"
    )
    text_model_path = resolve_bundle_path(
        bundle_dir, paths["text_model"], name="text_model"
    )
    gguf_path = resolve_bundle_path(bundle_dir, paths["gguf"], name="gguf")
    mmproj_path_raw = paths.get("mmproj_model")
    mmproj_path = (
        resolve_bundle_path(bundle_dir, mmproj_path_raw, name="mmproj_model")
        if mmproj_path_raw
        else None
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offline_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not gguf_path.exists():
        raise FileNotFoundError(f"fixture GGUF is missing: {gguf_path}")
    if not text_model_path.exists():
        raise FileNotFoundError(f"text GGUF is missing: {text_model_path}")

    expected_modalities_list = expected_modalities(
        model_family=str(manifest["model_family"]),
        modality_scope=payload.get("weight_plan", {}).get("modality_scope"),
    )
    if manifest.get("schema_version") != TRIALITY_FIXTURE_MANIFEST_VERSION:
        raise ValueError(
            f"manifest schema_version must be {TRIALITY_FIXTURE_MANIFEST_VERSION}"
        )
    if int(manifest.get("triality_schema_version", 0)) != int(
        payload["schema_version"]
    ):
        raise ValueError("manifest triality_schema_version does not match payload")
    if manifest.get("text_model_path") != text_model_path.name:
        raise ValueError("manifest text_model_path must match paths.text_model")
    if gguf_path != text_model_path:
        raise ValueError("manifest paths.gguf must match paths.text_model")
    if manifest.get("gguf") is not None:
        raise ValueError("manifest should not expose legacy gguf at top level")
    if manifest.get("modalities") != expected_modalities_list:
        raise ValueError("manifest modalities do not match expected modality contract")

    mmproj_required = bool(manifest.get("mmproj_required", False))
    if mmproj_required != (len(expected_modalities_list) > 1):
        raise ValueError("manifest mmproj_required does not match modality contract")
    if mmproj_required:
        if mmproj_path is None:
            raise ValueError(
                "manifest is missing paths.mmproj_model for required mmproj pair"
            )
        if not mmproj_path.exists():
            raise FileNotFoundError(f"fixture mmproj GGUF is missing: {mmproj_path}")
        if manifest.get("mmproj_model_path") != mmproj_path.name:
            raise ValueError("manifest mmproj_model_path must match paths.mmproj_model")
    elif mmproj_path is not None:
        raise ValueError("text-only manifest must not include a mmproj_model path")

    validate_triality_payload(payload)
    validate_triality_metadata(metadata)

    gguf = read_fixture_gguf(gguf_path)
    gguf_metadata = gguf["metadata"]
    for key, expected_value in metadata.items():
        if key not in gguf_metadata:
            raise ValueError(f"fixture GGUF is missing metadata key {key}")
        if not metadata_equal(expected_value, gguf_metadata[key]):
            raise ValueError(f"fixture GGUF metadata mismatch for {key}")

    base_tensor_names = {"blk.0.attn_q.weight"}
    if int(payload["schema_version"]) == TRIALITY_GGUF_SCHEMA_V2:
        expected_tensors = build_triality_v2_tensors(payload)
        actual_tensors = gguf["tensors"]
        if set(actual_tensors) != base_tensor_names | set(expected_tensors):
            raise ValueError("fixture GGUF schema-v2 tensor set is not exact")
        for name, expected_tensor in expected_tensors.items():
            if name not in actual_tensors:
                raise ValueError(f"fixture GGUF is missing schema-v2 tensor {name}")
            actual_tensor = actual_tensors[name]
            expected_manifest = payload["tensor_manifest"][name]
            if actual_tensor["dtype"] != expected_manifest["dtype"]:
                raise ValueError(f"fixture GGUF tensor dtype mismatch for {name}")
            if actual_tensor["shape"] != expected_manifest["shape"]:
                raise ValueError(f"fixture GGUF tensor shape mismatch for {name}")
            if tensor_sha256(actual_tensor["data"]) != expected_manifest["sha256"]:
                raise ValueError(f"fixture GGUF tensor hash mismatch for {name}")
            if tensor_sha256(expected_tensor["data"]) != expected_manifest["sha256"]:
                raise ValueError(
                    f"payload tensor manifest is not deterministic for {name}"
                )
        for layer in range(int(payload["num_layers"])):
            for view in TRIALITY_CONSENSUS_VIEWS:
                name = rotation_tensor_name(layer, view, str(payload["profile_id"]))
                validate_rotation_tensor(
                    actual_tensors[name], head_dim=int(payload["head_dim"])
                )
    elif set(gguf["tensors"]) != base_tensor_names:
        raise ValueError("schema-v1 fixture GGUF must not contain schema-v2 tensors")

    payload_json = payload_json_dumps(payload)
    if int(metadata["hypura.turboquant.payload_bytes"]) != len(
        payload_json.encode("utf-8")
    ):
        raise ValueError("payload_bytes does not match normalized payload JSON size")

    if manifest["mode"] != payload["mode"]:
        raise ValueError("manifest mode does not match payload mode")

    expected_payload_hash = stable_hash(payload_json)
    if manifest["hashes"]["payload_sha256"] != expected_payload_hash:
        raise ValueError("payload hash mismatch")

    expected_metadata_hash = stable_hash(metadata_path.read_text(encoding="utf-8"))
    if manifest["hashes"]["metadata_sha256"] != expected_metadata_hash:
        raise ValueError("metadata hash mismatch")
    expected_metrics_hash = stable_hash(metrics_path.read_text(encoding="utf-8"))
    if manifest["hashes"].get("offline_metrics_sha256") != expected_metrics_hash:
        raise ValueError("offline metrics hash mismatch")
    if manifest["hashes"].get("text_model_sha256") != file_sha256(text_model_path):
        raise ValueError("text model hash mismatch")
    expected_mmproj_hash = file_sha256(mmproj_path) if mmproj_path is not None else None
    if manifest["hashes"].get("mmproj_model_sha256") != expected_mmproj_hash:
        raise ValueError("mmproj model hash mismatch")

    if "runtime_tokens_per_second" not in offline_metrics:
        raise ValueError("offline metrics must include runtime_tokens_per_second")
    if mmproj_required:
        sample_env = manifest.get("sample_env", {})
        for key in ("text_model", "mmproj_model", "image", "audio"):
            if key not in sample_env:
                raise ValueError(f"multimodal manifest sample_env is missing {key!r}")
    else:
        sample_env = manifest.get("sample_env", {})
        if "text_model" not in sample_env:
            raise ValueError("text manifest sample_env is missing 'text_model'")

    rebuilt_payload = build_triality_payload(
        mode=payload["mode"],
        head_dim=int(payload["head_dim"]),
        num_layers=int(payload["num_layers"]),
        num_kv_heads=int(payload["num_kv_heads"]),
        model_family=str(
            payload.get("model_family", manifest.get("model_family", "generic"))
        ),
        weight_source_ftype=str(
            payload.get("weight_plan", {}).get(
                "source_ftype", manifest.get("source_ftype", "q8_0")
            )
        ),
        weight_policy=payload.get("weight_plan", {}).get("policy"),
        weight_protected_roles=payload.get("weight_plan", {}).get("protected_roles"),
        weight_protected_layers=payload.get("weight_plan", {}).get("protected_layers"),
        modality_scope=payload.get("weight_plan", {}).get("modality_scope"),
        rotation_seed=int(payload["rotation_seed"]),
        source_manifest=payload.get("source_manifest"),
        offline_metrics=offline_metrics,
        schema_version=int(payload["schema_version"]),
        profile_id=str(payload.get("profile_id", "v2")),
        enable_ncka=bool(payload.get("ncka", {}).get("enabled", False)),
        enable_urt=bool(payload.get("urt", {}).get("enabled", False)),
    )
    if payload_json_dumps(rebuilt_payload) != payload_json:
        raise ValueError("fixture payload is not deterministic when rebuilt")

    print(f"validated_manifest={manifest_path}")
    print(f"mode={manifest['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
