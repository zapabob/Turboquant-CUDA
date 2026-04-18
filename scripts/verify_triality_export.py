from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import stable_hash
from turboquant.triality_contract import (
    TRIALITY_FIXTURE_MANIFEST_VERSION,
    build_triality_payload,
    expected_modalities,
    payload_json_dumps,
    validate_triality_metadata,
    validate_triality_payload,
)


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
    payload_path = bundle_dir / paths["payload"]
    metadata_path = bundle_dir / paths["metadata"]
    metrics_path = bundle_dir / paths["offline_metrics"]
    text_model_path = bundle_dir / paths["text_model"]
    gguf_path = bundle_dir / paths["gguf"]
    mmproj_path_raw = paths.get("mmproj_model")
    mmproj_path = bundle_dir / mmproj_path_raw if mmproj_path_raw else None

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
    if manifest.get("text_model_path") != text_model_path.name:
        raise ValueError("manifest text_model_path must match paths.text_model")
    if manifest.get("gguf") is not None:
        raise ValueError("manifest should not expose legacy gguf at top level")
    if manifest.get("modalities") != expected_modalities_list:
        raise ValueError("manifest modalities do not match expected modality contract")

    mmproj_required = bool(manifest.get("mmproj_required", False))
    if mmproj_required != (len(expected_modalities_list) > 1):
        raise ValueError("manifest mmproj_required does not match modality contract")
    if mmproj_required:
        if mmproj_path is None:
            raise ValueError("manifest is missing paths.mmproj_model for required mmproj pair")
        if not mmproj_path.exists():
            raise FileNotFoundError(f"fixture mmproj GGUF is missing: {mmproj_path}")
        if manifest.get("mmproj_model_path") != mmproj_path.name:
            raise ValueError("manifest mmproj_model_path must match paths.mmproj_model")
    elif mmproj_path is not None:
        raise ValueError("text-only manifest must not include a mmproj_model path")

    validate_triality_payload(payload)
    validate_triality_metadata(metadata)

    payload_json = payload_json_dumps(payload)
    if int(metadata["hypura.turboquant.payload_bytes"]) != len(payload_json.encode("utf-8")):
        raise ValueError("payload_bytes does not match normalized payload JSON size")

    if manifest["mode"] != payload["mode"]:
        raise ValueError("manifest mode does not match payload mode")

    expected_payload_hash = stable_hash(payload_json)
    if manifest["hashes"]["payload_sha256"] != expected_payload_hash:
        raise ValueError("payload hash mismatch")

    expected_metadata_hash = stable_hash(metadata_path.read_text(encoding="utf-8"))
    if manifest["hashes"]["metadata_sha256"] != expected_metadata_hash:
        raise ValueError("metadata hash mismatch")

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
        model_family=str(payload.get("model_family", manifest.get("model_family", "generic"))),
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
    )
    if payload_json_dumps(rebuilt_payload) != payload_json:
        raise ValueError("fixture payload is not deterministic when rebuilt")

    print(f"validated_manifest={manifest_path}")
    print(f"mode={manifest['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
