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
    build_triality_payload,
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
    gguf_path = bundle_dir / paths["gguf"]

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offline_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not gguf_path.exists():
        raise FileNotFoundError(f"fixture GGUF is missing: {gguf_path}")

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

    rebuilt_payload = build_triality_payload(
        mode=payload["mode"],
        head_dim=int(payload["head_dim"]),
        num_layers=int(payload["num_layers"]),
        num_kv_heads=int(payload["num_kv_heads"]),
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
