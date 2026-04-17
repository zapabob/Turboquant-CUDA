from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import ensure_dir, stable_hash, write_json
from turboquant.triality_contract import (
    TRIALITY_ALLOWED_MODES,
    build_triality_metadata,
    build_triality_payload,
    payload_json_dumps,
)

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32


def _write_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(len(encoded).to_bytes(8, "little"))
    buffer.extend(encoded)


def _write_value(buffer: bytearray, value: object) -> None:
    if isinstance(value, bool):
        buffer.extend((7).to_bytes(4, "little"))
        buffer.extend((1 if value else 0).to_bytes(1, "little"))
        return
    if isinstance(value, int):
        if value >= 0x1_0000_0000:
            buffer.extend((10).to_bytes(4, "little"))
            buffer.extend(int(value).to_bytes(8, "little"))
        else:
            buffer.extend((4).to_bytes(4, "little"))
            buffer.extend(int(value).to_bytes(4, "little"))
        return
    if isinstance(value, float):
        import struct

        buffer.extend((6).to_bytes(4, "little"))
        buffer.extend(struct.pack("<f", float(value)))
        return
    buffer.extend((8).to_bytes(4, "little"))
    _write_string(buffer, str(value))


def write_fixture_gguf(
    *,
    path: Path,
    metadata: dict[str, object],
    head_dim: int,
    num_layers: int,
    num_kv_heads: int,
) -> None:
    buffer = bytearray()
    full_metadata = {
        "general.architecture": "llama",
        "llama.block_count": num_layers,
        "llama.embedding_length": head_dim * num_kv_heads,
        "llama.attention.head_count": num_kv_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.vocab_size": 32000,
        "llama.context_length": 4096,
        **metadata,
    }

    tensor_name = "blk.0.attn_q.weight"
    tensor_dims = (head_dim, head_dim)
    tensor_dtype = 0  # F32
    tensor_offset = 0

    buffer.extend(GGUF_MAGIC.to_bytes(4, "little"))
    buffer.extend(GGUF_VERSION.to_bytes(4, "little"))
    buffer.extend((1).to_bytes(8, "little"))
    buffer.extend(len(full_metadata).to_bytes(8, "little"))

    for key, value in full_metadata.items():
        _write_string(buffer, key)
        _write_value(buffer, value)

    _write_string(buffer, tensor_name)
    buffer.extend((len(tensor_dims)).to_bytes(4, "little"))
    for dim in tensor_dims:
        buffer.extend(int(dim).to_bytes(8, "little"))
    buffer.extend(int(tensor_dtype).to_bytes(4, "little"))
    buffer.extend(int(tensor_offset).to_bytes(8, "little"))

    while len(buffer) % GGUF_ALIGNMENT != 0:
        buffer.append(0)

    tensor_bytes = head_dim * head_dim * 4
    buffer.extend(b"\x00" * tensor_bytes)
    path.write_bytes(buffer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic Triality fixture bundle for parent-stack CI."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=TRIALITY_ALLOWED_MODES)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = ensure_dir(Path(args.output_dir) / args.mode)

    offline_metrics = {
        "hidden_mse": 0.0125 if args.mode == "paper-faithful" else 0.0081,
        "attention_mse": 0.0162 if args.mode == "paper-faithful" else 0.0104,
        "logit_delta": 0.0048 if args.mode == "paper-faithful" else 0.0033,
        "runtime_tokens_per_second": 36.5 if args.mode == "paper-faithful" else 44.2,
        "quality_label": "reference-fixture",
    }
    source_manifest = {
        "generator": "scripts/export_triality_fixture.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = build_triality_payload(
        mode=args.mode,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        source_manifest=source_manifest,
        offline_metrics=offline_metrics,
    )
    payload_json = payload_json_dumps(payload)
    metadata = build_triality_metadata(
        mode=args.mode,
        payload_json=payload_json,
    )

    payload_path = bundle_dir / "triality-payload.json"
    metadata_path = bundle_dir / "triality-contract-metadata.json"
    metrics_path = bundle_dir / "triality-offline-metrics.json"
    manifest_path = bundle_dir / "triality-fixture-manifest.json"
    gguf_path = bundle_dir / "triality-fixture.gguf"

    write_json(payload_path, payload)
    write_json(metadata_path, metadata)
    write_json(metrics_path, offline_metrics)
    write_fixture_gguf(
        path=gguf_path,
        metadata=metadata,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
    )
    manifest = {
        "schema_version": 1,
        "fixture_kind": "triality-fixture-bundle",
        "mode": args.mode,
        "generated_at_utc": source_manifest["generated_at_utc"],
        "paths": {
            "payload": payload_path.name,
            "metadata": metadata_path.name,
            "offline_metrics": metrics_path.name,
            "gguf": gguf_path.name,
        },
        "hashes": {
            "payload_sha256": stable_hash(payload_json),
            "metadata_sha256": stable_hash(metadata_path.read_text(encoding="utf-8")),
        },
    }
    write_json(manifest_path, manifest)

    print(f"fixture_dir={bundle_dir}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
