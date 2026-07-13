from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import ensure_dir, stable_hash, write_json
from turboquant.triality_fixture_gguf import write_fixture_gguf
from turboquant.triality_contract import (
    TRIALITY_ALLOWED_MODES,
    TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES,
    build_triality_fixture_manifest,
    build_triality_metadata,
    build_triality_payload,
    expected_modalities,
    payload_json_dumps,
)
from turboquant.triality_schema_v2 import build_triality_v2_tensors


def fixture_metadata(
    *,
    metadata: dict[str, object],
    head_dim: int,
    num_layers: int,
    num_kv_heads: int,
    architecture: str,
) -> dict[str, object]:
    return {
        "general.architecture": architecture,
        f"{architecture}.block_count": num_layers,
        f"{architecture}.embedding_length": head_dim * num_kv_heads,
        f"{architecture}.attention.head_count": num_kv_heads,
        f"{architecture}.attention.head_count_kv": num_kv_heads,
        f"{architecture}.vocab_size": 32000,
        f"{architecture}.context_length": 4096,
        **metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic Triality fixture bundle for parent-stack CI."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=TRIALITY_ALLOWED_MODES)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--model-family",
        default="huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
    )
    parser.add_argument(
        "--source-ftype",
        choices=TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES,
        default="q8_0",
    )
    parser.add_argument("--modality-scope", default=None)
    parser.add_argument("--schema-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--profile-id", default="v2")
    parser.add_argument("--enable-ncka", action="store_true")
    parser.add_argument("--enable-urt", action="store_true")
    parser.add_argument("--generated-at-utc", default="1970-01-01T00:00:00+00:00")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.schema_version != 2 and (args.enable_ncka or args.enable_urt):
        raise ValueError("NC-KA and URT fixture export require --schema-version 2")
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
        "generated_at_utc": args.generated_at_utc,
    }
    payload = build_triality_payload(
        mode=args.mode,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        model_family=args.model_family,
        weight_source_ftype=args.source_ftype,
        modality_scope=args.modality_scope,
        source_manifest=source_manifest,
        offline_metrics=offline_metrics,
        schema_version=args.schema_version,
        profile_id=args.profile_id,
        enable_ncka=args.enable_ncka,
        enable_urt=args.enable_urt,
    )
    payload_json = payload_json_dumps(payload)
    metadata = build_triality_metadata(
        mode=args.mode,
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
    )

    payload_path = bundle_dir / "triality-payload.json"
    metadata_path = bundle_dir / "triality-contract-metadata.json"
    metrics_path = bundle_dir / "triality-offline-metrics.json"
    manifest_path = bundle_dir / "triality-fixture-manifest.json"
    text_gguf_path = bundle_dir / "triality-fixture.gguf"
    modalities = expected_modalities(
        model_family=args.model_family,
        modality_scope=args.modality_scope,
    )
    mmproj_required = len(modalities) > 1
    mmproj_path = (
        bundle_dir / "mmproj-triality-fixture.gguf" if mmproj_required else None
    )
    stale_mmproj_path = bundle_dir / "mmproj-triality-fixture.gguf"
    if mmproj_path is None and stale_mmproj_path.exists():
        stale_mmproj_path.unlink()

    write_json(payload_path, payload)
    write_json(metadata_path, metadata)
    write_json(metrics_path, offline_metrics)
    text_tensors = {
        "blk.0.attn_q.weight": {
            "dtype": "f32",
            "shape": [args.head_dim, args.head_dim],
            "data": [0.0] * (args.head_dim * args.head_dim),
        }
    }
    if args.schema_version == 2:
        text_tensors.update(build_triality_v2_tensors(payload))
    write_fixture_gguf(
        path=text_gguf_path,
        metadata=fixture_metadata(
            metadata=metadata,
            head_dim=args.head_dim,
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            architecture="llama",
        ),
        tensors=text_tensors,
    )
    if mmproj_path is not None:
        write_fixture_gguf(
            path=mmproj_path,
            metadata=fixture_metadata(
                metadata={
                    "mmproj.required": True,
                    "mmproj.modalities": ",".join(modalities),
                },
                head_dim=max(args.head_dim, 128),
                num_layers=1,
                num_kv_heads=1,
                architecture="mmproj",
            ),
            tensors={
                "mmproj.proj.weight": {
                    "dtype": "f32",
                    "shape": [max(args.head_dim, 128), max(args.head_dim, 128)],
                    "data": [0.0] * (max(args.head_dim, 128) ** 2),
                }
            },
        )

    manifest = build_triality_fixture_manifest(
        mode=args.mode,
        model_family=args.model_family,
        source_ftype=args.source_ftype,
        generated_at_utc=source_manifest["generated_at_utc"],
        payload_path=payload_path.name,
        metadata_path=metadata_path.name,
        metrics_path=metrics_path.name,
        text_model_path=text_gguf_path.name,
        mmproj_model_path=mmproj_path.name if mmproj_path is not None else None,
        payload_hash=stable_hash(payload_json),
        metadata_hash=stable_hash(metadata_path.read_text(encoding="utf-8")),
        metrics_hash=stable_hash(metrics_path.read_text(encoding="utf-8")),
        text_model_hash=hashlib.sha256(text_gguf_path.read_bytes()).hexdigest(),
        mmproj_model_hash=(
            hashlib.sha256(mmproj_path.read_bytes()).hexdigest()
            if mmproj_path is not None
            else None
        ),
        triality_schema_version=args.schema_version,
        modality_scope=payload["weight_plan"].get("modality_scope"),
    )
    write_json(manifest_path, manifest)

    print(f"fixture_dir={bundle_dir}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
