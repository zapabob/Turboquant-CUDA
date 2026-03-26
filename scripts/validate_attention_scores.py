from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from turboquant.capture import CaptureMetadata
from turboquant.attention_metrics import summarize_attention_scores
from turboquant.io_utils import ensure_dir
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.runtime import DEFAULT_MODEL_ID, require_supported_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TurboQuant attention scores.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--output-dir", default="artifacts/metrics")
    parser.add_argument("--query-source", choices=["synthetic", "captured"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_queries(keys: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    query_count = max(1, min(4, keys.shape[-2]))
    queries = torch.randn(
        (*keys.shape[:-2], query_count, keys.shape[-1]),
        generator=generator,
        dtype=keys.dtype,
    )
    return queries


def main() -> int:
    require_supported_python()
    args = parse_args()
    kv_dir = Path(args.kv_dir)
    metrics_dir = ensure_dir(Path(args.output_dir))
    manifest_path = kv_dir / "capture_manifest.json"
    manifest_model_id = None
    if manifest_path.exists():
        manifest_model_id = pd.read_json(manifest_path, typ="series").get("model_name")
    rows: list[dict[str, float | int]] = []
    for key_path in sorted(kv_dir.glob("layer_*_key.pt")):
        layer_idx = int(key_path.stem.split("_")[1])
        keys = torch.load(key_path, map_location="cpu")
        if args.query_source == "captured":
            raise NotImplementedError(
                "Captured-query replay is reserved for real HF attention capture metadata. "
                "Use --query-source synthetic unless query tensors are added to the artifact set."
            )
        queries = choose_queries(keys=keys, seed=args.seed + layer_idx)
        exact = torch.einsum("bhqd,bhsd->bhqs", queries, keys)
        for bits in (2, 3, 4):
            codec = KVCodec(KVCodecConfig(head_dim=keys.shape[-1], key_bits=bits, value_bits=bits))
            estimated = codec.estimator.turboquant(queries, codec.encode_keys(keys))
            summary = summarize_attention_scores(exact, estimated)
            summary["layer"] = layer_idx
            summary["bits"] = bits
            summary["model_id"] = manifest_model_id or args.model_id
            summary["query_source"] = args.query_source
            rows.append(summary)
        for mixed_bits in (2.5, 3.5):
            codec = KVCodec(
                KVCodecConfig(
                    head_dim=keys.shape[-1],
                    key_bits=int(mixed_bits),
                    value_bits=int(mixed_bits),
                    mixed_key_bits=mixed_bits,
                )
            )
            estimated = codec.estimator.turboquant(queries, codec.encode_keys(keys))
            summary = summarize_attention_scores(exact, estimated)
            summary["layer"] = layer_idx
            summary["bits"] = mixed_bits
            summary["model_id"] = manifest_model_id or args.model_id
            summary["query_source"] = args.query_source
            rows.append(summary)

    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_dir / "attention_scores.csv", index=False)
    summary_path = metrics_dir / "attention_scores.md"
    try:
        summary_text = frame.to_markdown(index=False)
    except ImportError:
        summary_text = "```\n" + frame.to_string(index=False) + "\n```"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
