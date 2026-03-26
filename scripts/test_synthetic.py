from __future__ import annotations

import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from turboquant.attention_metrics import metrics_frame
from turboquant.io_utils import ensure_dir
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig


ARTIFACT_ROOT = Path("artifacts")


def sample_unit_vectors(num_vectors: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((num_vectors, dim), generator=generator)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def mse_experiment(dim: int, num_vectors: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_vectors, dim=dim, seed=0)
    for bits in (2, 3, 4):
        quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(dim=dim, bits=bits, device="cpu", dtype="float32")
        )
        encoded = quantizer.quantize(x)
        reconstruction = quantizer.dequantize(encoded)
        mse = torch.mean((x - reconstruction) ** 2).item()
        rows.append({"experiment": "mse", "bits": bits, "metric": "mse", "value": mse})
    return rows


def prod_experiment(dim: int, num_pairs: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=1)
    y = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=2)
    exact = (x * y).sum(dim=-1)
    for bits in (2, 3, 4):
        quantizer = TurboQuantProd(
            TurboQuantProdConfig(dim=dim, total_bits=bits, device="cpu", dtype="float32")
        )
        encoded = quantizer.quantize(x)
        estimate = quantizer.estimate_inner_product(y, encoded)
        error = estimate - exact
        rows.append({"experiment": "prod", "bits": bits, "metric": "bias", "value": error.mean().item()})
        rows.append(
            {
                "experiment": "prod",
                "bits": bits,
                "metric": "variance",
                "value": float(error.var(unbiased=False).item()),
            }
        )
        rows.append(
            {
                "experiment": "prod",
                "bits": bits,
                "metric": "mae",
                "value": float(error.abs().mean().item()),
            }
        )
    return rows


def replay_experiment(dim: int, seq_len: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    keys = torch.randn((1, 1, seq_len, dim), generator=generator)
    queries = torch.randn((1, 1, max(2, seq_len // 4), dim), generator=generator)
    exact_logits = torch.einsum("bhqd,bhsd->bhqs", queries, keys)
    for bits in (2, 3, 4):
        codec = KVCodec(KVCodecConfig(head_dim=dim, key_bits=bits, value_bits=bits))
        encoded = codec.encode_keys(keys)
        estimated_logits = codec.estimator.turboquant(queries, encoded)
        summary = {
            "experiment": "replay",
            "bits": bits,
            "metric": "mae",
            "value": float((estimated_logits - exact_logits).abs().mean().item()),
        }
        rows.append(summary)
    for mixed_bits in (2.5, 3.5):
        codec = KVCodec(
            KVCodecConfig(
                head_dim=dim,
                key_bits=math.floor(mixed_bits),
                value_bits=math.floor(mixed_bits),
                mixed_key_bits=mixed_bits,
            )
        )
        encoded = codec.encode_keys(keys)
        estimated_logits = codec.estimator.turboquant(queries, encoded)
        rows.append(
            {
                "experiment": "replay",
                "bits": mixed_bits,
                "metric": "mae",
                "value": float((estimated_logits - exact_logits).abs().mean().item()),
            }
        )
    return rows


def main() -> int:
    ensure_dir(ARTIFACT_ROOT / "metrics")
    rows = []
    rows.extend(mse_experiment(dim=128, num_vectors=1024))
    rows.extend(prod_experiment(dim=128, num_pairs=2048))
    rows.extend(replay_experiment(dim=128, seq_len=64))
    frame = metrics_frame(rows)
    metrics_path = ARTIFACT_ROOT / "metrics" / "synthetic_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    print(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
