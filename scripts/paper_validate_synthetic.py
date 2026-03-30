from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from turboquant.paper_baseline import (
    PaperMSEConfig,
    PaperMixedBitPolicy,
    PaperProdConfig,
    PaperTurboQuantMSE,
    PaperTurboQuantProd,
)
from turboquant.io_utils import ensure_dir
from turboquant.reporting import summarize_metric_trials


ARTIFACT_ROOT = Path("artifacts") / "paper_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyTorch-only paper-faithful TurboQuant synthetic validation.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-vectors", type=int, default=1024)
    parser.add_argument("--num-pairs", type=int, default=2048)
    return parser.parse_args()


def sample_unit_vectors(num_vectors: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((num_vectors, dim), generator=generator, dtype=torch.float32)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def mse_rows(dim: int, num_vectors: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_vectors, dim=dim, seed=1_000 + trial)
    for bits in (2, 3, 4, 8):
        quantizer = PaperTurboQuantMSE(PaperMSEConfig(dim=dim, bits=bits, device="cpu", dtype="float32"))
        encoded = quantizer.quantize(x)
        reconstruction = quantizer.dequantize(encoded)
        rows.append(
            {
                "experiment": "paper_mse",
                "mode": "stage1",
                "bits": float(bits),
                "metric": "mse",
                "trial": trial,
                "value": float(torch.mean((x - reconstruction) ** 2).item()),
            }
        )
    return rows


def prod_rows(dim: int, num_pairs: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=2_000 + trial)
    y = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=3_000 + trial)
    exact = (x * y).sum(dim=-1)
    for bits in (2, 3, 4, 8):
        quantizer = PaperTurboQuantProd(PaperProdConfig(dim=dim, bits_total=bits, device="cpu", dtype="float32"))
        encoded = quantizer.quantize(x)
        error = quantizer.estimate_inner_product(y, encoded) - exact
        rows.extend(
            [
                {"experiment": "paper_prod", "mode": "stage2", "bits": float(bits), "metric": "bias", "trial": trial, "value": float(error.mean().item())},
                {"experiment": "paper_prod", "mode": "stage2", "bits": float(bits), "metric": "variance", "trial": trial, "value": float(error.var(unbiased=False).item())},
                {"experiment": "paper_prod", "mode": "stage2", "bits": float(bits), "metric": "mae", "trial": trial, "value": float(error.abs().mean().item())},
            ]
        )
    return rows


def mixed_prod_rows(dim: int, num_pairs: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=4_000 + trial)
    y = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=5_000 + trial)
    exact = (x * y).sum(dim=-1)
    for total_bits in (2.5, 3.5):
        policy = PaperMixedBitPolicy.for_total_bits(total_bits=total_bits, dim=dim)
        quantizer = PaperTurboQuantProd(
            PaperProdConfig(dim=dim, bits_total=int(math.floor(total_bits)), device="cpu", dtype="float32")
        )
        encoded = quantizer.quantize(x, allocation=policy.allocation(dim))
        error = quantizer.estimate_inner_product(y, encoded) - exact
        rows.extend(
            [
                {"experiment": "paper_prod_mixed", "mode": "stage2_mixed", "bits": total_bits, "metric": "bias", "trial": trial, "value": float(error.mean().item())},
                {"experiment": "paper_prod_mixed", "mode": "stage2_mixed", "bits": total_bits, "metric": "variance", "trial": trial, "value": float(error.var(unbiased=False).item())},
                {"experiment": "paper_prod_mixed", "mode": "stage2_mixed", "bits": total_bits, "metric": "mae", "trial": trial, "value": float(error.abs().mean().item())},
            ]
        )
    return rows


def main() -> int:
    args = parse_args()
    ensure_dir(ARTIFACT_ROOT / "metrics")

    rows: list[dict[str, float | int | str]] = []
    for trial in range(args.trials):
        rows.extend(mse_rows(dim=args.dim, num_vectors=args.num_vectors, trial=trial))
        rows.extend(prod_rows(dim=args.dim, num_pairs=args.num_pairs, trial=trial))
        rows.extend(mixed_prod_rows(dim=args.dim, num_pairs=args.num_pairs, trial=trial))

    trials_frame = pd.DataFrame(rows)
    summary_frame = summarize_metric_trials(
        trials_frame,
        group_columns=["experiment", "mode", "bits", "metric"],
    )

    trials_frame.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_metrics_trials.csv", index=False)
    summary_frame.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_metrics.csv", index=False)
    print(summary_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
