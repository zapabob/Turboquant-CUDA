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

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import evaluate_layer_grid, melt_metric_rows, summarize_trial_metrics, synthetic_kv
from turboquant.io_utils import ensure_dir
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.reporting import summarize_metric_trials
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig


ARTIFACT_ROOT = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic TurboQuant validation sweeps.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-vectors", type=int, default=1024)
    parser.add_argument("--num-pairs", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    return parser.parse_args()


def sample_unit_vectors(num_vectors: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((num_vectors, dim), generator=generator, dtype=torch.float32)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def mse_experiment(dim: int, num_vectors: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_vectors, dim=dim, seed=1_000 + trial)
    for bits in (2, 3, 4):
        quantizer = TurboQuantMSE(TurboQuantMSEConfig(dim=dim, bits=bits, device="cpu", dtype="float32"))
        encoded = quantizer.quantize(x)
        reconstruction = quantizer.dequantize(encoded)
        mse = torch.mean((x - reconstruction) ** 2).item()
        rows.append({"experiment": "mse", "mode": "stage1", "bits": float(bits), "metric": "mse", "trial": trial, "value": mse})
    return rows


def prod_experiment(dim: int, num_pairs: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=2_000 + trial)
    y = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=3_000 + trial)
    exact = (x * y).sum(dim=-1)
    for bits in (2, 3, 4):
        quantizer = TurboQuantProd(TurboQuantProdConfig(dim=dim, total_bits=bits, device="cpu", dtype="float32"))
        encoded = quantizer.quantize(x)
        estimate = quantizer.estimate_inner_product(y, encoded)
        error = estimate - exact
        rows.append({"experiment": "prod", "mode": "stage2", "bits": float(bits), "metric": "bias", "trial": trial, "value": float(error.mean().item())})
        rows.append({"experiment": "prod", "mode": "stage2", "bits": float(bits), "metric": "variance", "trial": trial, "value": float(error.var(unbiased=False).item())})
        rows.append({"experiment": "prod", "mode": "stage2", "bits": float(bits), "metric": "mae", "trial": trial, "value": float(error.abs().mean().item())})
    return rows


def mixed_prod_experiment(dim: int, num_pairs: int, trial: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=4_000 + trial)
    y = sample_unit_vectors(num_vectors=num_pairs, dim=dim, seed=5_000 + trial)
    exact = (x * y).sum(dim=-1)
    for mixed_bits in (2.5, 3.5):
        allocation = ChannelBitAllocation.preset(effective_bits=mixed_bits - 1.0, width=dim)
        quantizer = TurboQuantProd(
            TurboQuantProdConfig(dim=dim, total_bits=int(math.floor(mixed_bits)), device="cpu", dtype="float32")
        )
        encoded = quantizer.quantize(x, allocation=allocation)
        estimate = quantizer.estimate_inner_product(y, encoded)
        error = estimate - exact
        rows.append({"experiment": "prod_mixed", "mode": "stage2_mixed", "bits": mixed_bits, "metric": "bias", "trial": trial, "value": float(error.mean().item())})
        rows.append({"experiment": "prod_mixed", "mode": "stage2_mixed", "bits": mixed_bits, "metric": "variance", "trial": trial, "value": float(error.var(unbiased=False).item())})
        rows.append({"experiment": "prod_mixed", "mode": "stage2_mixed", "bits": mixed_bits, "metric": "mae", "trial": trial, "value": float(error.abs().mean().item())})
    return rows


def replay_rows(dim: int, seq_len: int, heads: int, trial: int) -> pd.DataFrame:
    keys, values = synthetic_kv(seed=6_000 + trial, batch=1, heads=heads, seq_len=seq_len, dim=dim)
    rows = evaluate_layer_grid(
        dataset="synthetic_replay",
        keys=keys,
        values=values,
        trial=trial,
        layer_idx=0,
        bit_grid=[2.0, 2.5, 3.0, 3.5, 4.0, 8.0],
    )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    ensure_dir(ARTIFACT_ROOT / "metrics")

    rows: list[dict[str, float | int | str]] = []
    replay_frames: list[pd.DataFrame] = []
    for trial in range(args.trials):
        rows.extend(mse_experiment(dim=args.dim, num_vectors=args.num_vectors, trial=trial))
        rows.extend(prod_experiment(dim=args.dim, num_pairs=args.num_pairs, trial=trial))
        rows.extend(mixed_prod_experiment(dim=args.dim, num_pairs=args.num_pairs, trial=trial))
        replay_frames.append(replay_rows(dim=args.dim, seq_len=args.seq_len, heads=args.heads, trial=trial))

    trials_frame = pd.DataFrame(rows)
    summary_frame = summarize_metric_trials(
        trials_frame,
        group_columns=["experiment", "mode", "bits", "metric"],
    )

    replay_trials = pd.concat(replay_frames, ignore_index=True)
    replay_long = melt_metric_rows(replay_trials)
    replay_summary = summarize_trial_metrics(replay_trials)

    trials_frame.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_metrics_trials.csv", index=False)
    summary_frame.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_metrics.csv", index=False)
    replay_trials.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_replay_trials.csv", index=False)
    replay_long.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_replay_metrics_long.csv", index=False)
    replay_summary.to_csv(ARTIFACT_ROOT / "metrics" / "synthetic_replay_summary.csv", index=False)
    print(summary_frame)
    print(replay_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
