from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from turboquant.analysis import synthetic_kv, summarize_trial_metrics
from turboquant.io_utils import ensure_dir
from turboquant.paper_baseline import evaluate_paper_attention_grid


ARTIFACT_ROOT = Path("artifacts") / "paper_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch-only toy attention benchmark for the paper baseline.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--synthetic-layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(ARTIFACT_ROOT / "metrics")

    frames: list[pd.DataFrame] = []
    for trial in range(args.trials):
        for layer_idx in range(args.synthetic_layers):
            keys, values = synthetic_kv(
                seed=8_000 + (trial * 101) + layer_idx,
                batch=args.batch,
                heads=args.heads,
                seq_len=args.seq_len,
                dim=args.head_dim,
            )
            frames.append(
                pd.DataFrame(
                    evaluate_paper_attention_grid(
                        dataset="paper_synthetic_attention",
                        keys=keys,
                        values=values,
                        trial=trial,
                        layer_idx=layer_idx,
                        bit_grid=bit_grid,
                    )
                )
            )

    trial_frame = pd.concat(frames, ignore_index=True)
    summary_frame = summarize_trial_metrics(trial_frame)

    trial_frame.to_csv(output_dir / "attention_trials.csv", index=False)
    summary_frame.to_csv(output_dir / "attention_summary.csv", index=False)
    print(summary_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
