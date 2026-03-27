from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from turboquant.analysis import (
    compose_sensitive_layer_policy_rows,
    evaluate_layer_grid,
    summarize_value_sensitivity,
    synthetic_kv,
)
from turboquant.io_utils import ensure_dir
from turboquant.research_extension import compute_value_sensitivity_rows, evaluate_value_protection_grid


ARTIFACT_ROOT = Path("artifacts") / "research_extension"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run value sensitivity and protection-grid studies.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--synthetic-layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT / "metrics"))
    parser.add_argument("--protection-grid-layer-limit", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    low_bits_grid = tuple(sorted({int(bits) for bits in bit_grid if bits >= 2.0}))
    output_dir = ensure_dir(Path(args.output_dir))

    sensitivity_rows: list[dict[str, float | int | str]] = []
    protection_rows: list[dict[str, float | int | str]] = []
    trial_rows: list[dict[str, float | int | str]] = []
    for trial in range(args.trials):
        for layer_idx in range(args.synthetic_layers):
            keys, values = synthetic_kv(
                seed=9_000 + (trial * 131) + layer_idx,
                batch=args.batch,
                heads=args.heads,
                seq_len=args.seq_len,
                dim=args.head_dim,
            )
            sensitivity_rows.extend(
                compute_value_sensitivity_rows(
                    dataset="research_synthetic",
                    layer_idx=layer_idx,
                    trial=trial,
                    keys=keys,
                    values=values,
                )
            )
            trial_rows.extend(
                evaluate_layer_grid(
                    dataset="research_synthetic",
                    keys=keys,
                    values=values,
                    trial=trial,
                    layer_idx=layer_idx,
                    bit_grid=bit_grid,
                )
            )
            if layer_idx < args.protection_grid_layer_limit:
                protection_rows.extend(
                    evaluate_value_protection_grid(
                        dataset="research_synthetic",
                        layer_idx=layer_idx,
                        trial=trial,
                        keys=keys,
                        values=values,
                        low_bits_grid=low_bits_grid or (2, 3),
                    )
                )

    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    trial_frame = pd.DataFrame(trial_rows)
    sensitivity_summary = summarize_value_sensitivity(sensitivity_frame) if not sensitivity_frame.empty else pd.DataFrame()
    layer_policy_frame = (
        compose_sensitive_layer_policy_rows(trial_frame=trial_frame, sensitivity_frame=sensitivity_frame)
        if not sensitivity_frame.empty
        else pd.DataFrame()
    )
    protection_frame = pd.DataFrame(protection_rows)
    protection_summary = (
        protection_frame.groupby(["mode", "bit_setting", "protected_fraction", "secondary_fraction", "high_bits", "low_rank_rank"], as_index=False)
        .agg(
            hidden_cosine_similarity=("hidden_cosine_similarity", "mean"),
            attention_output_relative_error=("attention_output_relative_error", "mean"),
            memory_ratio_vs_exact=("memory_ratio_vs_exact", "mean"),
        )
        if not protection_frame.empty
        else pd.DataFrame()
    )

    sensitivity_frame.to_csv(output_dir / "value_sensitivity_synthetic.csv", index=False)
    sensitivity_summary.to_csv(output_dir / "value_sensitivity_summary_synthetic.csv", index=False)
    layer_policy_frame.to_csv(output_dir / "sensitive_layer_policy_synthetic.csv", index=False)
    protection_frame.to_csv(output_dir / "value_protection_grid_synthetic.csv", index=False)
    protection_summary.to_csv(output_dir / "value_protection_grid_summary_synthetic.csv", index=False)
    print(sensitivity_summary)
    print(protection_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
