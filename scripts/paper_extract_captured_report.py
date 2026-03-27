from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from scripts.export_report import (
    render_attention_matplotlib,
    render_attention_plotly,
    render_runtime_matplotlib,
    render_runtime_plotly,
)
from turboquant.io_utils import ensure_dir
from turboquant.analysis import summarize_trial_metrics


PAPER_MODES = {"exact", "key_only_random", "full_kv"}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(frame.iloc[row_idx][column]) for column in columns) + " |"
        for row_idx in range(len(frame))
    ]
    return "\n".join([header, separator, *rows])


def build_mean_pm_sd_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    metric_pairs = [
        ("logit_cosine_similarity", "std", "logit_cosine_mean_pm_sd"),
        ("hidden_cosine_similarity", "std", "hidden_cosine_mean_pm_sd"),
        ("memory_ratio_vs_exact", "std", "memory_ratio_mean_pm_sd"),
    ]
    for mode in ("key_only_random", "full_kv"):
        mode_frame = summary_frame.loc[summary_frame["mode"] == mode]
        for bit_setting in sorted(mode_frame["bit_setting"].astype(str).unique()):
            subset = mode_frame.loc[mode_frame["bit_setting"].astype(str) == bit_setting]
            if subset.empty:
                continue
            row: dict[str, str | float] = {"mode": mode, "bit_setting": bit_setting}
            for mean_metric, spread_column, label in metric_pairs:
                mean_row = subset.loc[subset["metric"] == mean_metric]
                if mean_row.empty:
                    continue
                mean_value = float(mean_row["mean"].iloc[0])
                spread_value = float(mean_row[spread_column].iloc[0])
                row[label] = f"{mean_value:.6f} +/- {spread_value:.6f}"
                row[f"{mean_metric}_mean"] = mean_value
                row[f"{mean_metric}_sd"] = spread_value
            rows.append(row)
    return pd.DataFrame(rows)


def render_mean_pm_sd_plot(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    metric_pairs = [
        ("logit_cosine_similarity_mean", "logit_cosine_similarity_sd", "Logit Cosine"),
        ("hidden_cosine_similarity_mean", "hidden_cosine_similarity_sd", "Hidden Cosine"),
        ("memory_ratio_vs_exact_mean", "memory_ratio_vs_exact_sd", "Memory Ratio"),
    ]
    for axis, (mean_col, sd_col, title) in zip(axes, metric_pairs, strict=True):
        for mode in ("key_only_random", "full_kv"):
            mode_frame = summary_table.loc[summary_table["mode"] == mode].copy()
            if mode_frame.empty or mean_col not in mode_frame.columns:
                continue
            axis.errorbar(
                mode_frame["bit_setting"].astype(float),
                mode_frame[mean_col],
                yerr=mode_frame[sd_col].fillna(0.0),
                marker="o",
                capsize=4,
                label=mode,
            )
        axis.set_title(f"{title} (mean +/- sd)")
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the paper-baseline captured report from an existing combined summary.")
    parser.add_argument("--source-csv", default="artifacts/metrics/attention_summary_captured.csv")
    parser.add_argument("--source-trials", default=None)
    parser.add_argument("--output-dir", default="artifacts/paper_baseline/qwen_captured_from_existing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")

    if args.source_trials is not None:
        trial_frame = pd.read_csv(Path(args.source_trials))
        trial_frame = trial_frame.loc[trial_frame["mode"].isin(PAPER_MODES)].copy()
        trial_frame["dataset"] = "paper_captured"
        summary = summarize_trial_metrics(trial_frame)
        trial_frame.to_csv(metrics_dir / "attention_trials_captured.csv", index=False)
    else:
        summary = pd.read_csv(Path(args.source_csv))
        summary = summary.loc[summary["mode"].isin(PAPER_MODES)].copy()

    summary.to_csv(metrics_dir / "attention_summary_captured.csv", index=False)
    (metrics_dir / "attention_summary_captured.md").write_text(markdown_table(summary), encoding="utf-8")

    mean_pm_sd_table = build_mean_pm_sd_table(summary)
    mean_pm_sd_table.to_csv(metrics_dir / "attention_summary_captured_mean_pm_sd.csv", index=False)
    (metrics_dir / "attention_summary_captured_mean_pm_sd.md").write_text(
        markdown_table(mean_pm_sd_table),
        encoding="utf-8",
    )

    render_attention_matplotlib(summary, plots_dir / "attention_tradeoffs_captured.png")
    render_attention_plotly(summary, plots_dir / "attention_tradeoffs_captured.html")
    render_runtime_matplotlib(summary, plots_dir / "attention_runtime_tradeoffs_captured.png")
    render_runtime_plotly(summary, plots_dir / "attention_runtime_tradeoffs_captured.html")
    render_mean_pm_sd_plot(mean_pm_sd_table, plots_dir / "attention_mean_pm_sd_captured.png")
    print(summary.loc[summary["metric"].isin(["hidden_cosine_similarity", "logit_cosine_similarity", "memory_ratio_vs_exact"])])
    print(mean_pm_sd_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
