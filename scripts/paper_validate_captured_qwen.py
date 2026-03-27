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
from turboquant.analysis import load_captured_runs, summarize_trial_metrics
from turboquant.io_utils import ensure_dir
from turboquant.paper_baseline import evaluate_paper_attention_grid


ARTIFACT_ROOT = Path("artifacts") / "paper_baseline" / "qwen_captured"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(frame.iloc[row_idx][column]) for column in columns) + " |"
        for row_idx in range(len(frame))
    ]
    return "\n".join([header, separator, *rows])


def build_mean_pm_mse_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "logit_cosine_similarity": "logit_cosine_mean_pm_mse",
        "hidden_cosine_similarity": "hidden_cosine_mean_pm_mse",
        "memory_ratio_vs_exact": "memory_ratio_mean_pm_mse",
    }
    rows: list[dict[str, str | float]] = []
    for mode in ("key_only_random", "full_kv"):
        for bit_setting in ("2", "2.5", "3", "3.5", "4"):
            record: dict[str, str | float] = {"mode": mode, "bit_setting": bit_setting}
            subset = summary_frame.loc[(summary_frame["mode"] == mode) & (summary_frame["bit_setting"].astype(str) == bit_setting)]
            if subset.empty:
                continue
            for metric, label in metrics.items():
                metric_row = subset.loc[subset["metric"] == metric]
                mse_metric = metric.replace("cosine_similarity", "mse")
                mse_row = subset.loc[subset["metric"] == mse_metric] if "cosine" in metric else metric_row
                if metric_row.empty:
                    continue
                mean_value = float(metric_row["mean"].iloc[0])
                mse_value = float(mse_row["mean"].iloc[0]) if not mse_row.empty else float("nan")
                record[label] = f"{mean_value:.6f} ± {mse_value:.6f}"
                record[f"{metric}_mean"] = mean_value
                record[f"{metric}_mse"] = mse_value
            rows.append(record)
    return pd.DataFrame(rows)


def render_mean_pm_mse_plot(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    metric_pairs = [
        ("logit_cosine_similarity_mean", "logit_cosine_similarity_mse", "Logit Cosine"),
        ("hidden_cosine_similarity_mean", "hidden_cosine_similarity_mse", "Hidden Cosine"),
        ("memory_ratio_vs_exact_mean", "memory_ratio_vs_exact_mse", "Memory Ratio"),
    ]
    for axis, (mean_col, mse_col, title) in zip(axes, metric_pairs, strict=True):
        for mode in ("key_only_random", "full_kv"):
            mode_frame = summary_table.loc[summary_table["mode"] == mode].copy()
            if mode_frame.empty or mean_col not in mode_frame.columns:
                continue
            axis.errorbar(
                mode_frame["bit_setting"].astype(float),
                mode_frame[mean_col],
                yerr=mode_frame[mse_col].fillna(0.0),
                marker="o",
                capsize=4,
                label=mode,
            )
        axis.set_title(f"{title} (mean ± mse)")
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-baseline captured replay on Qwen3.5-9B KV tensors.")
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")

    bundles = load_captured_runs(Path(args.kv_dir))
    frames: list[pd.DataFrame] = []
    for trial in range(args.trials):
        for bundle in bundles:
            if args.max_layers > 0 and bundle.layer_idx >= args.max_layers:
                continue
            frames.append(
                pd.DataFrame(
                    evaluate_paper_attention_grid(
                        dataset=f"paper_captured:{bundle.metadata.prompt_label}",
                        keys=bundle.keys,
                        values=bundle.values,
                        trial=trial,
                        layer_idx=bundle.layer_idx,
                        bit_grid=bit_grid,
                    )
                )
            )

    trial_frame = pd.concat(frames, ignore_index=True)
    summary_frame = summarize_trial_metrics(trial_frame)

    trial_path = metrics_dir / "attention_trials_captured.csv"
    summary_path = metrics_dir / "attention_summary_captured.csv"
    markdown_path = metrics_dir / "attention_summary_captured.md"
    mean_pm_mse_path = metrics_dir / "attention_summary_captured_mean_pm_mse.csv"
    mean_pm_mse_markdown_path = metrics_dir / "attention_summary_captured_mean_pm_mse.md"
    trial_frame.to_csv(trial_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    markdown_path.write_text(markdown_table(summary_frame), encoding="utf-8")

    mean_pm_mse_table = build_mean_pm_mse_table(summary_frame)
    mean_pm_mse_table.to_csv(mean_pm_mse_path, index=False)
    mean_pm_mse_markdown_path.write_text(markdown_table(mean_pm_mse_table), encoding="utf-8")

    render_attention_matplotlib(summary_frame, plots_dir / "attention_tradeoffs_captured.png")
    render_attention_plotly(summary_frame, plots_dir / "attention_tradeoffs_captured.html")
    render_runtime_matplotlib(summary_frame, plots_dir / "attention_runtime_tradeoffs_captured.png")
    render_runtime_plotly(summary_frame, plots_dir / "attention_runtime_tradeoffs_captured.html")
    render_mean_pm_mse_plot(mean_pm_mse_table, plots_dir / "attention_mean_pm_mse_captured.png")

    headline = summary_frame.loc[
        summary_frame["metric"].isin(["hidden_cosine_similarity", "logit_cosine_similarity", "memory_ratio_vs_exact"])
    ].copy()
    print(headline)
    print(mean_pm_mse_table)
    print(f"saved baseline captured metrics to {metrics_dir}")
    print(f"saved baseline captured plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
