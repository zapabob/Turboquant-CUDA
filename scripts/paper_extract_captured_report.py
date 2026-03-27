from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from scripts.export_report import (
    render_attention_matplotlib,
    render_attention_plotly,
    render_runtime_matplotlib,
    render_runtime_plotly,
)
from turboquant.io_utils import ensure_dir
from turboquant.analysis import summarize_trial_metrics
from turboquant.schema import build_paper_turboquant_config, write_turboquant_config


PAPER_MODES = {"exact", "key_only_random", "full_kv"}
PAIRWISE_MODES = ("key_only_random", "full_kv")


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
        ("attention_output_relative_error", "std", "attention_output_error_mean_pm_sd"),
    ]
    for mode in PAIRWISE_MODES:
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
        for mode in PAIRWISE_MODES:
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


def build_memory_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for mode in PAIRWISE_MODES:
        mode_frame = summary_frame.loc[summary_frame["mode"] == mode]
        for bit_setting in sorted(mode_frame["bit_setting"].astype(str).unique(), key=float):
            subset = mode_frame.loc[mode_frame["bit_setting"].astype(str) == bit_setting]
            memory_bits = subset.loc[subset["metric"] == "memory_bits"]
            memory_ratio = subset.loc[subset["metric"] == "memory_ratio_vs_exact"]
            if memory_bits.empty or memory_ratio.empty:
                continue
            rows.append(
                {
                    "mode": mode,
                    "bit_setting": bit_setting,
                    "memory_bits_mean": float(memory_bits["mean"].iloc[0]),
                    "memory_bits_sd": float(memory_bits["std"].iloc[0]),
                    "memory_ratio_mean": float(memory_ratio["mean"].iloc[0]),
                    "memory_ratio_sd": float(memory_ratio["std"].iloc[0]),
                    "memory_bits_mean_pm_sd": f"{float(memory_bits['mean'].iloc[0]):.1f} +/- {float(memory_bits['std'].iloc[0]):.1f}",
                    "memory_ratio_mean_pm_sd": f"{float(memory_ratio['mean'].iloc[0]):.6f} +/- {float(memory_ratio['std'].iloc[0]):.6f}",
                }
            )
    return pd.DataFrame(rows)


def render_v_breakage_plot(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    metric_pairs = [
        ("logit_cosine_similarity_mean", "logit_cosine_similarity_sd", "Logit Cosine"),
        ("hidden_cosine_similarity_mean", "hidden_cosine_similarity_sd", "Hidden Cosine"),
        ("attention_output_relative_error_mean", "attention_output_relative_error_sd", "Attention Output Relative Error"),
    ]
    for axis, (mean_col, sd_col, title) in zip(axes, metric_pairs, strict=True):
        for mode in PAIRWISE_MODES:
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
        axis.set_title(f"{title} by bit")
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = (total - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def compute_statistics(trial_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantized = trial_frame.loc[trial_frame["mode"].isin(PAIRWISE_MODES)].copy()
    omnibus_rows: list[dict[str, str | float]] = []
    for metric in ("hidden_cosine_similarity", "attention_output_relative_error", "logit_cosine_similarity"):
        grouped = [group[metric].to_numpy() for _, group in quantized.groupby(["mode", "bits"])]
        result = stats.kruskal(*grouped)
        omnibus_rows.append(
            {
                "metric": metric,
                "test": "kruskal",
                "group_count": len(grouped),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    omnibus_frame = pd.DataFrame(omnibus_rows)

    pairwise_rows: list[dict[str, str | float]] = []
    hidden_p_values: list[float] = []
    error_p_values: list[float] = []
    hidden_indices: list[int] = []
    error_indices: list[int] = []

    for bits in sorted(quantized["bits"].dropna().unique()):
        key_only = quantized.loc[(quantized["mode"] == "key_only_random") & (quantized["bits"] == bits)].copy()
        full_kv = quantized.loc[(quantized["mode"] == "full_kv") & (quantized["bits"] == bits)].copy()
        tests = [
            ("hidden_cosine_similarity", "greater", "key_only_random > full_kv"),
            ("attention_output_relative_error", "less", "key_only_random < full_kv"),
            ("logit_cosine_similarity", "two-sided", "key_only_random != full_kv"),
        ]
        for metric, alternative, interpretation in tests:
            result = stats.mannwhitneyu(
                key_only[metric],
                full_kv[metric],
                alternative=alternative,
                method="exact",
            )
            row = {
                "bits": float(bits),
                "metric": metric,
                "test": "mannwhitneyu_exact",
                "alternative": alternative,
                "interpretation": interpretation,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "key_only_random_mean": float(key_only[metric].mean()),
                "key_only_random_sd": float(key_only[metric].std(ddof=1)),
                "full_kv_mean": float(full_kv[metric].mean()),
                "full_kv_sd": float(full_kv[metric].std(ddof=1)),
                "delta_key_minus_full": float(key_only[metric].mean() - full_kv[metric].mean()),
            }
            pairwise_rows.append(row)
            row_idx = len(pairwise_rows) - 1
            if metric == "hidden_cosine_similarity":
                hidden_p_values.append(float(result.pvalue))
                hidden_indices.append(row_idx)
            elif metric == "attention_output_relative_error":
                error_p_values.append(float(result.pvalue))
                error_indices.append(row_idx)

    pairwise_frame = pd.DataFrame(pairwise_rows)
    if hidden_p_values:
        adjusted = holm_bonferroni(hidden_p_values)
        for frame_idx, adjusted_p in zip(hidden_indices, adjusted, strict=True):
            pairwise_frame.loc[frame_idx, "p_value_holm"] = adjusted_p
    if error_p_values:
        adjusted = holm_bonferroni(error_p_values)
        for frame_idx, adjusted_p in zip(error_indices, adjusted, strict=True):
            pairwise_frame.loc[frame_idx, "p_value_holm"] = adjusted_p
    pairwise_frame["p_value_holm"] = pairwise_frame["p_value_holm"].fillna(pairwise_frame["p_value"])
    pairwise_frame["significant_0_05"] = pairwise_frame["p_value_holm"] < 0.05
    return omnibus_frame, pairwise_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the paper-baseline captured report from an existing combined summary.")
    parser.add_argument("--source-csv", default="artifacts/metrics/attention_summary_captured.csv")
    parser.add_argument("--source-trials", default=None)
    parser.add_argument("--output-dir", default="artifacts/paper_baseline/qwen_captured_from_existing")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
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
        omnibus_frame, pairwise_frame = compute_statistics(trial_frame)
        omnibus_frame.to_csv(metrics_dir / "attention_statistics_omnibus.csv", index=False)
        pairwise_frame.to_csv(metrics_dir / "attention_statistics_pairwise.csv", index=False)
        (metrics_dir / "attention_statistics_omnibus.md").write_text(markdown_table(omnibus_frame), encoding="utf-8")
        (metrics_dir / "attention_statistics_pairwise.md").write_text(markdown_table(pairwise_frame), encoding="utf-8")
    else:
        summary = pd.read_csv(Path(args.source_csv))
        summary = summary.loc[summary["mode"].isin(PAPER_MODES)].copy()
        omnibus_frame = pd.DataFrame()
        pairwise_frame = pd.DataFrame()

    summary.to_csv(metrics_dir / "attention_summary_captured.csv", index=False)
    (metrics_dir / "attention_summary_captured.md").write_text(markdown_table(summary), encoding="utf-8")

    mean_pm_sd_table = build_mean_pm_sd_table(summary)
    memory_table = build_memory_table(summary)
    mean_pm_sd_table.to_csv(metrics_dir / "attention_summary_captured_mean_pm_sd.csv", index=False)
    (metrics_dir / "attention_summary_captured_mean_pm_sd.md").write_text(
        markdown_table(mean_pm_sd_table),
        encoding="utf-8",
    )
    memory_table.to_csv(metrics_dir / "attention_memory_by_bit_mean_pm_sd.csv", index=False)
    (metrics_dir / "attention_memory_by_bit_mean_pm_sd.md").write_text(
        markdown_table(memory_table),
        encoding="utf-8",
    )

    render_attention_matplotlib(summary, plots_dir / "attention_tradeoffs_captured.png")
    render_attention_plotly(summary, plots_dir / "attention_tradeoffs_captured.html")
    render_runtime_matplotlib(summary, plots_dir / "attention_runtime_tradeoffs_captured.png")
    render_runtime_plotly(summary, plots_dir / "attention_runtime_tradeoffs_captured.html")
    render_mean_pm_sd_plot(mean_pm_sd_table, plots_dir / "attention_mean_pm_sd_captured.png")
    render_v_breakage_plot(mean_pm_sd_table, plots_dir / "attention_v_breakage_by_bit_sd.png")
    if args.write_config:
        config_out = Path(args.config_out) if args.config_out else output_dir / "turboquant_config.paper.json"
        payload = build_paper_turboquant_config(
            bit_grid=sorted({float(item) for item in summary["bits"].dropna().tolist()}),
            dim=128,
            artifact_refs={
                "summary_csv": str((metrics_dir / "attention_summary_captured.csv")).replace("\\", "/"),
                "mean_pm_sd_csv": str((metrics_dir / "attention_summary_captured_mean_pm_sd.csv")).replace("\\", "/"),
                "statistics_omnibus_csv": str((metrics_dir / "attention_statistics_omnibus.csv")).replace("\\", "/"),
                "statistics_pairwise_csv": str((metrics_dir / "attention_statistics_pairwise.csv")).replace("\\", "/"),
            },
        )
        write_turboquant_config(config_out, payload)
    print(summary.loc[summary["metric"].isin(["hidden_cosine_similarity", "logit_cosine_similarity", "memory_ratio_vs_exact"])])
    print(mean_pm_sd_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
