"""Statistical summaries for TurboQuant experiment metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from turboquant.eval_stats import summarize_continuous_metrics

QWEN_3060_QUALITY_METRICS = [
    "logit_cosine_similarity",
    "next_logit_kl",
    "hidden_cosine_similarity",
    "memory_ratio_vs_exact",
]

QWEN_3060_PLOT_MODES = [
    "exact",
    "key_only_random",
    "full_kv",
    "asym_q8_turbo4",
    "asym_q8_turbo3",
    "multiscreen_relevance",
    "key_only_block_so8_triality_vector",
]


def summarize_metric_trials(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate trial rows into summary statistics with 95% confidence intervals."""

    return summarize_continuous_metrics(frame, group_columns=group_columns, value_column="value")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(frame.iloc[row_idx][column]) for column in columns) + " |"
        for row_idx in range(len(frame))
    ]
    return "\n".join([header, separator, *rows])


def build_qwen_3060_mean_pm_sd_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for mode in QWEN_3060_PLOT_MODES:
        bit_values = sorted({str(value) for value in summary_frame["bit_setting"].astype(str).unique()}, key=float)
        for bit_setting in bit_values:
            subset = summary_frame.loc[
                (summary_frame["mode"] == mode) & (summary_frame["bit_setting"].astype(str) == bit_setting)
            ]
            if subset.empty:
                continue
            row: dict[str, str | float] = {"mode": mode, "bit_setting": bit_setting}
            for metric in QWEN_3060_QUALITY_METRICS:
                metric_row = subset.loc[subset["metric"] == metric]
                if metric_row.empty:
                    continue
                mean_value = float(metric_row["mean"].iloc[0])
                sd_value = float(metric_row["std"].iloc[0])
                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_sd"] = sd_value
                row[f"{metric}_mean_pm_sd"] = f"{mean_value:.6f} +/- {sd_value:.6f}"
            rows.append(row)
    return pd.DataFrame(rows)


def write_qwen_3060_markdown_summary(
    *,
    summary_frame: pd.DataFrame,
    mean_pm_sd_frame: pd.DataFrame,
    friedman_frame: pd.DataFrame | None,
    pairwise_frame: pd.DataFrame | None,
    output_path: Path,
) -> None:
    lines = [
        "# Qwen 3060 Matrix Report",
        "",
        "## Main Groups",
        "",
        "- exact",
        "- key_only_random",
        "- full_kv",
        "- asym_q8_turbo4",
        "- asym_q8_turbo3",
        "- multiscreen_relevance",
        "- key_only_block_so8_triality_vector",
        "",
        "## Mean +/- SD",
        "",
        markdown_table(mean_pm_sd_frame) or "_No summary rows._",
        "",
        "## Friedman",
        "",
        markdown_table(friedman_frame) if friedman_frame is not None and not friedman_frame.empty else "_No Friedman rows._",
        "",
        "## Pairwise Wilcoxon-Holm",
        "",
        markdown_table(pairwise_frame) if pairwise_frame is not None and not pairwise_frame.empty else "_No pairwise rows._",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
