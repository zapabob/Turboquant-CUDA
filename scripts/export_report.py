from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from turboquant.io_utils import ensure_dir


ARTIFACT_ROOT = Path("artifacts")
METRICS_DIR = ARTIFACT_ROOT / "metrics"
PLOTS_DIR = ARTIFACT_ROOT / "plots"
REPORTS_DIR = ARTIFACT_ROOT / "reports"

QUALITY_METRICS = ["logit_cosine_similarity", "hidden_cosine_similarity", "memory_ratio_vs_exact"]


def _metric_label(metric: str) -> str:
    labels = {
        "mse": "MSE",
        "bias": "Bias",
        "variance": "Variance",
        "mae": "MAE",
        "logit_cosine_similarity": "Logit Cosine",
        "hidden_cosine_similarity": "Hidden Cosine",
        "memory_ratio_vs_exact": "Memory / Exact",
        "prefill_seconds": "Prefill Seconds",
        "decode_seconds": "Decode Seconds",
        "peak_vram_mb": "Peak VRAM (MB)",
    }
    return labels.get(metric, metric)


def _mode_label(mode: str) -> str:
    return {
        "exact": "Exact",
        "key_only_random": "Key-Only (Random)",
        "key_only_block_so8_static": "Key-Only (SO8 Static)",
        "key_only_block_so8_learned": "Key-Only (SO8 Learned)",
        "protected_v": "Protected-V",
        "protected_v_lowrank": "Protected-V + LR",
        "full_kv": "Full-KV",
    }.get(mode, mode)


def render_attention_matplotlib(summary: pd.DataFrame, output_path: Path) -> None:
    quantized = summary.loc[
        summary["mode"].isin(
            [
                "key_only_random",
                "key_only_block_so8_static",
                "key_only_block_so8_learned",
                "protected_v",
                "protected_v_lowrank",
                "full_kv",
            ]
        )
    ].copy()
    metrics = ["logit_cosine_similarity", "hidden_cosine_similarity", "memory_ratio_vs_exact"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = quantized.loc[quantized["metric"] == metric].sort_values(["mode", "bits"])
        for mode in subset["mode"].unique():
            mode_frame = subset.loc[subset["mode"] == mode]
            axis.errorbar(
                mode_frame["bits"],
                mode_frame["mean"],
                yerr=mode_frame["sem"],
                marker="o",
                capsize=4,
                label=_mode_label(mode),
            )
        axis.set_title(_metric_label(metric))
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
        if metric == "memory_ratio_vs_exact":
            axis.set_ylabel("Ratio")
        else:
            axis.set_ylabel("Value")
    axes[0].legend()
    fig.suptitle("TurboQuant Replay Trade-offs")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_attention_plotly(summary: pd.DataFrame, output_path: Path) -> None:
    quantized = summary.loc[
        summary["mode"].isin(
            [
                "key_only_random",
                "key_only_block_so8_static",
                "key_only_block_so8_learned",
                "protected_v",
                "protected_v_lowrank",
                "full_kv",
            ]
        )
    ].copy()
    metrics = ["logit_cosine_similarity", "hidden_cosine_similarity", "memory_ratio_vs_exact"]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[_metric_label(metric) for metric in metrics])
    for col_idx, metric in enumerate(metrics, start=1):
        subset = quantized.loc[quantized["metric"] == metric].sort_values(["mode", "bits"])
        for mode in subset["mode"].unique():
            mode_frame = subset.loc[subset["mode"] == mode]
            fig.add_trace(
                go.Scatter(
                    x=mode_frame["bits"],
                    y=mode_frame["mean"],
                    error_y={"type": "data", "array": mode_frame["sem"], "visible": True},
                    mode="lines+markers",
                    name=f"{_mode_label(mode)}: {_metric_label(metric)}",
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(title="TurboQuant Replay Trade-offs (Interactive)", height=500, width=1500)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def render_runtime_matplotlib(summary: pd.DataFrame, output_path: Path) -> None:
    quantized = summary.loc[
        summary["mode"].isin(
            [
                "exact",
                "key_only_random",
                "key_only_block_so8_static",
                "key_only_block_so8_learned",
                "protected_v",
                "protected_v_lowrank",
                "full_kv",
            ]
        )
    ].copy()
    metrics = ["prefill_seconds", "decode_seconds", "peak_vram_mb"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = quantized.loc[quantized["metric"] == metric].sort_values(["mode", "bits"])
        for mode in subset["mode"].unique():
            mode_frame = subset.loc[subset["mode"] == mode]
            axis.errorbar(
                mode_frame["bits"].fillna(0.0),
                mode_frame["mean"],
                yerr=mode_frame["sem"],
                marker="o",
                capsize=4,
                label=_mode_label(mode),
            )
        axis.set_title(_metric_label(metric))
        axis.set_xlabel("Bits (exact=0)")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("TurboQuant Runtime Trade-offs")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_runtime_plotly(summary: pd.DataFrame, output_path: Path) -> None:
    quantized = summary.loc[
        summary["mode"].isin(
            [
                "exact",
                "key_only_random",
                "key_only_block_so8_static",
                "key_only_block_so8_learned",
                "protected_v",
                "protected_v_lowrank",
                "full_kv",
            ]
        )
    ].copy()
    metrics = ["prefill_seconds", "decode_seconds", "peak_vram_mb"]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[_metric_label(metric) for metric in metrics])
    for col_idx, metric in enumerate(metrics, start=1):
        subset = quantized.loc[quantized["metric"] == metric].sort_values(["mode", "bits"])
        for mode in subset["mode"].unique():
            mode_frame = subset.loc[subset["mode"] == mode]
            fig.add_trace(
                go.Scatter(
                    x=mode_frame["bits"].fillna(0.0),
                    y=mode_frame["mean"],
                    error_y={"type": "data", "array": mode_frame["sem"], "visible": True},
                    mode="lines+markers",
                    name=f"{_mode_label(mode)}: {_metric_label(metric)}",
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(title="TurboQuant Runtime Trade-offs (Interactive)", height=500, width=1500)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def render_synthetic_matplotlib(summary: pd.DataFrame, output_path: Path) -> None:
    experiments = ["mse", "prod", "prod_mixed"]
    fig, axes = plt.subplots(1, len(experiments), figsize=(18, 5), constrained_layout=True)
    for axis, experiment in zip(axes, experiments, strict=True):
        subset = summary.loc[summary["experiment"] == experiment].copy()
        for metric in subset["metric"].unique():
            metric_frame = subset.loc[subset["metric"] == metric].sort_values("bits")
            axis.errorbar(
                metric_frame["bits"],
                metric_frame["mean"],
                yerr=metric_frame["sem"],
                marker="o",
                capsize=4,
                label=_metric_label(metric),
            )
        axis.set_title(experiment.upper())
        axis.set_xlabel("Bits")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("TurboQuant Synthetic Core Validation")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def runtime_bottleneck_line(attention_summary: pd.DataFrame) -> str:
    hidden = attention_summary.loc[
        (attention_summary["metric"] == "hidden_cosine_similarity")
        & (
            attention_summary["mode"].isin(
                [
                    "key_only_block_so8_learned",
                    "protected_v",
                    "protected_v_lowrank",
                    "full_kv",
                ]
            )
        )
        & (attention_summary["bit_setting"] != "exact")
    ].copy()
    if hidden.empty:
        return "Current mathematical bottleneck: insufficient replay evidence yet."
    pivot = hidden.pivot_table(index="bit_setting", columns="mode", values="mean", aggfunc="first")
    if not {"key_only_block_so8_learned", "full_kv"}.issubset(pivot.columns):
        return "Current mathematical bottleneck: insufficient replay evidence yet."
    gap = (pivot["key_only_block_so8_learned"] - pivot["full_kv"]).sort_values(ascending=False)
    if gap.empty:
        return "Current mathematical bottleneck: insufficient replay evidence yet."
    strongest = gap.index[0]
    extra_line = ""
    if "protected_v" in pivot.columns:
        protected_gap = pivot["protected_v"] - pivot["full_kv"]
        if protected_gap.notna().any():
            extra_line = f" Protected-V improves over full-KV by up to {float(protected_gap.max()):.4f} hidden cosine."
    return (
        "Current mathematical bottleneck: value quantization drives most of the downstream hidden-state drift. "
        f"At {strongest} bits, learned-SO(8) key-only exceeds full-KV by {float(gap.iloc[0]):.4f} hidden cosine."
        f"{extra_line}"
    )


def runtime_recommendation_line(attention_summary: pd.DataFrame) -> str:
    hidden = attention_summary.loc[
        (attention_summary["metric"] == "hidden_cosine_similarity")
        & (
            attention_summary["mode"].isin(
                [
                    "key_only_block_so8_learned",
                    "protected_v",
                    "protected_v_lowrank",
                    "full_kv",
                ]
            )
        )
        & (attention_summary["bit_setting"].isin(["2", "2.5", "3", "3.5", "4"]))
    ].copy()
    if hidden.empty:
        return "Runtime recommendation: keep runtime default as key-only."
    pivot = hidden.pivot_table(index="bit_setting", columns="mode", values="mean", aggfunc="first")
    if not {"key_only_block_so8_learned", "full_kv"}.issubset(pivot.columns):
        return "Runtime recommendation: keep runtime default as key-only."
    full_kv = pivot["full_kv"]
    key_only = pivot["key_only_block_so8_learned"]
    protected = pivot["protected_v"] if "protected_v" in pivot.columns else None
    lowrank = pivot["protected_v_lowrank"] if "protected_v_lowrank" in pivot.columns else None

    if lowrank is not None:
        improvement = lowrank - full_kv
        closeness = (key_only - lowrank).abs()
        baseline_gap = (key_only - full_kv).abs().clip(lower=1e-8)
        if bool(((improvement > 0.02) & (closeness <= 0.5 * baseline_gap)).any()):
            return "Runtime recommendation: protected-V + low-rank is competitive enough to consider a runtime branch."
    candidate = lowrank if lowrank is not None else protected
    if candidate is not None and bool(((candidate - full_kv) > 0.005).any()):
        return "Runtime recommendation: protected-V is promising but not ready."
    return "Runtime recommendation: keep runtime default as key-only."


def compact_table(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    metric_order = {metric: index for index, metric in enumerate(metrics)}
    mode_order = {
        "exact": 0,
        "key_only_random": 1,
        "key_only_block_so8_static": 2,
        "key_only_block_so8_learned": 3,
        "protected_v": 4,
        "protected_v_lowrank": 5,
        "full_kv": 6,
    }
    bit_order = {"exact": -1.0, "2": 2.0, "2.5": 2.5, "3": 3.0, "3.5": 3.5, "4": 4.0}
    subset = frame.loc[
        frame["metric"].isin(metrics)
        & frame["mode"].isin(
            [
                "exact",
                "key_only_random",
                "key_only_block_so8_static",
                "key_only_block_so8_learned",
                "protected_v",
                "protected_v_lowrank",
                "full_kv",
            ]
        )
        & frame["bit_setting"].isin(["exact", "2", "2.5", "3", "3.5", "4"]),
        ["mode", "bit_setting", "metric", "mean", "std", "sem", "ci95_low", "ci95_high"],
    ].copy()
    subset["metric_sort"] = subset["metric"].map(metric_order)
    subset["mode_sort"] = subset["mode"].map(mode_order)
    subset["bit_sort"] = subset["bit_setting"].map(bit_order)
    return subset.sort_values(["metric_sort", "mode_sort", "bit_sort"]).drop(columns=["metric_sort", "mode_sort", "bit_sort"])


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_markdown_summary(
    synthetic_summary: pd.DataFrame,
    attention_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    captured_summary: pd.DataFrame | None,
    captured_thresholds: pd.DataFrame | None,
    output_path: Path,
) -> None:
    bottleneck = runtime_bottleneck_line(attention_summary)
    synthetic_table = compact_table(
        attention_summary,
        metrics=[
            "memory_ratio_vs_exact",
            "hidden_cosine_similarity",
            "hidden_mse",
            "logit_cosine_similarity",
            "logit_top1_match",
            "logit_top5_overlap",
        ],
    )
    synthetic_runtime_table = compact_table(
        attention_summary,
        metrics=["prefill_seconds", "decode_seconds", "peak_vram_mb"],
    )
    captured_table = None
    captured_bottleneck = None
    captured_recommendation = None
    captured_runtime_table = None
    if captured_summary is not None and not captured_summary.empty:
        captured_table = compact_table(
            captured_summary,
            metrics=[
                "memory_ratio_vs_exact",
                "hidden_cosine_similarity",
                "hidden_mse",
                "logit_cosine_similarity",
                "logit_top1_match",
                "logit_top5_overlap",
            ],
        )
        captured_runtime_table = compact_table(
            captured_summary,
            metrics=["prefill_seconds", "decode_seconds", "peak_vram_mb"],
        )
        captured_bottleneck = runtime_bottleneck_line(captured_summary)
        captured_recommendation = runtime_recommendation_line(captured_summary)
    try:
        synthetic_text = synthetic_summary.to_markdown(index=False)
    except ImportError:
        synthetic_text = "```\n" + synthetic_summary.to_string(index=False) + "\n```"
    try:
        replay_text = synthetic_table.to_markdown(index=False)
    except ImportError:
        replay_text = "```\n" + synthetic_table.to_string(index=False) + "\n```"
    try:
        runtime_text = synthetic_runtime_table.to_markdown(index=False)
    except ImportError:
        runtime_text = "```\n" + synthetic_runtime_table.to_string(index=False) + "\n```"
    try:
        threshold_text = threshold_summary.to_markdown(index=False)
    except ImportError:
        threshold_text = "```\n" + threshold_summary.to_string(index=False) + "\n```"
    if captured_table is not None:
        try:
            captured_text = captured_table.to_markdown(index=False)
        except ImportError:
            captured_text = "```\n" + captured_table.to_string(index=False) + "\n```"
        try:
            captured_runtime_text = captured_runtime_table.to_markdown(index=False) if captured_runtime_table is not None else ""
        except ImportError:
            captured_runtime_text = "```\n" + captured_runtime_table.to_string(index=False) + "\n```" if captured_runtime_table is not None else ""
    else:
        captured_text = ""
        captured_runtime_text = ""
    if captured_thresholds is not None and not captured_thresholds.empty:
        try:
            captured_threshold_text = captured_thresholds.to_markdown(index=False)
        except ImportError:
            captured_threshold_text = "```\n" + captured_thresholds.to_string(index=False) + "\n```"
    else:
        captured_threshold_text = ""

    lines = [
        "# TurboQuant Report",
        "",
        "## Synthetic Replay",
        "",
        bottleneck,
        "",
        "### Synthetic Core Summary",
        "",
        synthetic_text,
        "",
        "### Synthetic Primary Pareto Table",
        "",
        replay_text,
        "",
        "### Synthetic Secondary Runtime Table",
        "",
        runtime_text,
        "",
        "### Synthetic First-Layer Thresholds",
        "",
        threshold_text,
        "",
    ]
    if captured_table is not None and captured_bottleneck is not None:
        lines.extend(
            [
                "## Captured Replay",
                "",
                captured_bottleneck,
                captured_recommendation or "Runtime recommendation: keep runtime default as key-only.",
                "",
                "### Captured Primary Pareto Table",
                "",
                captured_text,
                "",
            ]
        )
        if captured_runtime_text:
            lines.extend(
                [
                    "### Captured Secondary Runtime Table",
                    "",
                    captured_runtime_text,
                    "",
                ]
            )
        if captured_threshold_text:
            lines.extend(
                [
                    "### Captured First-Layer Thresholds",
                    "",
                    captured_threshold_text,
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(PLOTS_DIR)
    ensure_dir(REPORTS_DIR)

    synthetic_summary = pd.read_csv(METRICS_DIR / "synthetic_metrics.csv")
    attention_summary = load_optional_csv(METRICS_DIR / "attention_summary_synthetic.csv")
    if attention_summary is None:
        attention_summary = pd.read_csv(METRICS_DIR / "attention_summary.csv")
    threshold_summary = load_optional_csv(METRICS_DIR / "attention_thresholds_synthetic.csv")
    if threshold_summary is None:
        threshold_summary = pd.read_csv(METRICS_DIR / "attention_thresholds.csv")
    captured_summary = load_optional_csv(METRICS_DIR / "attention_summary_captured.csv")
    captured_thresholds = load_optional_csv(METRICS_DIR / "attention_thresholds_captured.csv")

    render_synthetic_matplotlib(synthetic_summary, PLOTS_DIR / "synthetic_errorbars.png")
    render_attention_matplotlib(attention_summary, PLOTS_DIR / "attention_tradeoffs.png")
    render_attention_plotly(attention_summary, PLOTS_DIR / "attention_tradeoffs.html")
    render_runtime_matplotlib(attention_summary, PLOTS_DIR / "attention_runtime_tradeoffs.png")
    render_runtime_plotly(attention_summary, PLOTS_DIR / "attention_runtime_tradeoffs.html")
    if captured_summary is not None and not captured_summary.empty:
        render_attention_matplotlib(captured_summary, PLOTS_DIR / "attention_tradeoffs_captured.png")
        render_attention_plotly(captured_summary, PLOTS_DIR / "attention_tradeoffs_captured.html")
        render_runtime_matplotlib(captured_summary, PLOTS_DIR / "attention_runtime_tradeoffs_captured.png")
        render_runtime_plotly(captured_summary, PLOTS_DIR / "attention_runtime_tradeoffs_captured.html")
    render_markdown_summary(
        synthetic_summary=synthetic_summary,
        attention_summary=attention_summary,
        threshold_summary=threshold_summary,
        captured_summary=captured_summary,
        captured_thresholds=captured_thresholds,
        output_path=REPORTS_DIR / "summary.md",
    )

    print(attention_summary)
    print(threshold_summary)
    if captured_summary is not None:
        print(captured_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
