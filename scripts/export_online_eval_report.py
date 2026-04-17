from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from turboquant.eval_stats import (
    compute_benchmark_pairwise_statistics,
    compute_continuous_pairwise_statistics,
    summarize_continuous_metrics,
)
from turboquant.io_utils import ensure_dir
from turboquant.reporting import markdown_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export combined online-eval plots and markdown summaries.")
    parser.add_argument("--hf-dir", default="artifacts/hf_online_eval", help="HF online eval artifact root.")
    parser.add_argument("--runtime-dir", default="artifacts/runtime_eval", help="Runtime eval artifact root.")
    parser.add_argument(
        "--replay-summary-csv",
        default="artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_summary.csv",
        help="Replay summary CSV used for memory/Pareto joins.",
    )
    parser.add_argument("--output-dir", default="artifacts/online_eval_report", help="Output directory.")
    return parser.parse_args(argv)


def _discover_metrics_dirs(root: Path) -> list[Path]:
    discovered: list[Path] = []
    direct = root / "metrics"
    if direct.exists():
        discovered.append(direct)
    for path in sorted(root.rglob("metrics")):
        if not path.is_dir() or path in discovered:
            continue
        discovered.append(path)
    return discovered


def _optional_csv_from_metrics_dirs(metrics_dirs: list[Path], filename: str) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for metrics_dir in metrics_dirs:
        path = metrics_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_hf_ppl_summary(metrics_dirs: list[Path]) -> pd.DataFrame:
    summary = _optional_csv_from_metrics_dirs(metrics_dirs, "hf_online_ppl_summary.csv")
    if summary is None:
        return pd.DataFrame()
    return summary.assign(backend="hf")


def _load_hf_benchmark_summary(metrics_dirs: list[Path]) -> pd.DataFrame:
    summary = _optional_csv_from_metrics_dirs(metrics_dirs, "hf_online_benchmark_summary.csv")
    if summary is None:
        return pd.DataFrame()
    return summary.assign(backend="hf", metric="accuracy")


def _load_runtime_ppl_summary(metrics_dirs: list[Path]) -> pd.DataFrame:
    chunks = _optional_csv_from_metrics_dirs(metrics_dirs, "runtime_ppl_chunks.csv")
    if chunks is None or chunks.empty:
        return pd.DataFrame()
    long_frame = pd.concat(
        [
            chunks[["backend", "mode", "chunk_id", "perplexity"]]
            .rename(columns={"chunk_id": "sample_id", "perplexity": "value"})
            .assign(metric="perplexity"),
            chunks[["backend", "mode", "chunk_id", "log_perplexity"]]
            .rename(columns={"chunk_id": "sample_id", "log_perplexity": "value"})
            .assign(metric="log_perplexity"),
        ],
        ignore_index=True,
    )
    return summarize_continuous_metrics(long_frame, group_columns=["backend", "mode", "metric"])


def _load_runtime_bench_summary(metrics_dirs: list[Path]) -> pd.DataFrame:
    samples = _optional_csv_from_metrics_dirs(metrics_dirs, "runtime_bench_samples.csv")
    if samples is None or samples.empty:
        return pd.DataFrame()
    return summarize_continuous_metrics(
        samples,
        group_columns=["backend", "mode", "test", "metric"],
    )


def _load_runtime_task_summary(metrics_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in ("lm_eval_mcq_summary.csv", "lm_eval_chat_summary.csv"):
        frame = _optional_csv_from_metrics_dirs(metrics_dirs, name)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_hf_ppl_chunks(metrics_dirs: list[Path]) -> pd.DataFrame:
    frame = _optional_csv_from_metrics_dirs(metrics_dirs, "hf_online_ppl_chunks.csv")
    if frame is None:
        return pd.DataFrame()
    return frame


def _load_runtime_ppl_chunks(metrics_dirs: list[Path]) -> pd.DataFrame:
    frame = _optional_csv_from_metrics_dirs(metrics_dirs, "runtime_ppl_chunks.csv")
    if frame is None:
        return pd.DataFrame()
    return frame


def _load_hf_benchmark_items(metrics_dirs: list[Path]) -> pd.DataFrame:
    frame = _optional_csv_from_metrics_dirs(metrics_dirs, "hf_online_benchmark_items.csv")
    if frame is None:
        return pd.DataFrame()
    return frame


def _load_runtime_benchmark_items(metrics_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in ("lm_eval_mcq_items.csv", "lm_eval_chat_items.csv"):
        frame = _optional_csv_from_metrics_dirs(metrics_dirs, name)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_macro_benchmark_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, str | float | int]] = []
    for keys, group in frame.groupby(["backend", "mode"], dropna=False, sort=True):
        backend, mode = keys
        values = group["mean"].astype(float)
        n = int(values.count())
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem = std / (n**0.5) if n > 1 else 0.0
        rows.append(
            {
                "backend": backend,
                "mode": mode,
                "metric": "benchmark_macro",
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": mean - (1.96 * sem),
                "ci95_high": mean + (1.96 * sem),
            }
        )
    return pd.DataFrame(rows)


def _render_errorbar_by_mode(frame: pd.DataFrame, *, title: str, y_label: str, output_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [f"{row['backend']}:{row['mode']}" for _, row in frame.iterrows()]
    ax.errorbar(
        labels,
        frame["mean"],
        yerr=frame["sem"].fillna(0.0),
        fmt="o",
        capsize=4,
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_benchmark_errorbars(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for mode in sorted(frame["mode"].astype(str).unique()):
        subset = frame.loc[frame["mode"].astype(str) == mode].sort_values("task")
        labels = [f"{row['backend']}:{row['task']}" for _, row in subset.iterrows()]
        ax.errorbar(labels, subset["mean"], yerr=subset["sem"].fillna(0.0), fmt="o", capsize=4, label=mode)
    ax.set_title("Online Benchmark Accuracy")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_replay_pareto(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    metrics = ["hidden_cosine_similarity", "logit_cosine_similarity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        metric_frame = frame.loc[frame["metric"] == metric].copy()
        memory_frame = frame.loc[frame["metric"] == "memory_ratio_vs_exact", ["mode", "bit_setting", "mean"]].rename(
            columns={"mean": "memory_ratio_vs_exact_mean"}
        )
        joined = metric_frame.merge(memory_frame, on=["mode", "bit_setting"], how="inner")
        axis.scatter(joined["memory_ratio_vs_exact_mean"], joined["mean"])
        axis.set_title(metric)
        axis.set_xlabel("Memory / Exact")
        axis.set_ylabel("Replay Mean")
        axis.grid(alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_memory_join(frame: pd.DataFrame, *, title: str, y_label: str, output_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(
        frame["memory_ratio_vs_exact_mean"],
        frame["mean"],
        yerr=frame["sem"].fillna(0.0),
        fmt="o",
        capsize=4,
    )
    for _, row in frame.iterrows():
        ax.annotate(str(row["mode"]), (row["memory_ratio_vs_exact_mean"], row["mean"]))
    ax.set_title(title)
    ax.set_xlabel("Memory / Exact")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_frame_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _pairwise_note(frame: pd.DataFrame, *, label: str, mode_count: int) -> str:
    if not frame.empty:
        return markdown_table(frame)
    if mode_count < 2:
        return f"_Pairwise not available for {label}: fewer than 2 claimable modes._"
    return f"_No {label} pairwise rows loaded._"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    reports_dir = ensure_dir(output_dir / "reports")

    hf_metric_dirs = _discover_metrics_dirs(Path(args.hf_dir))
    runtime_metric_dirs = _discover_metrics_dirs(Path(args.runtime_dir))

    replay_summary = pd.read_csv(Path(args.replay_summary_csv)) if Path(args.replay_summary_csv).exists() else None
    hf_ppl_summary = _load_hf_ppl_summary(hf_metric_dirs)
    hf_benchmark_summary = _load_hf_benchmark_summary(hf_metric_dirs)
    hf_ppl_chunks = _load_hf_ppl_chunks(hf_metric_dirs)
    hf_benchmark_items = _load_hf_benchmark_items(hf_metric_dirs)
    runtime_ppl_summary = _load_runtime_ppl_summary(runtime_metric_dirs)
    runtime_bench_summary = _load_runtime_bench_summary(runtime_metric_dirs)
    runtime_task_summary = _load_runtime_task_summary(runtime_metric_dirs)
    runtime_ppl_chunks = _load_runtime_ppl_chunks(runtime_metric_dirs)
    runtime_benchmark_items = _load_runtime_benchmark_items(runtime_metric_dirs)

    ppl_summary = pd.concat([hf_ppl_summary, runtime_ppl_summary], ignore_index=True) if not hf_ppl_summary.empty or not runtime_ppl_summary.empty else pd.DataFrame()
    benchmark_summary = pd.concat([hf_benchmark_summary, runtime_task_summary], ignore_index=True) if not hf_benchmark_summary.empty or not runtime_task_summary.empty else pd.DataFrame()
    macro_benchmark = _build_macro_benchmark_summary(benchmark_summary)
    ppl_pairwise = pd.DataFrame()
    benchmark_pairwise = pd.DataFrame()
    throughput_pairwise = pd.DataFrame()

    ppl_long_frames: list[pd.DataFrame] = []
    if not hf_ppl_chunks.empty:
        ppl_long_frames.extend(
            [
                hf_ppl_chunks[["backend", "mode", "chunk_id", "perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "perplexity": "value"})
                .assign(metric="perplexity", run_id="hf"),
                hf_ppl_chunks[["backend", "mode", "chunk_id", "log_perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "log_perplexity": "value"})
                .assign(metric="log_perplexity", run_id="hf"),
            ]
        )
    if not runtime_ppl_chunks.empty:
        ppl_long_frames.extend(
            [
                runtime_ppl_chunks[["backend", "mode", "run_id", "chunk_id", "perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "perplexity": "value"})
                .assign(metric="perplexity"),
                runtime_ppl_chunks[["backend", "mode", "run_id", "chunk_id", "log_perplexity"]]
                .rename(columns={"chunk_id": "sample_id", "log_perplexity": "value"})
                .assign(metric="log_perplexity"),
            ]
        )
    ppl_mode_count = 0
    if ppl_long_frames:
        ppl_long = pd.concat(ppl_long_frames, ignore_index=True)
        ppl_mode_count = int(ppl_long["mode"].astype(str).nunique())
    else:
        ppl_long = pd.DataFrame()
    if not ppl_long.empty and ppl_mode_count >= 2:
        ppl_pairwise = compute_continuous_pairwise_statistics(
            ppl_long,
            group_columns=["backend", "metric"],
            pairing_columns=["run_id", "sample_id"],
        )
    benchmark_items = pd.concat(
        [frame for frame in (hf_benchmark_items, runtime_benchmark_items) if not frame.empty],
        ignore_index=True,
    ) if not hf_benchmark_items.empty or not runtime_benchmark_items.empty else pd.DataFrame()
    benchmark_mode_count = int(benchmark_items["mode"].astype(str).nunique()) if not benchmark_items.empty else 0
    if not benchmark_items.empty and benchmark_mode_count >= 2:
        benchmark_pairwise = compute_benchmark_pairwise_statistics(
            benchmark_items,
            group_columns=["backend", "task"],
            pairing_columns=["doc_id"],
        )
    throughput_mode_count = 0
    if not runtime_bench_summary.empty:
        bench_samples = _optional_csv_from_metrics_dirs(runtime_metric_dirs, "runtime_bench_samples.csv")
        if bench_samples is not None and not bench_samples.empty:
            throughput_mode_count = int(bench_samples["mode"].astype(str).nunique())
        if bench_samples is not None and not bench_samples.empty and throughput_mode_count >= 2:
            throughput_pairwise = compute_continuous_pairwise_statistics(
                bench_samples,
                group_columns=["backend", "test", "metric"],
                pairing_columns=["run_id", "sample_group", "sample_idx"],
            )

    replay_memory = pd.DataFrame()
    if replay_summary is not None and not replay_summary.empty:
        replay_memory = replay_summary.loc[
            replay_summary["metric"] == "memory_ratio_vs_exact",
            ["mode", "bit_setting", "mean"],
        ].rename(columns={"mean": "memory_ratio_vs_exact_mean"})

    ppl_pareto = pd.DataFrame()
    if not replay_memory.empty and not ppl_summary.empty:
        ppl_exact = ppl_summary.loc[ppl_summary["metric"] == "perplexity"].copy()
        ppl_exact["bit_setting"] = ppl_exact.get("bit_setting", "4")
        ppl_pareto = ppl_exact.merge(replay_memory, on="mode", how="inner")

    benchmark_pareto = pd.DataFrame()
    if not replay_memory.empty and not macro_benchmark.empty:
        benchmark_pareto = macro_benchmark.merge(replay_memory, on="mode", how="inner")

    if replay_summary is not None and not replay_summary.empty:
        _render_replay_pareto(replay_summary, plots_dir / "replay_memory_vs_quality.png")
    if not ppl_summary.empty:
        _render_errorbar_by_mode(
            ppl_summary.loc[ppl_summary["metric"] == "perplexity"],
            title="Online Perplexity",
            y_label="Perplexity",
            output_path=plots_dir / "online_perplexity.png",
        )
    if not runtime_bench_summary.empty:
        _render_errorbar_by_mode(
            runtime_bench_summary.loc[runtime_bench_summary["metric"] == "tokens_per_second"],
            title="Runtime Throughput",
            y_label="Tokens / Second",
            output_path=plots_dir / "runtime_throughput.png",
        )
    if not benchmark_summary.empty:
        _render_benchmark_errorbars(benchmark_summary, plots_dir / "online_benchmark_accuracy.png")
    if not ppl_pareto.empty:
        _render_memory_join(
            ppl_pareto,
            title="Memory vs Perplexity",
            y_label="Perplexity",
            output_path=plots_dir / "pareto_memory_vs_ppl.png",
        )
    if not benchmark_pareto.empty:
        _render_memory_join(
            benchmark_pareto,
            title="Memory vs Benchmark Macro",
            y_label="Macro Accuracy",
            output_path=plots_dir / "pareto_memory_vs_benchmark.png",
        )
    _write_frame_csv(ppl_pairwise, metrics_dir / "online_eval_ppl_pairwise.csv")
    _write_frame_csv(benchmark_pairwise, metrics_dir / "online_eval_benchmark_pairwise.csv")
    _write_frame_csv(throughput_pairwise, metrics_dir / "online_eval_throughput_pairwise.csv")

    lines = [
        "# Online Eval Report",
        "",
        "## Replay Pareto",
        "",
        markdown_table(replay_summary.head(16)) if replay_summary is not None and not replay_summary.empty else "_No replay summary loaded._",
        "",
        "## Perplexity Summary",
        "",
        markdown_table(ppl_summary) if not ppl_summary.empty else "_No perplexity rows loaded._",
        "",
        "## Runtime Throughput Summary",
        "",
        markdown_table(runtime_bench_summary) if not runtime_bench_summary.empty else "_No runtime throughput rows loaded._",
        "",
        "## Benchmark Summary",
        "",
        markdown_table(benchmark_summary) if not benchmark_summary.empty else "_No benchmark rows loaded._",
        "",
        "## Benchmark Macro Summary",
        "",
        markdown_table(macro_benchmark) if not macro_benchmark.empty else "_No macro benchmark rows loaded._",
        "",
        "## Perplexity Pairwise",
        "",
        _pairwise_note(ppl_pairwise, label="perplexity", mode_count=ppl_mode_count),
        "",
        "## Throughput Pairwise",
        "",
        _pairwise_note(throughput_pairwise, label="throughput", mode_count=throughput_mode_count),
        "",
        "## Benchmark Pairwise",
        "",
        _pairwise_note(benchmark_pairwise, label="benchmark", mode_count=benchmark_mode_count),
        "",
    ]
    (reports_dir / "online_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
