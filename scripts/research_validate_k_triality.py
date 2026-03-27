from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from turboquant.io_utils import ensure_dir
from turboquant.research_extension import (
    KeyResearchConfig,
    TRIALITY_MODE_BY_VIEW,
    TRIALITY_PROXY_VIEWS,
    ValueResearchConfig,
    compute_triality_statistics,
    evaluate_triality_proxy_captured,
)
from turboquant.schema import build_research_turboquant_config, write_turboquant_config


ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "triality_k_only_eval"
PLOT_MODES = (
    "key_only_random",
    "key_only_block_so8_learned",
    "key_only_block_so8_triality_vector",
    "key_only_block_so8_triality_plus",
    "key_only_block_so8_triality_minus",
    "full_kv",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalCheckpoint:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.status_path = output_dir / "eval_status.json"
        self.started_at = _utc_now()
        self.current_stage = "initializing"
        self.last_completed_stage: str | None = None
        self.failed_stage: str | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None
        self.traceback_text: str | None = None
        self._stage_started = time.perf_counter()
        self.write_status()

    def write_status(self) -> None:
        payload = {
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "current_stage": self.current_stage,
            "last_completed_stage": self.last_completed_stage,
            "failed_stage": self.failed_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback_text,
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def start(self, stage: str) -> None:
        self.current_stage = stage
        self._stage_started = time.perf_counter()
        self.write_status()
        print(f"[triality-eval] start={stage}", flush=True)

    def complete(self, stage: str) -> None:
        elapsed = time.perf_counter() - self._stage_started
        self.last_completed_stage = stage
        self.current_stage = stage
        self.write_status()
        print(f"[triality-eval] done={stage} elapsed_s={elapsed:.3f}", flush=True)

    def fail(self, stage: str, exc: BaseException) -> None:
        self.failed_stage = stage
        self.current_stage = stage
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.traceback_text = traceback.format_exc()
        self.write_status()
        print(
            f"[triality-eval] failed={stage} error={self.error_type}: {self.error_message}",
            file=sys.stderr,
            flush=True,
        )
        print(self.traceback_text, file=sys.stderr, flush=True)


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


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


def build_mean_pm_sd_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "logit_cosine_similarity": "logit_cosine_mean_pm_sd",
        "hidden_cosine_similarity": "hidden_cosine_mean_pm_sd",
        "memory_ratio_vs_exact": "memory_ratio_mean_pm_sd",
    }
    rows: list[dict[str, str | float]] = []
    for mode in PLOT_MODES:
        for bit_setting in ("2", "2.5", "3", "3.5", "4"):
            subset = summary_frame.loc[
                (summary_frame["mode"] == mode) & (summary_frame["bit_setting"].astype(str) == bit_setting)
            ]
            if subset.empty:
                continue
            record: dict[str, str | float] = {"mode": mode, "bit_setting": bit_setting}
            for metric, label in metrics.items():
                metric_row = subset.loc[subset["metric"] == metric]
                if metric_row.empty:
                    continue
                mean_value = float(metric_row["mean"].iloc[0])
                sd_value = float(metric_row["std"].iloc[0])
                record[label] = f"{mean_value:.6f} +/- {sd_value:.6f}"
                record[f"{metric}_mean"] = mean_value
                record[f"{metric}_sd"] = sd_value
            rows.append(record)
    return pd.DataFrame(rows)


def render_mean_pm_sd_plot(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    metric_pairs = [
        ("logit_cosine_similarity_mean", "logit_cosine_similarity_sd", "Logit Cosine"),
        ("hidden_cosine_similarity_mean", "hidden_cosine_similarity_sd", "Hidden Cosine"),
        ("memory_ratio_vs_exact_mean", "memory_ratio_vs_exact_sd", "Memory Ratio"),
    ]
    for axis, (mean_col, sd_col, title) in zip(axes, metric_pairs, strict=True):
        for mode in PLOT_MODES:
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
        axis.set_title(f"{title} by bit (mean +/- SD)")
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_triality_attention_matplotlib(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["logit_cosine_similarity", "hidden_cosine_similarity", "memory_ratio_vs_exact"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = summary.loc[
            (summary["metric"] == metric) & (summary["mode"].isin(PLOT_MODES))
        ].sort_values(["mode", "bits"])
        for mode in PLOT_MODES:
            mode_frame = subset.loc[subset["mode"] == mode]
            if mode_frame.empty:
                continue
            axis.errorbar(
                mode_frame["bits"],
                mode_frame["mean"],
                yerr=mode_frame["sem"],
                marker="o",
                capsize=4,
                label=mode,
            )
        axis.set_title(metric)
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_triality_runtime_matplotlib(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["prefill_seconds", "decode_seconds", "peak_vram_mb"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = summary.loc[
            (summary["metric"] == metric) & (summary["mode"].isin(PLOT_MODES))
        ].sort_values(["mode", "bits"])
        for mode in PLOT_MODES:
            mode_frame = subset.loc[subset["mode"] == mode]
            if mode_frame.empty:
                continue
            axis.errorbar(
                mode_frame["bits"].fillna(0.0),
                mode_frame["mean"],
                yerr=mode_frame["sem"],
                marker="o",
                capsize=4,
                label=mode,
            )
        axis.set_title(metric)
        axis.set_xlabel("Bits")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_from_trials(trial_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        trial_frame,
        {
            "dataset",
            "mode",
            "bit_setting",
            "bits",
            "logit_cosine_similarity",
            "hidden_cosine_similarity",
            "memory_ratio_vs_exact",
        },
        "trial_frame",
    )
    from turboquant.analysis import summarize_trial_metrics

    summary_frame = summarize_trial_metrics(trial_frame)
    mean_pm_sd = build_mean_pm_sd_table(summary_frame)
    stats_frame = compute_triality_statistics(trial_frame)
    return summary_frame, mean_pm_sd, stats_frame


def write_eval_outputs(
    *,
    metrics_dir: Path,
    trial_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    mean_pm_sd: pd.DataFrame,
    stats_frame: pd.DataFrame | None,
) -> dict[str, Path]:
    trial_path = metrics_dir / "triality_trials_captured.csv"
    summary_path = metrics_dir / "triality_summary_captured.csv"
    mean_sd_path = metrics_dir / "triality_summary_mean_pm_sd.csv"
    stats_path = metrics_dir / "triality_statistics.csv"
    trial_frame.to_csv(trial_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    mean_pm_sd.to_csv(mean_sd_path, index=False)
    (metrics_dir / "triality_summary_captured.md").write_text(markdown_table(summary_frame), encoding="utf-8")
    (metrics_dir / "triality_summary_mean_pm_sd.md").write_text(markdown_table(mean_pm_sd), encoding="utf-8")
    if stats_frame is not None:
        stats_frame.to_csv(stats_path, index=False)
        (metrics_dir / "triality_statistics.md").write_text(markdown_table(stats_frame), encoding="utf-8")
    return {
        "trial_csv": trial_path,
        "summary_csv": summary_path,
        "mean_pm_sd_csv": mean_sd_path,
        "statistics_csv": stats_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triality-proxy K-only modes on captured Qwen KV.")
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--rotation-dir", default="artifacts/research_extension/triality_k_only/rotations")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--skip-statistics", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--from-existing-trials", default=None)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    checkpoint = EvalCheckpoint(output_dir)

    try:
        checkpoint.start("replay_or_load_trials")
        if args.from_existing_trials:
            trial_frame = pd.read_csv(args.from_existing_trials)
            summary_frame, mean_pm_sd, stats_frame = summarize_from_trials(trial_frame)
        else:
            trial_frame, summary_frame = evaluate_triality_proxy_captured(
                kv_root=Path(args.kv_dir),
                trial_count=args.trials,
                bit_grid=bit_grid,
                rotation_dir=Path(args.rotation_dir),
                max_layers=args.max_layers,
                eval_device=args.eval_device,
            )
            if trial_frame.empty:
                raise ValueError("evaluate_triality_proxy_captured returned an empty trial frame")
            mean_pm_sd = build_mean_pm_sd_table(summary_frame)
            stats_frame = pd.DataFrame() if args.skip_statistics else compute_triality_statistics(trial_frame)
        checkpoint.complete("replay_or_load_trials")

        checkpoint.start("write_csv_md")
        output_paths = write_eval_outputs(
            metrics_dir=metrics_dir,
            trial_frame=trial_frame,
            summary_frame=summary_frame,
            mean_pm_sd=mean_pm_sd,
            stats_frame=None if args.skip_statistics else stats_frame,
        )
        checkpoint.complete("write_csv_md")

        if args.evaluate_only:
            checkpoint.start("finish")
            checkpoint.complete("finish")
            print(summary_frame)
            if not args.skip_statistics:
                print(stats_frame)
            return 0

        if not args.skip_plots:
            checkpoint.start("plot_generation")
            filtered_summary = summary_frame.loc[summary_frame["mode"].isin(PLOT_MODES)].copy()
            render_triality_attention_matplotlib(filtered_summary, plots_dir / "triality_attention_tradeoffs_captured.png")
            render_triality_runtime_matplotlib(filtered_summary, plots_dir / "triality_runtime_tradeoffs_captured.png")
            render_mean_pm_sd_plot(mean_pm_sd, plots_dir / "triality_mean_pm_sd_captured.png")
            checkpoint.complete("plot_generation")

        if args.write_config and not args.skip_config:
            checkpoint.start("config_write")
            config_out = Path(args.config_out) if args.config_out else output_dir / "turboquant_config.research.json"
            payload = build_research_turboquant_config(
                key_config=KeyResearchConfig(head_dim=128, views=TRIALITY_PROXY_VIEWS),
                value_config=ValueResearchConfig(),
                artifact_refs={
                    "trial_csv": str(output_paths["trial_csv"]).replace("\\", "/"),
                    "summary_csv": str(output_paths["summary_csv"]).replace("\\", "/"),
                    "mean_pm_sd_csv": str(output_paths["mean_pm_sd_csv"]).replace("\\", "/"),
                    "statistics_csv": str(output_paths["statistics_csv"]).replace("\\", "/"),
                    "rotation_dir": str(Path(args.rotation_dir)).replace("\\", "/"),
                },
            )
            write_turboquant_config(config_out, payload)
            checkpoint.complete("config_write")

        checkpoint.start("finish")
        checkpoint.complete("finish")
        print(summary_frame)
        if not args.skip_statistics:
            print(stats_frame)
        return 0
    except Exception as exc:
        checkpoint.fail(checkpoint.current_stage, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
