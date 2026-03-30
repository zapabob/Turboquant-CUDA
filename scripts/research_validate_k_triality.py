from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import threading
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
    ROTATION_COMPARE_MODES,
    TRIALITY_PROXY_VIEWS,
    ValueResearchConfig,
    bit_setting_sort_key,
    compute_friedman_rotation_mode_statistics,
    compute_pairwise_wilcoxon_rotation_modes,
    compute_triality_statistics,
    evaluate_triality_proxy_captured,
)
from turboquant.schema import build_research_turboquant_config, write_turboquant_config


ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "triality_k_only_eval"
# Paper-style baselines + static SO8 + learned SO8 + triality proxies + full-KV (plots / mean±SD / multi-group stats).
PLOT_MODES = ROTATION_COMPARE_MODES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_line(
    log_path: Path,
    message: str,
    *,
    also_stdout: bool = False,
    stream=None,
) -> None:
    """Append timestamped line to log file; optionally print with OSError guard (Windows consoles)."""
    ensure_dir(log_path.parent)
    line = f"[{_utc_now()}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if also_stdout:
        try:
            print(message, flush=True, file=stream or sys.stdout)
        except OSError:
            pass


class RunLogger:
    """File-backed eval log with safe stdout/stderr."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def line(self, message: str, *, also_stdout: bool = False, stream=None) -> None:
        _log_line(self.log_path, message, also_stdout=also_stdout, stream=stream)

    def stderr_line(self, message: str) -> None:
        _log_line(self.log_path, message, also_stdout=False)
        try:
            print(message, file=sys.stderr, flush=True)
        except OSError:
            pass


class RollingCheckpointManager:
    """Timer-driven rolling copies of eval_status.json and optional partial trial CSV into checkpoints/cp_*."""

    def __init__(
        self,
        *,
        checkpoints_root: Path,
        status_src: Path,
        partial_csv_src: Path | None,
        interval_seconds: float,
        n_slots: int,
    ) -> None:
        self.checkpoints_root = checkpoints_root
        self.status_src = status_src
        self.partial_csv_src = partial_csv_src
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.n_slots = max(1, int(n_slots))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._lock = threading.Lock()

    def _snapshot(self) -> None:
        with self._lock:
            slot = self._generation % self.n_slots
            gen = self._generation
            self._generation += 1
        root = ensure_dir(self.checkpoints_root)
        slot_dir = ensure_dir(root / f"cp_{slot}")
        if self.status_src.exists():
            shutil.copy2(self.status_src, slot_dir / "eval_status.json")
        if self.partial_csv_src is not None and self.partial_csv_src.exists():
            shutil.copy2(self.partial_csv_src, slot_dir / "triality_trials_partial.csv")
        meta = {
            "generation": gen,
            "slot": slot,
            "updated_at": _utc_now(),
        }
        (root / "rolling_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _run(self) -> None:
        while not self._stop.wait(timeout=self.interval_seconds):
            try:
                self._snapshot()
            except OSError:
                pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="triality-rolling-cp", daemon=True)
        self._thread.start()

    def stop(self, *, final_snapshot: bool = True) -> None:
        self._stop.set()
        if final_snapshot:
            try:
                self._snapshot()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None


class EvalCheckpoint:
    def __init__(self, output_dir: Path, run_logger: RunLogger):
        self.output_dir = output_dir
        self.run_logger = run_logger
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
        self.run_logger.line(f"[triality-eval] start={stage}", also_stdout=True)

    def complete(self, stage: str) -> None:
        elapsed = time.perf_counter() - self._stage_started
        self.last_completed_stage = stage
        self.current_stage = stage
        self.write_status()
        self.run_logger.line(f"[triality-eval] done={stage} elapsed_s={elapsed:.3f}", also_stdout=True)

    def fail(self, stage: str, exc: BaseException) -> None:
        self.failed_stage = stage
        self.current_stage = stage
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.traceback_text = traceback.format_exc()
        self.write_status()
        head = f"[triality-eval] failed={stage} error={self.error_type}: {self.error_message}"
        self.run_logger.line(head, also_stdout=False)
        if self.traceback_text:
            self.run_logger.line(self.traceback_text.rstrip(), also_stdout=False)
        self.run_logger.stderr_line(head)
        if self.traceback_text:
            self.run_logger.stderr_line(self.traceback_text.rstrip())


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
    bit_settings = sorted(
        summary_frame["bit_setting"].astype(str).unique().tolist(),
        key=bit_setting_sort_key,
    )
    rows: list[dict[str, str | float]] = []
    for mode in PLOT_MODES:
        for bit_setting in bit_settings:
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


def summarize_from_trials(
    trial_frame: pd.DataFrame, *, skip_statistics: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    if skip_statistics:
        empty = pd.DataFrame()
        return summary_frame, mean_pm_sd, empty, empty, empty
    stats_frame = compute_triality_statistics(trial_frame)
    friedman_frame = compute_friedman_rotation_mode_statistics(trial_frame)
    pairwise_frame = compute_pairwise_wilcoxon_rotation_modes(trial_frame)
    return summary_frame, mean_pm_sd, stats_frame, friedman_frame, pairwise_frame


def write_eval_outputs(
    *,
    metrics_dir: Path,
    trial_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    mean_pm_sd: pd.DataFrame,
    stats_frame: pd.DataFrame | None,
    friedman_frame: pd.DataFrame | None = None,
    pairwise_frame: pd.DataFrame | None = None,
) -> dict[str, Path]:
    trial_path = metrics_dir / "triality_trials_captured.csv"
    summary_path = metrics_dir / "triality_summary_captured.csv"
    mean_sd_path = metrics_dir / "triality_summary_mean_pm_sd.csv"
    stats_path = metrics_dir / "triality_statistics.csv"
    friedman_path = metrics_dir / "triality_friedman_rotation_modes.csv"
    pairwise_path = metrics_dir / "triality_pairwise_wilcoxon_rotation_modes.csv"
    trial_frame.to_csv(trial_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    mean_pm_sd.to_csv(mean_sd_path, index=False)
    (metrics_dir / "triality_summary_captured.md").write_text(markdown_table(summary_frame), encoding="utf-8")
    (metrics_dir / "triality_summary_mean_pm_sd.md").write_text(markdown_table(mean_pm_sd), encoding="utf-8")
    if stats_frame is not None:
        stats_frame.to_csv(stats_path, index=False)
        (metrics_dir / "triality_statistics.md").write_text(markdown_table(stats_frame), encoding="utf-8")
    if friedman_frame is not None and not friedman_frame.empty:
        friedman_frame.to_csv(friedman_path, index=False)
        (metrics_dir / "triality_friedman_rotation_modes.md").write_text(
            markdown_table(friedman_frame), encoding="utf-8"
        )
    if pairwise_frame is not None and not pairwise_frame.empty:
        pairwise_frame.to_csv(pairwise_path, index=False)
        (metrics_dir / "triality_pairwise_wilcoxon_rotation_modes.md").write_text(
            markdown_table(pairwise_frame), encoding="utf-8"
        )
    return {
        "trial_csv": trial_path,
        "summary_csv": summary_path,
        "mean_pm_sd_csv": mean_sd_path,
        "statistics_csv": stats_path,
        "friedman_csv": friedman_path,
        "pairwise_csv": pairwise_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triality-proxy K-only modes on captured Qwen KV.")
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--rotation-dir", default="artifacts/research_extension/triality_k_only/rotations")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4,8")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--skip-statistics", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--from-existing-trials", default=None)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append timestamped eval log (default: <output-dir>/logs/eval_run.log).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from metrics/triality_trials_partial.csv + metrics/eval_resume_state.json when present.",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Ignore resume state and partial CSV even if --resume is set.",
    )
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=300.0,
        help="Rolling checkpoint interval (default: 300).",
    )
    parser.add_argument(
        "--rolling-slots",
        type=int,
        default=3,
        help="Number of checkpoint slots under checkpoints/cp_* (default: 3).",
    )
    return parser.parse_args()


def _cuda_requested(eval_device: str | object) -> bool:
    spec = str(eval_device).strip().lower()
    return spec == "cuda" or spec.startswith("cuda:")


def _fail_if_cuda_requested_but_unavailable(eval_device: str | object) -> int | None:
    """Return exit code if user asked for CUDA but this interpreter's Torch cannot use it."""
    if not _cuda_requested(eval_device):
        return None
    import torch

    if torch.cuda.is_available():
        return None
    msg = (
        "CUDA was requested (--eval-device cuda) but torch.cuda.is_available() is False.\n"
        "This often happens when you run `py -3` with a CPU-only PyTorch (e.g. Python 3.14 user site).\n"
        "This repo targets Python 3.12 + CUDA wheels via uv:\n"
        "  uv sync --extra cu128 --extra dev\n"
        "  uv run python scripts\\research_validate_k_triality.py ... --eval-device cuda\n"
        "Or use CPU explicitly:\n"
        "  ... --eval-device cpu\n"
    )
    print(msg, file=sys.stderr, end="")
    return 2


def main() -> int:
    args = parse_args()
    early = _fail_if_cuda_requested_but_unavailable(args.eval_device)
    if early is not None:
        return early
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    log_path = Path(args.log_file) if args.log_file else output_dir / "logs" / "eval_run.log"
    run_logger = RunLogger(log_path)
    checkpoint = EvalCheckpoint(output_dir, run_logger)
    partial_csv = metrics_dir / "triality_trials_partial.csv"
    rolling = RollingCheckpointManager(
        checkpoints_root=output_dir / "checkpoints",
        status_src=checkpoint.status_path,
        partial_csv_src=partial_csv,
        interval_seconds=args.checkpoint_interval_seconds,
        n_slots=args.rolling_slots,
    )
    rolling.start()

    try:
        checkpoint.start("replay_or_load_trials")
        friedman_frame: pd.DataFrame
        pairwise_frame: pd.DataFrame
        if args.from_existing_trials:
            trial_frame = pd.read_csv(args.from_existing_trials)
            summary_frame, mean_pm_sd, stats_frame, friedman_frame, pairwise_frame = summarize_from_trials(
                trial_frame, skip_statistics=args.skip_statistics
            )
        else:
            trial_frame, summary_frame = evaluate_triality_proxy_captured(
                kv_root=Path(args.kv_dir),
                trial_count=args.trials,
                bit_grid=bit_grid,
                rotation_dir=Path(args.rotation_dir),
                max_layers=args.max_layers,
                eval_device=args.eval_device,
                metrics_dir=metrics_dir,
                resume=args.resume,
                force_fresh=args.force_fresh,
            )
            if trial_frame.empty:
                raise ValueError("evaluate_triality_proxy_captured returned an empty trial frame")
            mean_pm_sd = build_mean_pm_sd_table(summary_frame)
            if args.skip_statistics:
                stats_frame = pd.DataFrame()
                friedman_frame = pd.DataFrame()
                pairwise_frame = pd.DataFrame()
            else:
                stats_frame = compute_triality_statistics(trial_frame)
                friedman_frame = compute_friedman_rotation_mode_statistics(trial_frame)
                pairwise_frame = compute_pairwise_wilcoxon_rotation_modes(trial_frame)
        checkpoint.complete("replay_or_load_trials")

        checkpoint.start("write_csv_md")
        output_paths = write_eval_outputs(
            metrics_dir=metrics_dir,
            trial_frame=trial_frame,
            summary_frame=summary_frame,
            mean_pm_sd=mean_pm_sd,
            stats_frame=None if args.skip_statistics else stats_frame,
            friedman_frame=None if args.skip_statistics else friedman_frame,
            pairwise_frame=None if args.skip_statistics else pairwise_frame,
        )
        checkpoint.complete("write_csv_md")

        if args.evaluate_only:
            checkpoint.start("finish")
            checkpoint.complete("finish")
            run_logger.line(str(summary_frame), also_stdout=True)
            if not args.skip_statistics:
                run_logger.line(str(stats_frame), also_stdout=True)
                run_logger.line(str(friedman_frame), also_stdout=True)
                run_logger.line(str(pairwise_frame), also_stdout=True)
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
                    "friedman_csv": str(output_paths["friedman_csv"]).replace("\\", "/"),
                    "pairwise_csv": str(output_paths["pairwise_csv"]).replace("\\", "/"),
                    "rotation_dir": str(Path(args.rotation_dir)).replace("\\", "/"),
                },
            )
            write_turboquant_config(config_out, payload)
            checkpoint.complete("config_write")

        checkpoint.start("finish")
        checkpoint.complete("finish")
        run_logger.line(str(summary_frame), also_stdout=True)
        if not args.skip_statistics:
            run_logger.line(str(stats_frame), also_stdout=True)
            run_logger.line(str(friedman_frame), also_stdout=True)
            run_logger.line(str(pairwise_frame), also_stdout=True)
        return 0
    except Exception as exc:
        checkpoint.fail(checkpoint.current_stage, exc)
        return 1
    finally:
        rolling.stop(final_snapshot=True)


if __name__ == "__main__":
    raise SystemExit(main())
