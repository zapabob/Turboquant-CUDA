from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import (
    QWEN_3060_MATRIX_MODES,
    QWEN_3060_PAIRWISE_BASELINES,
    QWEN_3060_STAT_METRICS,
    compute_qwen_3060_multigroup_statistics,
    evaluate_qwen_3060_matrix_rows,
    load_captured_runs,
    summarize_trial_metrics,
)
from turboquant.io_utils import ensure_dir
from turboquant.reporting import build_qwen_3060_mean_pm_sd_table, markdown_table, write_qwen_3060_markdown_summary
from turboquant.research_extension.k_triality import load_triality_proxy_rotations
from turboquant.runtime import RTX3060_12GB_LANE


ARTIFACT_ROOT = Path("artifacts") / "qwen_3060_matrix"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 12GB-only Qwen RTX 3060 comparison matrix.")
    parser.add_argument("--kv-dir", default="artifacts/kv_rtx3060_qwen9b")
    parser.add_argument(
        "--rotation-dir",
        default="artifacts/research_extension/triality_full_train/rotations",
        help="Triality rotation directory used for key_only_block_so8_triality_vector.",
    )
    parser.add_argument("--bits", default="3,3.5,4")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=0, help="0 = all layers")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--skip-statistics", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--ms-regular-bits", type=int, default=2)
    parser.add_argument("--ms-outlier-bits", type=int, default=4)
    parser.add_argument("--ms-outlier-count", type=int, default=64)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attach_capture_metadata(row: dict[str, float | int | str], bundle) -> None:
    metadata = bundle.metadata
    row["model_name"] = metadata.model_name
    row["tokenizer_name"] = metadata.tokenizer_name
    row["prompt_hash"] = metadata.prompt_hash
    row["capture_id"] = metadata.capture_id or ""
    row["lane_name"] = metadata.lane_name or ""
    row["prompt_label"] = metadata.prompt_label or ""
    row["capture_dtype"] = metadata.dtype
    row["capture_device"] = metadata.device
    row["model_preset"] = metadata.model_preset or ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")

    bundles = load_captured_runs(Path(args.kv_dir))
    if not bundles:
        raise ValueError("No captured Qwen bundles were loaded")
    for bundle in bundles:
        lane_name = bundle.metadata.lane_name or ""
        if lane_name != RTX3060_12GB_LANE:
            raise ValueError(
                "validate_qwen_3060_matrix.py requires 12GB capture metadata. "
                f"Expected lane_name={RTX3060_12GB_LANE!r}, got {lane_name!r}."
            )

    rotation_dir = Path(args.rotation_dir)
    if not rotation_dir.is_dir():
        raise FileNotFoundError(f"Missing triality rotation directory: {rotation_dir}")
    triality_artifacts = load_triality_proxy_rotations(rotation_dir)
    multiscreen_allocation = ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=args.ms_regular_bits,
        outlier_bits=args.ms_outlier_bits,
        outlier_count=args.ms_outlier_count,
    )

    rows: list[dict[str, float | int | str]] = []
    for trial in range(args.trials):
        for bundle in bundles:
            if args.max_layers > 0 and bundle.layer_idx >= args.max_layers:
                continue
            for row in evaluate_qwen_3060_matrix_rows(
                bundle=bundle,
                trial=trial,
                bit_grid=bit_grid,
                eval_device=args.eval_device,
                triality_artifacts=triality_artifacts,
                multiscreen_allocation=multiscreen_allocation,
                rotation_dir=rotation_dir,
            ):
                _attach_capture_metadata(row, bundle)
                rows.append(row)

    trial_frame = pd.DataFrame(rows)
    summary_frame = summarize_trial_metrics(trial_frame)
    mean_pm_sd_frame = build_qwen_3060_mean_pm_sd_table(summary_frame)

    trial_csv = metrics_dir / "qwen_3060_matrix_trials.csv"
    summary_csv = metrics_dir / "qwen_3060_matrix_summary.csv"
    mean_pm_sd_csv = metrics_dir / "qwen_3060_matrix_mean_pm_sd.csv"
    trial_frame.to_csv(trial_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    mean_pm_sd_frame.to_csv(mean_pm_sd_csv, index=False)
    (metrics_dir / "qwen_3060_matrix_summary.md").write_text(markdown_table(summary_frame), encoding="utf-8")
    (metrics_dir / "qwen_3060_matrix_mean_pm_sd.md").write_text(markdown_table(mean_pm_sd_frame), encoding="utf-8")

    if args.skip_statistics:
        friedman_frame = pd.DataFrame()
        pairwise_frame = pd.DataFrame()
    else:
        friedman_frame, pairwise_frame = compute_qwen_3060_multigroup_statistics(trial_frame)
        friedman_frame.to_csv(metrics_dir / "qwen_3060_matrix_friedman.csv", index=False)
        pairwise_frame.to_csv(metrics_dir / "qwen_3060_matrix_pairwise.csv", index=False)
        (metrics_dir / "qwen_3060_matrix_friedman.md").write_text(markdown_table(friedman_frame), encoding="utf-8")
        (metrics_dir / "qwen_3060_matrix_pairwise.md").write_text(markdown_table(pairwise_frame), encoding="utf-8")

    write_qwen_3060_markdown_summary(
        summary_frame=summary_frame,
        mean_pm_sd_frame=mean_pm_sd_frame,
        friedman_frame=friedman_frame,
        pairwise_frame=pairwise_frame,
        output_path=output_dir / "qwen_3060_matrix_report.md",
    )

    run_meta = {
        "timestamp_utc": _utc_now(),
        "script": "validate_qwen_3060_matrix.py",
        "lane_name": RTX3060_12GB_LANE,
        "kv_dir": str(Path(args.kv_dir).resolve()),
        "rotation_dir": str(rotation_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "eval_device": args.eval_device,
        "bit_grid": bit_grid,
        "trials": args.trials,
        "max_layers": args.max_layers,
        "mode_count": len(QWEN_3060_MATRIX_MODES),
        "modes": list(QWEN_3060_MATRIX_MODES),
        "stat_metrics": list(QWEN_3060_STAT_METRICS),
        "pairwise_baselines": list(QWEN_3060_PAIRWISE_BASELINES),
        "trial_csv": str(trial_csv).replace("\\", "/"),
        "summary_csv": str(summary_csv).replace("\\", "/"),
        "mean_pm_sd_csv": str(mean_pm_sd_csv).replace("\\", "/"),
        "friedman_csv": str((metrics_dir / "qwen_3060_matrix_friedman.csv")).replace("\\", "/"),
        "pairwise_csv": str((metrics_dir / "qwen_3060_matrix_pairwise.csv")).replace("\\", "/"),
    }
    (metrics_dir / "qwen_3060_matrix_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(summary_frame.loc[summary_frame["metric"].isin(QWEN_3060_STAT_METRICS)])
    if not args.skip_statistics:
        print(friedman_frame)
        print(pairwise_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
