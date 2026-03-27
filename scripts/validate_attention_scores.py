from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from tqdm.auto import tqdm

from turboquant.analysis import (
    evaluate_layer_grid,
    load_captured_runs,
    melt_metric_rows,
    summarize_layer_thresholds,
    summarize_trial_metrics,
    synthetic_kv,
)
from turboquant.io_utils import ensure_dir
from turboquant.runtime import DEFAULT_MODEL_ID, require_supported_python


LOGGER = logging.getLogger("turboquant.validate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TurboQuant attention scores and hidden-state drift.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--output-dir", default="artifacts/metrics")
    parser.add_argument("--query-source", choices=["synthetic", "captured"], default="synthetic")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--synthetic-layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4")
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def parse_bit_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one bit-width")
    return values


def resolve_eval_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def total_mode_steps(bit_grid: list[float]) -> int:
    return 1 + (len(bit_grid) * 6)


def synthetic_rows(args: argparse.Namespace, bit_grid: list[float]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    eval_device = resolve_eval_device(args.eval_device)
    progress = tqdm(
        total=args.trials * args.synthetic_layers * total_mode_steps(bit_grid),
        desc="synthetic replay",
        unit="mode",
        dynamic_ncols=True,
    )

    def on_step(row: dict[str, float | int | str]) -> None:
        progress.set_postfix_str(
            f"trial={row['trial']} layer={row['layer']} mode={row['mode']} bits={row['bit_setting']}"
        )
        progress.update(1)

    for trial in range(args.trials):
        for layer_idx in range(args.synthetic_layers):
            LOGGER.info(
                "synthetic trial=%s layer=%s device=%s",
                trial,
                layer_idx,
                eval_device,
            )
            keys, values = synthetic_kv(
                seed=5_000 + (trial * 257) + layer_idx,
                batch=args.batch,
                heads=args.heads,
                seq_len=args.seq_len,
                dim=args.head_dim,
            )
            rows.extend(
                evaluate_layer_grid(
                    dataset="synthetic",
                    keys=keys,
                    values=values,
                    trial=trial,
                    layer_idx=layer_idx,
                    bit_grid=bit_grid,
                    eval_device=eval_device,
                    progress_callback=on_step,
                )
            )
    progress.close()
    return rows


def captured_rows(args: argparse.Namespace, bit_grid: list[float]) -> tuple[list[dict[str, float | int | str]], str]:
    kv_dir = Path(args.kv_dir)
    bundles = load_captured_runs(kv_dir)
    rows: list[dict[str, float | int | str]] = []
    eval_device = resolve_eval_device(args.eval_device)
    selected_bundles = bundles
    if args.max_layers > 0:
        per_capture: dict[str, list] = {}
        for bundle in bundles:
            capture_id = bundle.metadata.capture_id or bundle.capture_dir.name
            per_capture.setdefault(capture_id, []).append(bundle)
        selected_bundles = []
        for capture_id in sorted(per_capture):
            selected_bundles.extend(per_capture[capture_id][: args.max_layers])
    model_name = selected_bundles[0].metadata.model_name
    progress = tqdm(
        total=args.trials * len(selected_bundles) * total_mode_steps(bit_grid),
        desc="captured replay",
        unit="mode",
        dynamic_ncols=True,
    )

    def on_step(row: dict[str, float | int | str]) -> None:
        progress.set_postfix_str(
            "prompt="
            f"{row.get('prompt_label', 'captured')} "
            f"layer={row['layer']} mode={row['mode']} bits={row['bit_setting']}"
        )
        progress.update(1)

    for trial in range(args.trials):
        for bundle in selected_bundles:
            LOGGER.info(
                "captured trial=%s prompt=%s capture_id=%s layer=%s device=%s",
                trial,
                bundle.metadata.prompt_label or "unknown",
                bundle.metadata.capture_id or bundle.capture_dir.name,
                bundle.layer_idx,
                eval_device,
            )
            layer_rows = evaluate_layer_grid(
                dataset="captured",
                keys=bundle.keys,
                values=bundle.values,
                trial=trial,
                layer_idx=bundle.layer_idx,
                bit_grid=bit_grid,
                eval_device=eval_device,
                progress_callback=on_step,
            )
            for row in layer_rows:
                row["capture_id"] = bundle.metadata.capture_id or bundle.capture_dir.name
                row["prompt_label"] = bundle.metadata.prompt_label or "unknown"
                row["prompt_hash"] = bundle.metadata.prompt_hash
            rows.extend(layer_rows)
            LOGGER.info(
                "captured completed trial=%s prompt=%s layer=%s",
                trial,
                bundle.metadata.prompt_label or "unknown",
                bundle.layer_idx,
            )
    progress.close()
    return rows, model_name


def metric_table(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
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
        & frame["mode"].isin(list(mode_order))
        & frame["bit_setting"].isin(list(bit_order)),
        ["mode", "bit_setting", "metric", "mean", "std", "sem", "ci95_low", "ci95_high"],
    ].copy()
    subset["metric_sort"] = subset["metric"].map(metric_order)
    subset["mode_sort"] = subset["mode"].map(mode_order)
    subset["bit_sort"] = subset["bit_setting"].map(bit_order)
    subset = subset.sort_values(["metric_sort", "mode_sort", "bit_sort"]).drop(columns=["metric_sort", "mode_sort", "bit_sort"])
    return subset


def runtime_default_recommendation(summary_frame: pd.DataFrame, query_source: str) -> str:
    if query_source != "captured":
        return ""
    hidden_rows = summary_frame.loc[
        (summary_frame["metric"] == "hidden_cosine_similarity")
        & (summary_frame["bit_setting"].isin(["2", "2.5", "3", "3.5", "4"]))
    ].copy()
    if hidden_rows.empty:
        return "Runtime recommendation: keep runtime default as key-only."
    pivot = hidden_rows.pivot_table(index="bit_setting", columns="mode", values="mean", aggfunc="first")
    required = {"key_only_block_so8_learned", "full_kv"}
    if not required.issubset(pivot.columns):
        return "Runtime recommendation: keep runtime default as key-only."
    key_only = pivot["key_only_block_so8_learned"]
    full_kv = pivot["full_kv"]
    lowrank = pivot["protected_v_lowrank"] if "protected_v_lowrank" in pivot.columns else None
    protected = pivot["protected_v"] if "protected_v" in pivot.columns else None

    if lowrank is not None:
        improvement = lowrank - full_kv
        closeness = (key_only - lowrank).abs()
        baseline_gap = (key_only - full_kv).abs().clip(lower=1e-8)
        if bool(((improvement > 0.02) & (closeness <= 0.5 * baseline_gap)).any()):
            return "Runtime recommendation: protected-V + low-rank is competitive enough to consider a runtime branch."
    if protected is not None or lowrank is not None:
        candidate = lowrank if lowrank is not None else protected
        if candidate is not None and bool(((candidate - full_kv) > 0.005).any()):
            return "Runtime recommendation: protected-V is promising but not ready."
    return "Runtime recommendation: keep runtime default as key-only."


def markdown_summary(
    *,
    summary_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    output_path: Path,
    model_id: str,
    query_source: str,
) -> None:
    quantized = summary_frame.loc[summary_frame["mode"] != "exact"].copy()
    hidden_rows = quantized.loc[quantized["metric"] == "hidden_cosine_similarity"]
    candidate_rows = hidden_rows.loc[hidden_rows["bit_setting"].isin(["4", "3.5", "3", "2.5", "2"])]
    bottleneck_line = "Current mathematical bottleneck: insufficient evidence yet."
    if not candidate_rows.empty:
        grouped = candidate_rows.pivot_table(
            index="bit_setting",
            columns="mode",
            values="mean",
            aggfunc="first",
        )
        if {"key_only_block_so8_learned", "full_kv"}.issubset(grouped.columns):
            gap = (grouped["key_only_block_so8_learned"] - grouped["full_kv"]).sort_values(ascending=False)
            if not gap.empty:
                top_bit = gap.index[0]
                top_gap = float(gap.iloc[0])
                bottleneck_line = (
                    "Current mathematical bottleneck: value quantization amplifies attention-output "
                    f"drift more than key quantization alone. At {top_bit} bits, full-KV trails "
                    f"block-SO(8) key-only by {top_gap:.4f} hidden-state cosine."
                )
    recommendation_line = runtime_default_recommendation(summary_frame, query_source)
    try:
        primary_table = metric_table(
            summary_frame,
            metrics=[
                "memory_ratio_vs_exact",
                "hidden_cosine_similarity",
                "hidden_mse",
                "logit_cosine_similarity",
                "logit_top1_match",
                "logit_top5_overlap",
            ],
        ).to_markdown(index=False)
    except ImportError:
        primary_table = "```\n" + metric_table(
            summary_frame,
            metrics=[
                "memory_ratio_vs_exact",
                "hidden_cosine_similarity",
                "hidden_mse",
                "logit_cosine_similarity",
                "logit_top1_match",
                "logit_top5_overlap",
            ],
        ).to_string(index=False) + "\n```"
    try:
        secondary_table = metric_table(
            summary_frame,
            metrics=[
                "prefill_seconds",
                "decode_seconds",
                "peak_vram_mb",
            ],
        ).to_markdown(index=False)
    except ImportError:
        secondary_table = "```\n" + metric_table(
            summary_frame,
            metrics=[
                "prefill_seconds",
                "decode_seconds",
                "peak_vram_mb",
            ],
        ).to_string(index=False) + "\n```"
    try:
        summary_table = summary_frame.to_markdown(index=False)
    except ImportError:
        summary_table = "```\n" + summary_frame.to_string(index=False) + "\n```"
    try:
        threshold_table = threshold_frame.to_markdown(index=False)
    except ImportError:
        threshold_table = "```\n" + threshold_frame.to_string(index=False) + "\n```"
    lines = [
        "# Attention Replay Summary",
        "",
        f"- Model: {model_id}",
        f"- Query source: {query_source}",
        "",
        bottleneck_line,
        recommendation_line,
        "",
        "## Primary Pareto Table",
        "",
        primary_table,
        "",
        "## Secondary Runtime Table",
        "",
        secondary_table,
        "",
        "## Summary Statistics",
        "",
        summary_table,
        "",
        "## First-Layer Thresholds",
        "",
        threshold_table,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    require_supported_python()
    args = parse_args()
    configure_logging(args.log_level)
    bit_grid = parse_bit_grid(args.bits)
    output_dir = ensure_dir(Path(args.output_dir))
    LOGGER.info(
        "starting validate_attention_scores query_source=%s eval_device=%s bits=%s",
        args.query_source,
        resolve_eval_device(args.eval_device),
        ",".join(f"{bit:g}" for bit in bit_grid),
    )

    if args.query_source == "synthetic":
        rows = synthetic_rows(args, bit_grid)
        model_id = args.model_id
    else:
        rows, model_id = captured_rows(args, bit_grid)

    trial_frame = pd.DataFrame(rows)
    trial_frame["model_id"] = model_id
    trial_frame["query_source"] = args.query_source
    trial_frame["eval_device"] = resolve_eval_device(args.eval_device)
    trial_frame.to_csv(output_dir / "attention_trials.csv", index=False)
    trial_frame.to_csv(output_dir / f"attention_trials_{args.query_source}.csv", index=False)

    long_frame = melt_metric_rows(trial_frame)
    long_frame["model_id"] = model_id
    long_frame["query_source"] = args.query_source
    long_frame.to_csv(output_dir / "attention_metrics_long.csv", index=False)
    long_frame.to_csv(output_dir / f"attention_metrics_long_{args.query_source}.csv", index=False)

    summary_frame = summarize_trial_metrics(trial_frame)
    summary_frame["model_id"] = model_id
    summary_frame["query_source"] = args.query_source
    summary_frame.to_csv(output_dir / "attention_summary.csv", index=False)
    summary_frame.to_csv(output_dir / f"attention_summary_{args.query_source}.csv", index=False)

    threshold_rows = [
        summarize_layer_thresholds(trial_frame, metric="hidden_cosine_similarity", threshold=0.99),
        summarize_layer_thresholds(trial_frame, metric="hidden_cosine_similarity", threshold=0.95),
    ]
    threshold_frame = pd.concat([frame for frame in threshold_rows if not frame.empty], ignore_index=True)
    if not threshold_frame.empty:
        threshold_frame["model_id"] = model_id
        threshold_frame["query_source"] = args.query_source
        threshold_frame.to_csv(output_dir / "attention_thresholds.csv", index=False)
        threshold_frame.to_csv(output_dir / f"attention_thresholds_{args.query_source}.csv", index=False)

    markdown_summary(
        summary_frame=summary_frame,
        threshold_frame=threshold_frame,
        output_path=output_dir / "attention_summary.md",
        model_id=model_id,
        query_source=args.query_source,
    )
    markdown_summary(
        summary_frame=summary_frame,
        threshold_frame=threshold_frame,
        output_path=output_dir / f"attention_summary_{args.query_source}.md",
        model_id=model_id,
        query_source=args.query_source,
    )
    print(summary_frame)
    if not threshold_frame.empty:
        print(threshold_frame)
    LOGGER.info("finished validate_attention_scores query_source=%s", args.query_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
