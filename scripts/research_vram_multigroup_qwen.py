"""Multi-group comparison of PyTorch TurboQuant (+ Multiscreen) KV replay on CUDA vs VRAM / memory ratio.

This measures **offline replay** on tensors captured from a Hugging Face safetensors model
(e.g. ``H:\\Qwen3.5-9B-official-hf`` via ``scripts/capture_qwen_kv.py``): ``peak_vram_mb`` and
``memory_ratio_vs_exact`` from the same evaluation path as ``research_validate_multiscreen_kv.py``.
It does **not** invoke llama.cpp / Hypura TurboQuant (use ``rust/hypura`` for that stack).

Outputs under ``artifacts/research_extension/vram_multigroup/`` (override with ``--output-dir``):
  - ``metrics/vram_trials.csv`` — all rows
  - ``metrics/vram_summary_by_mode.csv`` — mean ± SEM per mode
  - ``metrics/vram_run_meta.json``
  - ``plots/vram_peak_mb_by_mode.png`` — peak VRAM (mean ± SEM)
  - ``plots/kv_memory_ratio_by_mode.png`` — KV storage ratio vs exact (mean ± SEM)
  - ``plots/hidden_cosine_similarity_by_mode.png`` — attention-output cosine vs exact (mean ± SEM)
  - ``metrics/vram_summary_by_mode.md`` — Markdown table mirror of the summary CSV
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import load_captured_runs
from turboquant.io_utils import ensure_dir
from turboquant.research_extension.captured_kv_modes import (
    CAPTURED_KEY_EVAL_MODES,
    eval_captured_key_mode_row,
)
from turboquant.research_extension.k_triality import (
    DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR,
    MULTISCREEN_TRIALITY_VECTOR_MODE,
    load_triality_proxy_rotations,
)

_MODES_NEEDING_TRIALITY_ROTATIONS: frozenset[str] = frozenset(
    {"key_only_block_so8_triality_vector", MULTISCREEN_TRIALITY_VECTOR_MODE}
)

ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "vram_multigroup"

# Triality mode requires trained rotations; add it explicitly when artifacts exist.
DEFAULT_MODES = "exact,key_only_random,key_only_block_so8_static,multiscreen_relevance"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kv-dir",
        "--captured-dir",
        dest="kv_dir",
        default="artifacts/kv_4bit",
        help="Captured KV root (same as paper_validate --kv-dir).",
    )
    p.add_argument(
        "--model-id",
        default=r"H:\Qwen3.5-9B-official-hf",
        help="Recorded in metadata only (expected HF source for the capture).",
    )
    p.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help="Comma-separated modes; triality requires trained rotations and --rotation-dir if not default.",
    )
    p.add_argument("--bits", type=float, default=3.0)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--eval-device", default="cuda", help="Use cuda for VRAM statistics (cpu gives peak_vram_mb≈0).")
    p.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    p.add_argument(
        "--rotation-dir",
        default="",
        help="Triality *.pt dir; if empty and triality is in --modes, uses DEFAULT_PRODUCTION path.",
    )
    p.add_argument("--ms-regular-bits", type=int, default=2)
    p.add_argument("--ms-outlier-bits", type=int, default=4)
    p.add_argument("--ms-outlier-count", type=int, default=64)
    return p.parse_args()


def _parse_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    bad = [m for m in modes if m not in CAPTURED_KEY_EVAL_MODES]
    if bad:
        raise SystemExit(f"Unknown mode(s): {bad}. Allowed: {sorted(CAPTURED_KEY_EVAL_MODES)}")
    return modes


def _triality_path_for_modes(modes: list[str], rotation_arg: str) -> Path | None:
    if not _MODES_NEEDING_TRIALITY_ROTATIONS.intersection(modes):
        return None
    s = rotation_arg.strip() or DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR
    path = Path(s)
    if not path.is_dir():
        raise SystemExit(
            f"Triality mode requested but rotation dir missing: {path.resolve()}\n"
            "Train with research_train_k_triality.py or run_triality_full_pipeline.py, "
            f"or pass --rotation-dir (default would be {DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR})."
        )
    return path


def _sem(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def _build_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for mode, group in trials.groupby("mode", sort=True):
        rows.append(
            {
                "mode": mode,
                "n_rows": int(len(group)),
                "peak_vram_mb_mean": float(group["peak_vram_mb"].mean()),
                "peak_vram_mb_std": float(group["peak_vram_mb"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "peak_vram_mb_sem": _sem(group["peak_vram_mb"]),
                "memory_ratio_vs_exact_mean": float(group["memory_ratio_vs_exact"].mean()),
                "memory_ratio_vs_exact_std": float(group["memory_ratio_vs_exact"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "memory_ratio_vs_exact_sem": _sem(group["memory_ratio_vs_exact"]),
                "hidden_cosine_similarity_mean": float(group["hidden_cosine_similarity"].mean()),
                "hidden_cosine_similarity_std": float(group["hidden_cosine_similarity"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "hidden_cosine_similarity_sem": _sem(group["hidden_cosine_similarity"]),
            }
        )
    return pd.DataFrame(rows)


def _write_summary_markdown(summary: pd.DataFrame, path: Path) -> None:
    """Write a simple GitHub-flavored markdown table (no extra deps)."""

    cols = list(summary.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body_lines: list[str] = []
    for _, row in summary.iterrows():
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and not math.isnan(v):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |\n")
    path.write_text(
        "# VRAM multigroup summary\n\n"
        "Per-mode aggregates over all trial×layer rows; error bars in plots use SEM.\n\n"
        + header
        + sep
        + "".join(body_lines),
        encoding="utf-8",
    )


def _plot_bars(summary: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    order = list(summary["mode"])
    x = range(len(order))
    means = summary["peak_vram_mb_mean"].tolist()
    errs = summary["peak_vram_mb_sem"].tolist()
    mmeans = summary["memory_ratio_vs_exact_mean"].tolist()
    merrs = summary["memory_ratio_vs_exact_sem"].tolist()
    hmeans = summary["hidden_cosine_similarity_mean"].tolist()
    herrs = summary["hidden_cosine_similarity_sem"].tolist()

    fig_p, ax_p = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax_p.bar(x, means, yerr=errs, capsize=4, color="steelblue", ecolor="black", alpha=0.85)
    ax_p.set_xticks(list(x))
    ax_p.set_xticklabels(order, rotation=25, ha="right")
    ax_p.set_ylabel("Peak VRAM (MB)")
    ax_p.set_title("Peak GPU memory during KV replay by mode (mean ± SEM)")
    ax_p.grid(axis="y", alpha=0.3)
    fig_p.savefig(plots_dir / "vram_peak_mb_by_mode.png", dpi=160)
    plt.close(fig_p)

    fig_m, ax_m = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax_m.bar(x, mmeans, yerr=merrs, capsize=4, color="seagreen", ecolor="black", alpha=0.85)
    ax_m.set_xticks(list(x))
    ax_m.set_xticklabels(order, rotation=25, ha="right")
    ax_m.set_ylabel("Ratio vs exact KV storage")
    ax_m.set_title("KV memory ratio vs exact baseline (mean ± SEM; lower = smaller footprint)")
    ax_m.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="exact storage")
    ax_m.legend(loc="upper right", fontsize=8)
    ax_m.grid(axis="y", alpha=0.3)
    fig_m.savefig(plots_dir / "kv_memory_ratio_by_mode.png", dpi=160)
    plt.close(fig_m)

    fig_h, ax_h = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax_h.bar(x, hmeans, yerr=herrs, capsize=4, color="darkorchid", ecolor="black", alpha=0.85)
    ax_h.set_xticks(list(x))
    ax_h.set_xticklabels(order, rotation=25, ha="right")
    ax_h.set_ylabel("Cosine similarity")
    ax_h.set_ylim(0.0, 1.05)
    ax_h.set_title(
        "Hidden-state cosine vs exact baseline by mode (mean ± SEM over trials×layers; higher = better)"
    )
    ax_h.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_h.grid(axis="y", alpha=0.3)
    fig_h.savefig(plots_dir / "hidden_cosine_similarity_by_mode.png", dpi=160)
    plt.close(fig_h)


def main() -> int:
    args = parse_args()
    modes = _parse_modes(args.modes)
    kv_dir = Path(args.kv_dir)
    out = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(out / "metrics")
    plots_dir = ensure_dir(out / "plots")

    bundles = load_captured_runs(kv_dir)
    eval_device = torch.device(args.eval_device)
    if eval_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    rotation_path = _triality_path_for_modes(modes, args.rotation_dir)
    triality_artifacts = load_triality_proxy_rotations(rotation_path) if rotation_path else None

    ms_alloc = ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=args.ms_regular_bits,
        outlier_bits=args.ms_outlier_bits,
        outlier_count=args.ms_outlier_count,
    )

    rows: list[dict[str, float | int | str]] = []
    bit_value = float(args.bits)

    total_steps = len(modes) * args.trials * sum(
        1
        for b in bundles
        if args.max_layers <= 0 or b.layer_idx < args.max_layers
    )
    pbar = tqdm(total=total_steps, desc="vram-multigroup", unit="eval")

    for mode in modes:
        for trial in range(args.trials):
            for bundle in bundles:
                if args.max_layers > 0 and bundle.layer_idx >= args.max_layers:
                    continue
                if eval_device.type == "cuda":
                    torch.cuda.empty_cache()
                ms_use = (
                    ms_alloc
                    if mode in ("multiscreen_relevance", MULTISCREEN_TRIALITY_VECTOR_MODE)
                    else None
                )
                row = eval_captured_key_mode_row(
                    mode=mode,
                    bundle=bundle,
                    trial=trial,
                    bit_value=bit_value,
                    eval_device=eval_device,
                    rotation_dir=rotation_path,
                    triality_artifacts=triality_artifacts,
                    ms_alloc=ms_use,
                )
                row["eval_mode_tag"] = mode
                row["model_id_meta"] = args.model_id
                meta = bundle.metadata
                row["model_name"] = meta.model_name
                row["prompt_hash"] = meta.prompt_hash
                rows.append(row)
                pbar.update(1)
    pbar.close()

    trials = pd.DataFrame(rows)
    trial_path = metrics_dir / "vram_trials.csv"
    trials.to_csv(trial_path, index=False)

    summary = _build_summary(trials)
    summary_path = metrics_dir / "vram_summary_by_mode.csv"
    summary.to_csv(summary_path, index=False)

    summary_md_path = metrics_dir / "vram_summary_by_mode.md"
    _write_summary_markdown(summary, summary_md_path)

    _plot_bars(summary, plots_dir)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "research_vram_multigroup_qwen.py",
        "model_id": args.model_id,
        "kv_dir": str(kv_dir.resolve()),
        "modes": modes,
        "bits": bit_value,
        "trials": args.trials,
        "max_layers": args.max_layers,
        "eval_device": str(eval_device),
        "rotation_dir": str(rotation_path.resolve()) if rotation_path else None,
        "trials_csv": str(trial_path).replace("\\", "/"),
        "summary_csv": str(summary_path).replace("\\", "/"),
        "summary_md": str(summary_md_path).replace("\\", "/"),
        "plots": {
            "peak_vram_mb": str((plots_dir / "vram_peak_mb_by_mode.png").resolve()).replace("\\", "/"),
            "kv_memory_ratio": str((plots_dir / "kv_memory_ratio_by_mode.png").resolve()).replace("\\", "/"),
            "hidden_cosine": str((plots_dir / "hidden_cosine_similarity_by_mode.png").resolve()).replace(
                "\\", "/"
            ),
        },
        "note": "PyTorch TurboQuant replay metrics; not llama.cpp Hypura build.",
    }
    (metrics_dir / "vram_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"wrote {trial_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {summary_md_path}")
    print(f"plots -> {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
