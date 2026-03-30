"""Build Triality-focused figures from triality_summary_mean_pm_sd.csv (Pareto + deltas).

Reads aggregated mean/SD per mode and bit width, writes PNGs under plots/ and a small CSV
under metrics/ for README tables. Run from repo root:

    py -3 scripts\\plot_triality_advantage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from turboquant.io_utils import ensure_dir
from turboquant.research_extension import ROTATION_COMPARE_MODES

# Short labels for crowded legends (English).
MODE_SHORT: dict[str, str] = {
    "key_only_random": "rand",
    "key_only_block_so8_static": "SO8-static",
    "key_only_block_so8_learned": "SO8-learned",
    "key_only_block_so8_triality_vector": "Triality-vector",
    "key_only_block_so8_triality_plus": "Triality+",
    "key_only_block_so8_triality_minus": "Triality−",
    "full_kv": "full-KV",
}

# Distinct colors for ROTATION_COMPARE_MODES order.
MODE_COLORS: list[str] = [
    "#7f7f7f",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#d62728",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Triality advantage from mean±SD summary CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT
        / "artifacts/research_extension/triality_full_eval_prod_bf16/metrics/triality_summary_mean_pm_sd.csv",
        help="Path to triality_summary_mean_pm_sd.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/research_extension/triality_full_eval_prod_bf16/plots",
        help="Directory for output PNG files",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/research_extension/triality_full_eval_prod_bf16/metrics",
        help="Directory for triality_advantage_deltas.csv",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default="2,4,8",
        help="Comma-separated bit widths for highlight panels and bar charts",
    )
    return parser.parse_args()


def load_mean_pm_sd(path: Path) -> pd.DataFrame:
    """Load mean±SD table; fail loudly if required columns are missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    frame = pd.read_csv(path)
    required = {
        "mode",
        "bit_setting",
        "hidden_cosine_similarity_mean",
        "hidden_cosine_similarity_sd",
        "memory_ratio_vs_exact_mean",
        "memory_ratio_vs_exact_sd",
        "logit_cosine_similarity_mean",
        "logit_cosine_similarity_sd",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["bit_setting"] = frame["bit_setting"].astype(float)
    return frame


def _row(
    frame: pd.DataFrame,
    mode: str,
    bit: float,
) -> pd.Series | None:
    subset = frame.loc[(frame["mode"] == mode) & (frame["bit_setting"] == bit)]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_delta_table(frame: pd.DataFrame, bits: list[float]) -> pd.DataFrame:
    """Rows: per-bit deltas Triality-vector minus random and minus full-KV (hidden cosine mean)."""
    tri = "key_only_block_so8_triality_vector"
    rand = "key_only_random"
    full = "full_kv"
    rows: list[dict[str, float | str]] = []
    for bit in bits:
        r_tri = _row(frame, tri, bit)
        r_rand = _row(frame, rand, bit)
        r_full = _row(frame, full, bit)
        if r_tri is None or r_rand is None or r_full is None:
            continue
        h_tri = float(r_tri["hidden_cosine_similarity_mean"])
        h_rand = float(r_rand["hidden_cosine_similarity_mean"])
        h_full = float(r_full["hidden_cosine_similarity_mean"])
        rows.append(
            {
                "bit_setting": bit,
                "hidden_triality_vector": h_tri,
                "hidden_key_only_random": h_rand,
                "hidden_full_kv": h_full,
                "delta_hidden_tri_minus_rand": h_tri - h_rand,
                "delta_hidden_tri_minus_full_kv": h_tri - h_full,
                "memory_triality": float(r_tri["memory_ratio_vs_exact_mean"]),
                "memory_random": float(r_rand["memory_ratio_vs_exact_mean"]),
                "memory_full_kv": float(r_full["memory_ratio_vs_exact_mean"]),
            }
        )
    return pd.DataFrame(rows)


def plot_pareto_hidden_vs_memory(
    frame: pd.DataFrame,
    bits: list[float],
    output_path: Path,
) -> None:
    """One column per bit: x = memory ratio, y = hidden cosine (mean ± SD error bars)."""
    modes = list(ROTATION_COMPARE_MODES)
    n = len(bits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True, sharey=True)
    if n == 1:
        axes = np.array([axes])
    for ax, bit in zip(axes, bits, strict=True):
        for mode_idx, mode in enumerate(modes):
            row = _row(frame, mode, bit)
            if row is None:
                continue
            mx = float(row["memory_ratio_vs_exact_mean"])
            my = float(row["hidden_cosine_similarity_mean"])
            ex = float(row["memory_ratio_vs_exact_sd"])
            ey = float(row["hidden_cosine_similarity_sd"])
            ax.errorbar(
                mx,
                my,
                xerr=ex,
                yerr=ey,
                fmt="o",
                color=MODE_COLORS[mode_idx % len(MODE_COLORS)],
                capsize=3,
                label=MODE_SHORT.get(mode, mode),
                markersize=8,
            )
        ax.set_xlabel("Memory ratio vs exact (K+V footprint proxy)")
        ax.set_title(f"Bit width = {bit:g}")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel("Hidden cosine similarity (mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8)
    fig.suptitle(
        "Pareto-style view: hidden quality vs memory footprint by mode",
        fontsize=12,
        y=1.08,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_hidden_bars(
    frame: pd.DataFrame,
    bits: list[float],
    output_path: Path,
) -> None:
    """Grouped bars: hidden cosine mean ± SD for rand / Triality-vector / full-KV."""
    modes = ["key_only_random", "key_only_block_so8_triality_vector", "full_kv"]
    labels = [MODE_SHORT[m] for m in modes]
    x = np.arange(len(bits), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, mode in enumerate(modes):
        means: list[float] = []
        errs: list[float] = []
        for bit in bits:
            row = _row(frame, mode, bit)
            if row is None:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                means.append(float(row["hidden_cosine_similarity_mean"]))
                errs.append(float(row["hidden_cosine_similarity_sd"]))
        offset = (i - 1) * width
        colors_bar = ["#7f7f7f", "#2ca02c", "#d62728"]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=labels[i],
            color=colors_bar[i],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bits])
    ax.set_xlabel("Bit width")
    ax.set_ylabel("Hidden cosine similarity (mean ± SD)")
    ax.set_title("Triality-vector vs random key-only vs full-KV (hidden state alignment)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.85, 1.02)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_bars(delta_frame: pd.DataFrame, output_path: Path) -> None:
    """Two panels: small Δ vs random (same memory) vs large Δ vs full-KV (different compression)."""
    if delta_frame.empty:
        return
    bits = delta_frame["bit_setting"].tolist()
    x = np.arange(len(bits), dtype=float)
    fig, (ax_small, ax_large) = plt.subplots(2, 1, figsize=(8, 6.5), constrained_layout=True)
    d1 = delta_frame["delta_hidden_tri_minus_rand"].astype(float)
    d2 = delta_frame["delta_hidden_tri_minus_full_kv"].astype(float)
    ax_small.bar(x, d1, color="#2ca02c", label="Triality-vector − random key-only")
    ax_small.set_xticks(x)
    ax_small.set_xticklabels([f"{b:g}" for b in bits])
    ax_small.set_ylabel("Δ hidden cosine")
    ax_small.axhline(0.0, color="k", linewidth=0.6)
    ax_small.set_title(
        "Same memory band as random: mean hidden-cosine gap (Triality-vector − random)"
    )
    ax_small.grid(axis="y", alpha=0.3)

    ax_large.bar(x, d2, color="#d62728", label="Triality-vector − full-KV")
    ax_large.set_xticks(x)
    ax_large.set_xticklabels([f"{b:g}" for b in bits])
    ax_large.set_xlabel("Bit width")
    ax_large.set_ylabel("Δ hidden cosine")
    ax_large.axhline(0.0, color="k", linewidth=0.6)
    ax_large.set_title(
        "vs aggressive full-KV quantization: hidden-cosine gap (Triality-vector − full-KV)"
    )
    ax_large.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    bits = [float(b.strip()) for b in args.bits.split(",") if b.strip()]
    frame = load_mean_pm_sd(args.input_csv)
    plots_dir = ensure_dir(args.plots_dir)
    metrics_dir = ensure_dir(args.metrics_dir)

    delta_frame = build_delta_table(frame, bits)
    delta_path = metrics_dir / "triality_advantage_deltas.csv"
    delta_frame.to_csv(delta_path, index=False)

    tasks = [
        ("Pareto hidden vs memory", lambda: plot_pareto_hidden_vs_memory(frame, bits, plots_dir / "triality_advantage_pareto_hidden_memory.png")),
        ("Grouped hidden bars", lambda: plot_grouped_hidden_bars(frame, bits, plots_dir / "triality_advantage_grouped_hidden.png")),
        ("Delta bars", lambda: plot_delta_bars(delta_frame, plots_dir / "triality_advantage_delta_hidden.png")),
    ]
    for _label, fn in tqdm(tasks, desc="Saving Triality advantage figures", unit="fig"):
        fn()

    print(f"Wrote: {delta_path}")
    print(f"PNG: {plots_dir / 'triality_advantage_pareto_hidden_memory.png'}")
    print(f"PNG: {plots_dir / 'triality_advantage_grouped_hidden.png'}")
    print(f"PNG: {plots_dir / 'triality_advantage_delta_hidden.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
