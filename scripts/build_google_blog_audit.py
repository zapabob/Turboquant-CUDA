from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from turboquant.io_utils import ensure_dir


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(frame.iloc[row_idx][column]) for column in columns) + " |"
        for row_idx in range(len(frame))
    ]
    return "\n".join([header, separator, *rows])


def build_audit_table(summary_stats: pd.DataFrame) -> pd.DataFrame:
    def stat(mode: str, bits: float, metric: str) -> str:
        row = summary_stats.loc[
            (summary_stats["mode"] == mode) & (summary_stats["bits"] == bits),
            :,
        ].iloc[0]
        return f"{row[f'{metric}_mean']:.6f} +/- {row[f'{metric}_sd']:.6f}"

    rows = [
        {
            "google_blog_claim": "TurboQuant achieves high KV reduction with zero accuracy loss.",
            "paper_actual_guarantee": (
                "The theory is centered on MSE-optimal quantization and unbiased low-distortion "
                "inner-product estimation on rotated unit-sphere vectors; it does not prove hidden-state "
                "or token-output preservation for all runtimes."
            ),
            "qwen35_9b_observation": (
                "KV memory shrinks strongly, but full_kv loses hidden geometry relative to key_only_random. "
                f"At 2-bit: hidden cosine {stat('full_kv', 2.0, 'hidden_cosine_similarity')} vs "
                f"{stat('key_only_random', 2.0, 'hidden_cosine_similarity')}."
            ),
            "audit_verdict": "Partially supported: memory yes, accuracy no in this runtime."
        },
        {
            "google_blog_claim": "TurboQuant achieves optimal scoring performance while minimizing KV memory footprint.",
            "paper_actual_guarantee": (
                "The paper emphasizes dot-product distortion / recall style results and unbiased inner-product "
                "estimation, not a blanket guarantee on downstream hidden transport."
            ),
            "qwen35_9b_observation": (
                "Logit cosine is nearly identical between key_only_random and full_kv across bits, "
                "while hidden cosine and attention-output error separate sharply. "
                f"At 3.5-bit: logit cosine full_kv {stat('full_kv', 3.5, 'logit_cosine_similarity')} "
                f"vs key_only_random {stat('key_only_random', 3.5, 'logit_cosine_similarity')}."
            ),
            "audit_verdict": "Supported for score-like metrics, not supported for value transport."
        },
        {
            "google_blog_claim": "Perfect downstream results across needle-style benchmarks.",
            "paper_actual_guarantee": (
                "The benchmark claim is task-specific and tied to the tested models/datasets; it is not a "
                "runtime-agnostic theorem."
            ),
            "qwen35_9b_observation": (
                "Qwen3.5-9B captured replay shows that full_kv increases attention_output_relative_error at every bit. "
                f"At 4-bit: full_kv {stat('full_kv', 4.0, 'attention_output_relative_error')} vs "
                f"key_only_random {stat('key_only_random', 4.0, 'attention_output_relative_error')}."
            ),
            "audit_verdict": "Not portable as a general claim."
        },
        {
            "google_blog_claim": "The method incurs negligible runtime overhead and is easy to deploy.",
            "paper_actual_guarantee": (
                "The paper discusses efficient primitives, but deployment cost still depends on the runtime path "
                "and whether K and V can use the same codec without regressions."
            ),
            "qwen35_9b_observation": (
                "The main deployment risk here is not logit score quality but V-path fragility: "
                "full_kv saves more memory, yet the observed token-output transport signal degrades first."
            ),
            "audit_verdict": "Operationally incomplete: the V-path failure mode is omitted."
        },
    ]
    return pd.DataFrame(rows)


def build_summary_stats(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    metrics = [
        "memory_ratio_vs_exact",
        "logit_cosine_similarity",
        "hidden_cosine_similarity",
        "attention_output_relative_error",
        "next_logit_kl",
    ]
    for (mode, bits), group in trials.groupby(["mode", "bits"]):
        row: dict[str, float | str] = {"mode": str(mode), "bits": float(bits)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mode", "bits"]).reset_index(drop=True)


def render_english_plot(summary_stats: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axis_specs = [
        ("memory_ratio_vs_exact", "KV Cache Memory Ratio vs Exact"),
        ("logit_cosine_similarity", "Logit Cosine by Bit"),
        ("hidden_cosine_similarity", "Hidden Cosine by Bit"),
        ("attention_output_relative_error", "Value-Path Transport Error by Bit"),
    ]
    colors = {"key_only_random": "#1f77b4", "full_kv": "#d62728"}
    labels = {"key_only_random": "Key-only random", "full_kv": "Full-KV"}

    for axis, (metric, title) in zip(axes.flat, axis_specs, strict=True):
        for mode in ("key_only_random", "full_kv"):
            group = summary_stats.loc[summary_stats["mode"] == mode].copy()
            axis.errorbar(
                group["bits"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_sd"].fillna(0.0),
                marker="o",
                capsize=4,
                linewidth=2,
                color=colors[mode],
                label=labels[mode],
            )
        axis.set_title(title)
        axis.set_xlabel("Bit-width")
        axis.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 1].set_ylabel("Cosine similarity")
    axes[1, 0].set_ylabel("Cosine similarity")
    axes[1, 1].set_ylabel("Relative error")
    fig.suptitle(
        "TurboQuant paper-faithful audit on Qwen3.5-9B captured replay\n"
        "Logits stay stable, but the V path degrades under full-KV compression",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Google blog vs paper vs Qwen audit artifact.")
    parser.add_argument(
        "--trials-csv",
        default="artifacts/paper_baseline/qwen_captured_reported/metrics/attention_trials_captured.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper_baseline/google_blog_audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")

    trials = pd.read_csv(Path(args.trials_csv))
    trials = trials.loc[trials["mode"].isin(["key_only_random", "full_kv"])].copy()
    summary_stats = build_summary_stats(trials)
    audit_table = build_audit_table(summary_stats)

    summary_stats.to_csv(metrics_dir / "qwen_summary_stats.csv", index=False)
    audit_table.to_csv(metrics_dir / "google_blog_paper_qwen_audit.csv", index=False)
    (metrics_dir / "qwen_summary_stats.md").write_text(markdown_table(summary_stats), encoding="utf-8")
    (metrics_dir / "google_blog_paper_qwen_audit.md").write_text(markdown_table(audit_table), encoding="utf-8")

    render_english_plot(summary_stats, plots_dir / "google_blog_paper_qwen_audit.png")
    print(summary_stats.to_string(index=False))
    print(audit_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
