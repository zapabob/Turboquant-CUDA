from __future__ import annotations

from pathlib import Path

import pandas as pd

from turboquant.reporting import (
    build_qwen_3060_mean_pm_sd_table,
    summarize_metric_trials,
    write_qwen_3060_markdown_summary,
)


def test_summarize_metric_trials_produces_confidence_columns() -> None:
    frame = pd.DataFrame(
        [
            {"experiment": "mse", "bits": 2.0, "metric": "mse", "trial": 0, "value": 1.0},
            {"experiment": "mse", "bits": 2.0, "metric": "mse", "trial": 1, "value": 3.0},
        ]
    )
    summary = summarize_metric_trials(frame, group_columns=["experiment", "bits", "metric"])
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["mean"] == 2.0
    assert row["std"] > 0
    assert row["ci95_high"] >= row["mean"] >= row["ci95_low"]


def test_qwen_3060_markdown_summary_lists_required_modes(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {"mode": "exact", "bit_setting": "3", "metric": "logit_cosine_similarity", "mean": 1.0, "std": 0.0, "sem": 0.0},
            {"mode": "exact", "bit_setting": "3", "metric": "next_logit_kl", "mean": 0.0, "std": 0.0, "sem": 0.0},
            {"mode": "exact", "bit_setting": "3", "metric": "hidden_cosine_similarity", "mean": 1.0, "std": 0.0, "sem": 0.0},
            {"mode": "exact", "bit_setting": "3", "metric": "memory_ratio_vs_exact", "mean": 1.0, "std": 0.0, "sem": 0.0},
            {"mode": "asym_q8_turbo4", "bit_setting": "3", "metric": "logit_cosine_similarity", "mean": 0.99, "std": 0.01, "sem": 0.005},
            {"mode": "asym_q8_turbo4", "bit_setting": "3", "metric": "next_logit_kl", "mean": 0.01, "std": 0.001, "sem": 0.001},
            {"mode": "asym_q8_turbo4", "bit_setting": "3", "metric": "hidden_cosine_similarity", "mean": 0.98, "std": 0.01, "sem": 0.005},
            {"mode": "asym_q8_turbo4", "bit_setting": "3", "metric": "memory_ratio_vs_exact", "mean": 0.7, "std": 0.01, "sem": 0.005},
        ]
    )
    mean_pm_sd = build_qwen_3060_mean_pm_sd_table(summary)
    friedman = pd.DataFrame([{"metric": "logit_cosine_similarity", "bit_setting": "3", "test": "friedman", "n_blocks": 3, "n_modes": 7, "statistic": 12.0, "p_value": 0.01, "modes": "exact,asym_q8_turbo4"}])
    pairwise = pd.DataFrame([{"metric": "logit_cosine_similarity", "bit_setting": "3", "baseline_mode": "exact", "candidate_mode": "asym_q8_turbo4", "test": "wilcoxon_signed_rank", "n_pairs": 3, "statistic": 0.0, "p_value": 0.25, "baseline_mean": 1.0, "candidate_mean": 0.99, "delta_candidate_minus_baseline": -0.01, "p_value_holm": 0.25, "significant_0_05": False}])

    output = tmp_path / "qwen_3060_matrix_summary.md"
    write_qwen_3060_markdown_summary(
        summary_frame=summary,
        mean_pm_sd_frame=mean_pm_sd,
        friedman_frame=friedman,
        pairwise_frame=pairwise,
        output_path=output,
    )

    text = output.read_text(encoding="utf-8")
    assert "Qwen 3060 Matrix Report" in text
    assert "asym_q8_turbo4" in text
    assert "key_only_block_so8_triality_vector" in text
