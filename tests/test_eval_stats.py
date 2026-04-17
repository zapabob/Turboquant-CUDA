from __future__ import annotations

import pandas as pd

from turboquant.eval_stats import (
    compute_benchmark_pairwise_statistics,
    compute_continuous_pairwise_statistics,
    summarize_benchmark_items,
    summarize_continuous_metrics,
)


def test_summarize_continuous_metrics_emits_mean_sem_and_ci() -> None:
    frame = pd.DataFrame(
        [
            {"backend": "hf", "metric": "ppl", "mode": "exact", "value": 1.0},
            {"backend": "hf", "metric": "ppl", "mode": "exact", "value": 2.0},
            {"backend": "hf", "metric": "ppl", "mode": "exact", "value": 3.0},
        ]
    )

    summary = summarize_continuous_metrics(
        frame,
        group_columns=["backend", "metric", "mode"],
    )

    row = summary.iloc[0]
    assert row["n"] == 3
    assert row["mean"] == 2.0
    assert row["std"] > 0.0
    assert row["sem"] > 0.0
    assert row["ci95_low"] < row["mean"] < row["ci95_high"]


def test_summarize_benchmark_items_emits_accuracy_and_wilson_interval() -> None:
    frame = pd.DataFrame(
        [
            {"backend": "hf", "task": "piqa", "mode": "exact", "doc_id": "0", "is_correct": 1},
            {"backend": "hf", "task": "piqa", "mode": "exact", "doc_id": "1", "is_correct": 1},
            {"backend": "hf", "task": "piqa", "mode": "exact", "doc_id": "2", "is_correct": 0},
            {"backend": "hf", "task": "piqa", "mode": "exact", "doc_id": "3", "is_correct": 1},
        ]
    )

    summary = summarize_benchmark_items(
        frame,
        group_columns=["backend", "task", "mode"],
    )

    row = summary.iloc[0]
    assert row["n"] == 4
    assert row["n_correct"] == 3
    assert row["mean"] == 0.75
    assert row["ci95_low"] < row["mean"] < row["ci95_high"]


def test_compute_continuous_pairwise_statistics_applies_wilcoxon_and_holm() -> None:
    frame = pd.DataFrame(
        [
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "0", "mode": "exact", "value": 1.0},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "0", "mode": "turbo", "value": 1.1},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "0", "mode": "triality", "value": 1.02},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "1", "mode": "exact", "value": 1.0},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "1", "mode": "turbo", "value": 1.2},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "1", "mode": "triality", "value": 1.03},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "2", "mode": "exact", "value": 1.0},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "2", "mode": "turbo", "value": 1.3},
            {"backend": "hf", "metric": "log_perplexity", "chunk_id": "2", "mode": "triality", "value": 1.01},
        ]
    )

    stats_frame = compute_continuous_pairwise_statistics(
        frame,
        group_columns=["backend", "metric"],
        pairing_columns=["chunk_id"],
        baseline_modes=["exact"],
    )

    assert set(stats_frame["candidate_mode"]) == {"triality", "turbo"}
    assert set(stats_frame["test"]) == {"wilcoxon_signed_rank"}
    assert all(stats_frame["n_pairs"] == 3)
    assert all(stats_frame["p_value_holm"] >= stats_frame["p_value"])


def test_compute_benchmark_pairwise_statistics_emits_mcnemar_counts() -> None:
    frame = pd.DataFrame(
        [
            {"backend": "runtime", "task": "hellaswag", "doc_id": "0", "mode": "exact", "is_correct": 1},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "0", "mode": "turbo", "is_correct": 1},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "1", "mode": "exact", "is_correct": 1},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "1", "mode": "turbo", "is_correct": 0},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "2", "mode": "exact", "is_correct": 0},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "2", "mode": "turbo", "is_correct": 1},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "3", "mode": "exact", "is_correct": 1},
            {"backend": "runtime", "task": "hellaswag", "doc_id": "3", "mode": "turbo", "is_correct": 0},
        ]
    )

    stats_frame = compute_benchmark_pairwise_statistics(
        frame,
        group_columns=["backend", "task"],
        pairing_columns=["doc_id"],
        baseline_modes=["exact"],
    )

    row = stats_frame.iloc[0]
    assert row["baseline_mode"] == "exact"
    assert row["candidate_mode"] == "turbo"
    assert row["test"] == "mcnemar_exact"
    assert row["n_pairs"] == 4
    assert row["discordant_baseline_only"] == 2
    assert row["discordant_candidate_only"] == 1
    assert row["p_value_holm"] >= row["p_value"]
