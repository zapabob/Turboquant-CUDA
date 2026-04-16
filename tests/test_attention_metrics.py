from __future__ import annotations

import math

import torch

from turboquant.attention_metrics import summarize_attention_scores
from turboquant.analysis import evaluate_asymmetric_q8_value_attention_row


def test_attention_metrics_perfect_match() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    summary = summarize_attention_scores(scores, scores)
    assert summary["mae"] == 0.0
    assert summary["top1_match"] == 1.0


def test_head_dim_256_asymmetric_rows_stay_finite() -> None:
    torch.manual_seed(0)
    keys = torch.randn(1, 2, 6, 256, dtype=torch.float32)
    values = torch.randn(1, 2, 6, 256, dtype=torch.float32)

    turbo4 = evaluate_asymmetric_q8_value_attention_row(
        dataset="synthetic",
        trial=0,
        layer_idx=0,
        mode="asym_q8_turbo4",
        keys=keys,
        values=values,
    )
    turbo3 = evaluate_asymmetric_q8_value_attention_row(
        dataset="synthetic",
        trial=0,
        layer_idx=0,
        mode="asym_q8_turbo3",
        keys=keys,
        values=values,
    )

    for row in (turbo4, turbo3):
        assert row["mode"] in {"asym_q8_turbo4", "asym_q8_turbo3"}
        for metric in (
            "logit_cosine_similarity",
            "next_logit_kl",
            "hidden_cosine_similarity",
            "attention_output_relative_error",
        ):
            assert math.isfinite(float(row[metric]))
        assert float(row["memory_ratio_vs_exact"]) < 1.0
