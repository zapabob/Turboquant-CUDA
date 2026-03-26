from __future__ import annotations

import torch

from turboquant.attention_metrics import summarize_attention_scores


def test_attention_metrics_perfect_match() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    summary = summarize_attention_scores(scores, scores)
    assert summary["mae"] == 0.0
    assert summary["top1_match"] == 1.0
