"""Smoke tests for ``evaluate_multiscreen_relevance_attention_row``."""

from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import evaluate_multiscreen_relevance_attention_row, select_queries


def test_evaluate_multiscreen_relevance_attention_row_runs() -> None:
    torch.manual_seed(0)
    # [batch, heads, seq, head_dim]
    keys = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    values = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    queries = select_queries(keys, seed=123)
    alloc = ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=2,
        outlier_bits=3,
        outlier_count=4,
    )
    row = evaluate_multiscreen_relevance_attention_row(
        dataset="synthetic",
        trial=0,
        layer_idx=0,
        bit_value=3.0,
        keys=keys,
        values=values,
        queries=queries,
        allocation=alloc,
        rotation_policy="block_so8_static",
        rotation_seed=1,
        qjl_seed=2,
    )
    assert row["mode"] == "multiscreen_relevance"
    assert 0.0 <= float(row["logit_cosine_similarity"]) <= 1.0 + 1e-5
    assert float(row["memory_ratio_vs_exact"]) < 1.0 + 1e-5
