"""Tests for Multiscreen-derived KV relevance scoring."""

from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.research_extension.multiscreen_kv import (
    compute_k_relevance,
    multiscreen_relevance_topk_indices,
)


def test_compute_k_relevance_shape() -> None:
    b, h, q_len, k_len, d = 1, 2, 4, 8, 16
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, h, k_len, d)
    rel = compute_k_relevance(q, k)
    assert rel.shape == (b, h, k_len)


def test_compute_k_relevance_range() -> None:
    b, h, q_len, k_len, d = 1, 1, 3, 5, 8
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, h, k_len, d)
    rel = compute_k_relevance(q, k)
    assert (rel >= 0).all()
    assert (rel <= 1).all()


def test_s_r_increases_sparsity() -> None:
    """Larger s_r should not increase mean relevance (typically lowers gate)."""
    b, h, q_len, k_len, d = 1, 1, 2, 4, 8
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, h, k_len, d)
    s_r_lo = torch.zeros(1)
    s_r_hi = torch.ones(1) * 4.0
    r_lo = compute_k_relevance(q, k, s_r=s_r_lo).mean()
    r_hi = compute_k_relevance(q, k, s_r=s_r_hi).mean()
    assert r_hi <= r_lo + 1e-5


def test_multiscreen_topk_indices_shape() -> None:
    relevance = torch.rand(1, 2, 10)
    idx = multiscreen_relevance_topk_indices(relevance, outlier_count=3)
    assert idx.shape == (1, 2, 3)


def test_make_bitwidths_from_relevance() -> None:
    alloc = ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=2,
        outlier_bits=4,
        outlier_count=2,
    )
    relevance = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.0])
    bits = alloc.make_bitwidths_from_relevance(relevance)
    assert bits.shape == (5,)
    assert bits.dtype == torch.int64
    high = (bits == 4).sum().item()
    assert high == 2


def test_make_bitwidths_from_relevance_zero_outliers() -> None:
    alloc = ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=2,
        outlier_bits=4,
        outlier_count=0,
    )
    relevance = torch.ones(4)
    bits = alloc.make_bitwidths_from_relevance(relevance)
    assert (bits == 2).all()
