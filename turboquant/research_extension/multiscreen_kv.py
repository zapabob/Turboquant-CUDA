"""Multiscreen-derived KV relevance scoring for mixed-bit allocation.

Ports the ScreeningUnit signal flow from multiscreen-pytorch/multiscreen/layers.py
as a parameter-free importance estimator for K positions. No dependency on
multiscreen-pytorch (logic copied for reproducibility).
"""

from __future__ import annotations

import torch


def normalize_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-normalize along the last dimension. Shape: [..., d] -> [..., d]."""
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def build_mipe_factor(
    relative_positions: torch.Tensor,  # [query, key]
    s_e: torch.Tensor,  # [head_dim]
    s_f: torch.Tensor,  # [head_dim]
) -> torch.Tensor:
    """Compute bounded MiPE factor. Returns [query, key]."""
    scale = torch.exp(s_e).clamp_min(1.0)
    threshold = torch.exp(s_f).clamp_min(1.0)
    rel = relative_positions.to(dtype=scale.dtype)
    phase = rel.unsqueeze(-1) / scale.view(1, 1, -1)
    per_dim = torch.cos(phase)
    activated = rel.abs().unsqueeze(-1) <= threshold.view(1, 1, -1)
    mixed = torch.where(activated, per_dim, torch.ones_like(per_dim))
    return mixed.mean(dim=-1)


def trim_and_square(similarity: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
    """Sparse relevance gate. High similarity -> high alpha, low -> zero."""
    r = torch.exp(s_r).clamp_min(0.0) + 1.0
    s = (similarity + 1.0) * 0.5
    return torch.clamp(1.0 - r * (1.0 - s), min=0.0).square()


def compute_k_relevance(
    q: torch.Tensor,  # [batch, heads, q_len, head_dim]
    k: torch.Tensor,  # [batch, heads, k_len, head_dim]
    s_e: torch.Tensor | None = None,
    s_f: torch.Tensor | None = None,
    s_r: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Multiscreen relevance scores for each K position.

    Returns:
        relevance: [batch, heads, k_len]
            Max relevance across all query positions for each K position.
            Higher = more important for quantization bit allocation.
    """
    head_dim = q.shape[-1]
    device, dtype = q.device, q.dtype

    if s_e is None:
        s_e = torch.zeros(head_dim, device=device, dtype=dtype)
    if s_f is None:
        s_f = torch.zeros(head_dim, device=device, dtype=dtype)
    if s_r is None:
        s_r = torch.zeros(1, device=device, dtype=dtype)

    q_unit = normalize_unit(q)  # [batch, heads, q_len, head_dim]
    k_unit = normalize_unit(k)  # [batch, heads, k_len, head_dim]

    # cosine similarity: [batch, heads, q_len, k_len]
    base_sim = torch.einsum("bhqd,bhkd->bhqk", q_unit, k_unit).clamp(-1.0, 1.0)

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    q_pos = torch.arange(q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    rel_pos = q_pos[:, None] - k_pos[None, :]  # [q_len, k_len]

    mipe = build_mipe_factor(rel_pos, s_e, s_f)  # [q_len, k_len]
    similarity = (base_sim * mipe.view(1, 1, q_len, k_len)).clamp(-1.0, 1.0)

    alpha = trim_and_square(similarity, s_r)  # [batch, heads, q_len, k_len]

    # K 位置ごとに全 Query の最大 relevance を取る: [batch, heads, k_len]
    return alpha.max(dim=-2).values


def multiscreen_relevance_topk_indices(
    relevance: torch.Tensor,  # [batch, heads, k_len]
    outlier_count: int,
) -> torch.Tensor:
    """Return indices of top-k most relevant K positions per head.

    Shape: [batch, heads, outlier_count]
    """
    return torch.topk(relevance, k=outlier_count, dim=-1).indices
