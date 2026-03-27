"""Attention score and hidden-state metrics used by offline validation."""

from __future__ import annotations

import pandas as pd
import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat)
    if denom <= 0:
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)
    return torch.dot(a_flat, b_flat) / denom


def spearman_rank_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_rank = a.argsort(dim=-1).argsort(dim=-1).to(torch.float32)
    b_rank = b.argsort(dim=-1).argsort(dim=-1).to(torch.float32)
    return cosine_similarity(a_rank, b_rank)


def topk_match_rate(reference: torch.Tensor, estimate: torch.Tensor, k: int) -> float:
    ref_top = reference.topk(k, dim=-1).indices
    est_top = estimate.topk(k, dim=-1).indices
    matches = (ref_top == est_top).any(dim=-1)
    return float(matches.to(torch.float32).mean().item())


def topk_overlap_rate(reference: torch.Tensor, estimate: torch.Tensor, k: int) -> float:
    ref_top = reference.topk(k, dim=-1).indices
    est_top = estimate.topk(k, dim=-1).indices
    overlap = (ref_top.unsqueeze(-1) == est_top.unsqueeze(-2)).any(dim=-1).to(torch.float32)
    return float((overlap.mean(dim=-1)).mean().item())


def summarize_attention_scores(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    diff = estimate - reference
    return {
        "cosine_similarity": float(cosine_similarity(reference, estimate).item()),
        "mae": float(diff.abs().mean().item()),
        "mse": float(diff.square().mean().item()),
        "spearman": float(spearman_rank_correlation(reference, estimate).item()),
        "top1_match": topk_match_rate(reference, estimate, k=1),
        "top5_overlap": topk_overlap_rate(reference, estimate, k=min(5, reference.shape[-1])),
        "top5_match": topk_match_rate(reference, estimate, k=min(5, reference.shape[-1])),
    }


def metrics_frame(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
