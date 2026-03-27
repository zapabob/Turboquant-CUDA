"""Numerical Lloyd-Max codebook fitting for sphere-coordinate marginals."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


def sphere_coordinate_pdf(dim: int, grid: np.ndarray) -> np.ndarray:
    """PDF of one coordinate of a random point on the unit sphere S^(d-1)."""

    if dim < 2:
        raise ValueError("dim must be at least 2")
    coef = math.gamma(dim / 2.0) / (math.sqrt(math.pi) * math.gamma((dim - 1) / 2.0))
    clipped = np.clip(1.0 - np.square(grid), 0.0, None)
    return coef * np.power(clipped, (dim - 3.0) / 2.0)


def _interval_centroid(grid: np.ndarray, pdf: np.ndarray, left: float, right: float) -> float:
    mask = (grid >= left) & (grid <= right)
    if not mask.any():
        return float((left + right) * 0.5)
    x = grid[mask]
    p = pdf[mask]
    denom = np.trapezoid(p, x)
    if denom <= 0:
        return float((left + right) * 0.5)
    return float(np.trapezoid(x * p, x) / denom)


@lru_cache(maxsize=64)
def fit_lloyd_max_codebook(dim: int, bits: int, grid_size: int = 16385) -> tuple[float, ...]:
    """Fit a symmetric Lloyd-Max codebook on the sphere-coordinate marginal."""

    if bits < 1:
        raise ValueError("bits must be positive")
    num_levels = 2**bits
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    pdf = sphere_coordinate_pdf(dim=dim, grid=grid)
    levels = np.linspace(-0.95, 0.95, num_levels, dtype=np.float64)
    for _ in range(200):
        boundaries = np.empty(num_levels + 1, dtype=np.float64)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = (levels[:-1] + levels[1:]) * 0.5
        new_levels = np.empty_like(levels)
        for idx in range(num_levels):
            new_levels[idx] = _interval_centroid(
                grid=grid,
                pdf=pdf,
                left=boundaries[idx],
                right=boundaries[idx + 1],
            )
        if np.max(np.abs(new_levels - levels)) < 1e-8:
            levels = new_levels
            break
        levels = new_levels
    levels = np.sort((levels - levels[::-1]) * 0.5)
    return tuple(float(v) for v in levels)


def codebook_tensor(dim: int, bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    levels = fit_lloyd_max_codebook(dim=dim, bits=bits)
    return torch.tensor(levels, device=device, dtype=dtype)


def decision_boundaries_tensor(
    dim: int,
    bits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    codebook = codebook_tensor(dim=dim, bits=bits, device=device, dtype=dtype)
    boundaries = torch.empty((codebook.numel() + 1,), device=device, dtype=dtype)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = (codebook[:-1] + codebook[1:]) * 0.5
    return boundaries
