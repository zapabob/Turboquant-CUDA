"""1-bit Gaussian sign sketches for residual inner-product correction."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from turboquant.types import QJLSketch


@lru_cache(maxsize=32)
def _gaussian_matrix_cpu(dim: int, sketch_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((sketch_dim, dim), generator=generator, dtype=torch.float32)


class GaussianSignSketch:
    """1-bit sketch with an unbiased inner-product estimator.

    For a Gaussian row vector g and unit residual direction u, the identity

        E[sign(g · u) (g · y)] = sqrt(2 / pi) <u, y>

    gives an unbiased estimator of the residual inner product.
    """

    def __init__(self, dim: int, sketch_dim: int, seed: int, device: str, dtype: str) -> None:
        self.dim = dim
        self.sketch_dim = sketch_dim
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        # Paper-faithful default: one Gaussian sketch row per input coordinate.
        self.matrix = _gaussian_matrix_cpu(dim=dim, sketch_dim=sketch_dim, seed=seed).to(
            device=self.device,
            dtype=self.dtype,
        )

    def encode(self, residual: torch.Tensor) -> QJLSketch:
        if residual.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {residual.shape[-1]}")
        if residual.device != self.device:
            raise ValueError(f"Expected residual on {self.device}, got {residual.device}")
        if residual.dtype != self.dtype:
            raise ValueError(f"Expected residual dtype {self.dtype}, got {residual.dtype}")
        norms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=torch.finfo(residual.dtype).eps)
        unit = torch.where(norms > 0, residual / safe_norms, torch.zeros_like(residual))
        projections = torch.matmul(unit, self.matrix.transpose(0, 1))
        signs = (projections >= 0).to(torch.int8)
        return QJLSketch(
            signs=signs,
            norms=norms,
            shape=tuple(residual.shape),
            dim=self.dim,
            sketch_dim=self.sketch_dim,
        )

    def estimate(self, y: torch.Tensor, sketch: QJLSketch) -> torch.Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {y.shape[-1]}")
        if y.device != self.device:
            raise ValueError(f"Expected y on {self.device}, got {y.device}")
        if y.dtype != self.dtype:
            raise ValueError(f"Expected y dtype {self.dtype}, got {y.dtype}")
        proj_y = torch.matmul(y, self.matrix.transpose(0, 1))
        signs_pm = sketch.signs.to(dtype=proj_y.dtype).mul(2.0).sub(1.0)
        estimate = math.sqrt(math.pi / 2.0) * sketch.norms.squeeze(-1) * (
            (signs_pm * proj_y).mean(dim=-1)
        )
        return estimate

    def decode(self, sketch: QJLSketch) -> torch.Tensor:
        """Back-project the 1-bit sketch into a residual vector estimate.

        This is not used by the paper-faithful key-side estimator, but it is a
        useful research ablation for testing whether the QJL residual path is a
        poor transport codec for values.
        """

        if sketch.dim != self.dim:
            raise ValueError(f"Expected sketch dimension {self.dim}, got {sketch.dim}")
        signs_pm = sketch.signs.to(dtype=self.dtype).mul(2.0).sub(1.0)
        decoded = math.sqrt(math.pi / 2.0) * torch.einsum("...m,md->...d", signs_pm, self.matrix)
        decoded = decoded / float(self.sketch_dim)
        return decoded * sketch.norms

    def pairwise_estimate(self, q: torch.Tensor, sketch: QJLSketch) -> torch.Tensor:
        """Estimate pairwise inner products for q:[..., Q, D] and sketch:[..., S, M]."""

        if q.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {q.shape[-1]}")
        if q.device != self.device:
            raise ValueError(f"Expected q on {self.device}, got {q.device}")
        if q.dtype != self.dtype:
            raise ValueError(f"Expected q dtype {self.dtype}, got {q.dtype}")
        proj_q = torch.matmul(q, self.matrix.transpose(0, 1))
        signs_pm = sketch.signs.to(dtype=proj_q.dtype).mul(2.0).sub(1.0)
        correction = torch.einsum("...sm,...qm->...qs", signs_pm, proj_q) / self.sketch_dim
        return math.sqrt(math.pi / 2.0) * correction * sketch.norms.squeeze(-1).unsqueeze(-2)
