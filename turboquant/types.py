"""Shared public types for the TurboQuant prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TurboQuantMSEConfig:
    """Configuration for the Stage 1 MSE-optimized quantizer."""

    dim: int
    bits: int
    rotation_seed: int = 0
    codebook_kind: str = "sphere-lloyd-max"
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(slots=True)
class TurboQuantProdConfig:
    """Configuration for the Stage 2 inner-product estimator."""

    dim: int
    total_bits: int
    mse_bits: int | None = None
    qjl_bits: int = 1
    qjl_dim: int | None = None
    rotation_seed: int = 0
    qjl_seed: int = 1
    device: str = "cpu"
    dtype: str = "float32"

    def resolved_mse_bits(self) -> int:
        if self.mse_bits is not None:
            return self.mse_bits
        return self.total_bits - self.qjl_bits

    def resolved_qjl_dim(self) -> int:
        if self.qjl_dim is not None:
            return self.qjl_dim
        return max(self.dim * 4, 64)


@dataclass(slots=True)
class QuantizedMSEBatch:
    """Encoded Stage 1 representation.

    Shapes:
    - norms: [..., 1]
    - indices: [..., dim]
    - bitwidths: [..., dim]
    """

    norms: torch.Tensor
    indices: torch.Tensor
    bitwidths: torch.Tensor
    shape: tuple[int, ...]
    dim: int


@dataclass(slots=True)
class QJLSketch:
    """1-bit random-projection sketch for residual inner-product correction.

    Shapes:
    - signs: [..., sketch_dim]
    - norms: [..., 1]
    """

    signs: torch.Tensor
    norms: torch.Tensor
    shape: tuple[int, ...]
    dim: int
    sketch_dim: int


@dataclass(slots=True)
class QuantizedProdBatch:
    """Combined Stage 1 + Stage 2 representation."""

    mse: QuantizedMSEBatch
    qjl: QJLSketch
    shape: tuple[int, ...]
    dim: int
