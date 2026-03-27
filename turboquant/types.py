"""Shared public types for the paper-faithful TurboQuant prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

RotationPolicy = Literal["random_haar", "block_so8_static", "block_so8_learned"]
ProtectionPolicy = Literal["none", "sensitivity_mixed", "sensitive_layer_exact", "protected_low_rank"]
Granularity = Literal["per-layer", "per-head", "per-channel"]
ScoreSource = Literal["attention-output-sensitivity", "teacher-gradient-proxy"]


@dataclass(slots=True)
class SensitivitySpec:
    """Calibration configuration for value sensitivity estimation."""

    granularity: Granularity = "per-channel"
    score_source: ScoreSource = "attention-output-sensitivity"
    calibration_samples: int = 32


@dataclass(slots=True)
class ValueCodecConfig:
    """Configuration for value-side output-preserving compression."""

    base_bits: int = 3
    protection_policy: ProtectionPolicy = "sensitivity_mixed"
    protected_fraction: float = 0.10
    high_bits: int = 8
    low_rank_rank: int = 0
    low_rank_coeff_bits: int = 16
    secondary_fraction: float = 0.10
    sensitivity: SensitivitySpec = field(default_factory=SensitivitySpec)


@dataclass(slots=True)
class MemoryBudgetSpec:
    """Target budget for cache compression experiments."""

    target_ratio: float = 0.25
    max_metadata_ratio: float = 0.02


@dataclass(slots=True)
class TurboQuantMSEConfig:
    """Configuration for the Stage 1 MSE-optimized quantizer."""

    dim: int
    bits: int
    rotation_seed: int = 0
    rotation_policy: RotationPolicy = "random_haar"
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
    rotation_policy: RotationPolicy = "random_haar"
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
        return self.dim


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

    def quantized_bits(self) -> int:
        return int(self.bitwidths.to(torch.int64).sum().item())

    def metadata_bits(self) -> int:
        return int(self.norms.numel() * self.norms.element_size() * 8)

    def total_bits(self) -> int:
        return self.quantized_bits() + self.metadata_bits()


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

    def quantized_bits(self) -> int:
        return int(self.signs.numel())

    def metadata_bits(self) -> int:
        return int(self.norms.numel() * self.norms.element_size() * 8)

    def total_bits(self) -> int:
        return self.quantized_bits() + self.metadata_bits()


@dataclass(slots=True)
class QuantizedProdBatch:
    """Combined Stage 1 + Stage 2 representation."""

    mse: QuantizedMSEBatch
    qjl: QJLSketch
    shape: tuple[int, ...]
    dim: int

    def total_bits(self) -> int:
        return self.mse.total_bits() + self.qjl.total_bits()


@dataclass(slots=True)
class ProtectedValueBatch:
    """Value batch with quantized base, exact protected channels, and optional low-rank residual."""

    base: QuantizedMSEBatch
    exact_channel_mask: torch.Tensor
    exact_values: torch.Tensor
    high_precision_mask: torch.Tensor
    low_rank_coefficients: torch.Tensor | None
    shape: tuple[int, ...]
    dim: int

    def metadata_bits(self) -> int:
        # Channel masks are stored once per prefix tensor, not per token.
        return int(self.exact_channel_mask.numel() + self.high_precision_mask.numel())

    def exact_bits(self) -> int:
        expanded_mask = self.exact_channel_mask.unsqueeze(-2).expand(self.shape)
        return int(expanded_mask.to(torch.int64).sum().item() * self.exact_values.element_size() * 8)

    def low_rank_bits(self) -> int:
        if self.low_rank_coefficients is None:
            return 0
        return int(self.low_rank_coefficients.numel() * self.low_rank_coefficients.element_size() * 8)

    def total_bits(self) -> int:
        return self.base.total_bits() + self.metadata_bits() + self.exact_bits() + self.low_rank_bits()
