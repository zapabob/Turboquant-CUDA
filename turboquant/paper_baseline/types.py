"""PyTorch-only public types for the paper-faithful TurboQuant baseline."""

from __future__ import annotations

from dataclasses import dataclass

from turboquant.allocation import ChannelBitAllocation
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig


@dataclass(slots=True)
class PaperMSEConfig:
    """Stage 1 configuration constrained to the paper baseline."""

    dim: int
    bits: int
    rotation_seed: int = 0
    rotation_policy: str = "random_haar"
    norm_mode: str = "explicit"
    codebook_kind: str = "sphere-lloyd-max"
    device: str = "cpu"
    dtype: str = "float32"

    def to_runtime_config(self) -> TurboQuantMSEConfig:
        if self.rotation_policy != "random_haar":
            raise ValueError("PaperMSEConfig only supports rotation_policy='random_haar'")
        if self.norm_mode != "explicit":
            raise ValueError("PaperMSEConfig only supports norm_mode='explicit'")
        return TurboQuantMSEConfig(
            dim=self.dim,
            bits=self.bits,
            rotation_seed=self.rotation_seed,
            rotation_policy=self.rotation_policy,
            codebook_kind=self.codebook_kind,
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(slots=True)
class PaperProdConfig:
    """Stage 2 configuration constrained to the paper baseline."""

    dim: int
    bits_total: int
    mse_bits: int | None = None
    qjl_bits: int = 1
    qjl_dim: int | None = None
    rotation_seed: int = 0
    rotation_policy: str = "random_haar"
    qjl_seed: int = 1
    device: str = "cpu"
    dtype: str = "float32"

    def to_runtime_config(self) -> TurboQuantProdConfig:
        if self.rotation_policy != "random_haar":
            raise ValueError("PaperProdConfig only supports rotation_policy='random_haar'")
        return TurboQuantProdConfig(
            dim=self.dim,
            total_bits=self.bits_total,
            mse_bits=self.mse_bits,
            qjl_bits=self.qjl_bits,
            qjl_dim=self.qjl_dim,
            rotation_seed=self.rotation_seed,
            rotation_policy=self.rotation_policy,
            qjl_seed=self.qjl_seed,
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(slots=True)
class PaperMixedBitPolicy:
    """Explicit paper mixed-bit policy for 2.5 / 3.5-bit settings."""

    low_bits: int
    high_bits: int
    high_count: int
    selector: str = "paper_outlier_magnitude"

    @classmethod
    def for_total_bits(cls, total_bits: float, dim: int) -> "PaperMixedBitPolicy":
        if total_bits not in {2.5, 3.5}:
            raise ValueError(f"Unsupported paper mixed-bit setting: {total_bits}")
        high_count = min(dim, max(1, int(round(dim * 0.25))))
        low_bits = int(total_bits - 0.5)
        high_bits = low_bits + 1
        return cls(low_bits=low_bits, high_bits=high_bits, high_count=high_count)

    def allocation(self, dim: int) -> ChannelBitAllocation:
        if self.selector != "paper_outlier_magnitude":
            raise ValueError(f"Unsupported paper selector: {self.selector!r}")
        if self.high_count > dim:
            raise ValueError(f"high_count={self.high_count} exceeds dim={dim}")
        return ChannelBitAllocation(
            regular_bits=self.low_bits,
            outlier_bits=self.high_bits,
            outlier_count=self.high_count,
            selection_policy="magnitude-topk",
        )
