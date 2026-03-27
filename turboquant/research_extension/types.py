"""Public types for K/V-separated research extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from turboquant.kv_codec import KVCodecConfig
from turboquant.types import SensitivitySpec, ValueCodecConfig

ResearchViewMode = Literal["single", "triality_proxy"]
ResearchViewSelection = Literal["report_all", "best_per_layer"]


@dataclass(slots=True)
class KeyResearchConfig:
    """Research configuration for key-side score-preserving experiments."""

    head_dim: int
    bits_total: int = 3
    mse_bits: int | None = None
    qjl_bits: int = 1
    rotation_policy: str = "block_so8_learned"
    rotation_seed: int = 0
    qjl_seed: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    view_mode: ResearchViewMode = "triality_proxy"
    view_selection: ResearchViewSelection = "report_all"
    views: tuple[str, ...] = ("vector", "spinor_plus_proxy", "spinor_minus_proxy")

    def to_kv_codec_config(self, value_codec: ValueCodecConfig | None = None) -> KVCodecConfig:
        return KVCodecConfig(
            head_dim=self.head_dim,
            key_bits=self.bits_total,
            value_bits=self.bits_total,
            rotation_seed=self.rotation_seed,
            qjl_seed=self.qjl_seed,
            device=self.device,
            dtype=self.dtype,
            rotation_policy=self.rotation_policy,
            value_codec=value_codec or ValueCodecConfig(),
        )


@dataclass(slots=True)
class ValueResearchConfig:
    """Research configuration for value-side transport experiments."""

    base_bits: int = 3
    high_bits: int = 8
    protected_fraction: float = 0.10
    secondary_fraction: float = 0.10
    channel_group_size: int = 8
    sensitivity_source: str = "attention-output-sensitivity"
    low_rank_rank: int = 0
    calibration_samples: int = 32

    def to_value_codec_config(self) -> ValueCodecConfig:
        return ValueCodecConfig(
            base_bits=self.base_bits,
            protected_fraction=self.protected_fraction,
            high_bits=self.high_bits,
            low_rank_rank=self.low_rank_rank,
            secondary_fraction=self.secondary_fraction,
            channel_group_size=self.channel_group_size,
            sensitivity=SensitivitySpec(
                granularity="per-channel",
                score_source=self.sensitivity_source,
                calibration_samples=self.calibration_samples,
            ),
        )


__all__ = ["KeyResearchConfig", "ResearchViewMode", "ResearchViewSelection", "ValueResearchConfig"]
