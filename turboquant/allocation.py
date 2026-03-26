"""Mixed-bit channel allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ChannelBitAllocation:
    """Describe mixed-bit allocation over the last tensor dimension."""

    regular_bits: int
    outlier_bits: int
    outlier_count: int
    selection_policy: str = "magnitude-topk"

    @classmethod
    def preset(cls, effective_bits: float, width: int) -> "ChannelBitAllocation":
        if effective_bits == 2.5:
            return cls.from_ratio(regular_bits=2, outlier_bits=3, width=width, outlier_ratio=0.25)
        if effective_bits == 3.5:
            return cls.from_ratio(regular_bits=3, outlier_bits=4, width=width, outlier_ratio=0.25)
        raise ValueError(f"Unsupported mixed-bit preset: {effective_bits}")

    @classmethod
    def from_ratio(
        cls,
        regular_bits: int,
        outlier_bits: int,
        width: int,
        outlier_ratio: float,
    ) -> "ChannelBitAllocation":
        outlier_count = max(1, int(round(width * outlier_ratio)))
        outlier_count = min(outlier_count, width)
        return cls(
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            outlier_count=outlier_count,
        )

    def make_bitwidths(self, values: torch.Tensor) -> torch.Tensor:
        """Select outlier coordinates along the last dimension.

        The policy is intentionally calibration-free and depends only on the
        current tensor values.
        """

        if values.shape[-1] < self.outlier_count:
            raise ValueError(
                f"outlier_count={self.outlier_count} exceeds width={values.shape[-1]}"
            )
        bitwidths = torch.full_like(values, fill_value=self.regular_bits, dtype=torch.uint8)
        if self.outlier_count == 0:
            return bitwidths
        _, indices = torch.topk(values.abs(), k=self.outlier_count, dim=-1)
        bitwidths.scatter_(-1, indices, self.outlier_bits)
        return bitwidths
