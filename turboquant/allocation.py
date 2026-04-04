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
        if effective_bits == 1.5:
            return cls.from_ratio(regular_bits=1, outlier_bits=2, width=width, outlier_ratio=0.25)
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

    def outlier_ratio(self, width: int | None = None) -> float:
        if width is None:
            raise ValueError("width is required to compute the effective outlier ratio")
        return float(self.outlier_count) / float(width)

    def effective_bits(self, width: int) -> float:
        ratio = self.outlier_ratio(width)
        return (self.regular_bits * (1.0 - ratio)) + (self.outlier_bits * ratio)

    @classmethod
    def from_multiscreen_relevance(
        cls,
        regular_bits: int,
        outlier_bits: int,
        outlier_count: int,
    ) -> "ChannelBitAllocation":
        """Build allocation config for Multiscreen relevance-based outlier selection.

        Use :meth:`make_bitwidths_from_relevance` with a relevance tensor to obtain
        per-position bitwidths.
        """
        return cls(
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            outlier_count=outlier_count,
            selection_policy="multiscreen-relevance",
        )

    def make_bitwidths_from_relevance(self, relevance: torch.Tensor) -> torch.Tensor:
        """Select outlier K positions by Multiscreen relevance score.

        ``relevance`` may be any shape; indices are chosen among all elements via
        global top-``outlier_count``. Returns a 1D tensor of length ``relevance.numel()``
        with dtype ``torch.int64`` (bits per flattened position).

        Args:
            relevance: Non-negative importance scores (larger = keep more bits).

        Returns:
            One int64 bitwidth per element of ``relevance``, ``outlier_bits`` on
            the ``outlier_count`` highest-relevance positions and ``regular_bits``
            elsewhere.
        """
        if self.selection_policy != "multiscreen-relevance":
            raise ValueError(
                f"make_bitwidths_from_relevance requires selection_policy="
                f"'multiscreen-relevance', got {self.selection_policy!r}"
            )
        flat = relevance.detach().flatten()
        n = flat.numel()
        if self.outlier_count > n:
            raise ValueError(
                f"outlier_count={self.outlier_count} exceeds relevance numel={n}"
            )
        if self.outlier_count == 0:
            return torch.full((n,), self.regular_bits, dtype=torch.int64, device=flat.device)
        bitwidths = torch.full((n,), self.regular_bits, dtype=torch.int64, device=flat.device)
        _, topk_idx = torch.topk(flat, k=self.outlier_count)
        bitwidths[topk_idx] = self.outlier_bits
        return bitwidths

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
        if self.selection_policy != "magnitude-topk":
            raise ValueError(f"Unsupported selection_policy={self.selection_policy!r}")
        _, indices = torch.topk(values.abs(), k=self.outlier_count, dim=-1)
        bitwidths.scatter_(-1, indices, self.outlier_bits)
        return bitwidths
