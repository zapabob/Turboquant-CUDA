"""Paper-faithful Stage 1 wrapper with a PyTorch-only public surface."""

from __future__ import annotations

import torch

from turboquant.paper_baseline.types import PaperMSEConfig
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import QuantizedMSEBatch


class PaperTurboQuantMSE:
    """Stage 1 wrapper that intentionally exposes only paper-baseline behavior."""

    def __init__(self, config: PaperMSEConfig) -> None:
        self.config = config
        self._impl = TurboQuantMSE(config.to_runtime_config())

    @property
    def rotation(self) -> torch.Tensor:
        return self._impl.rotation

    def fit_codebook(self, dim: int | None = None, bits: int | None = None) -> torch.Tensor:
        return self._impl.fit_codebook(dim=dim, bits=bits)

    def decision_boundaries(self, bits: int | None = None) -> torch.Tensor:
        return self._impl.decision_boundaries(bits=bits)

    def quantize(self, x: torch.Tensor, allocation=None) -> QuantizedMSEBatch:
        return self._impl.quantize(x, allocation=allocation)

    def quantize_with_bitwidths(self, x: torch.Tensor, bitwidths: torch.Tensor) -> QuantizedMSEBatch:
        return self._impl.quantize_with_bitwidths(x, bitwidths=bitwidths)

    def dequantize(self, encoded: QuantizedMSEBatch) -> torch.Tensor:
        return self._impl.dequantize(encoded)
