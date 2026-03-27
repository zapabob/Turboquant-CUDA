"""Paper-faithful Stage 2 wrapper with no transport-decode API."""

from __future__ import annotations

import torch

from turboquant.paper_baseline.types import PaperProdConfig
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import QuantizedProdBatch


class PaperTurboQuantProd:
    """Stage 2 wrapper exposing only the paper's inner-product interface."""

    def __init__(self, config: PaperProdConfig) -> None:
        self.config = config
        self._impl = TurboQuantProd(config.to_runtime_config())

    @property
    def mse_quantizer(self):
        return self._impl.mse_quantizer

    @property
    def qjl(self):
        return self._impl.qjl

    def quantize(self, x: torch.Tensor, allocation=None) -> QuantizedProdBatch:
        return self._impl.quantize(x, allocation=allocation)

    def dequantize(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self._impl.dequantize(encoded)

    def estimate_inner_product(self, y: torch.Tensor, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self._impl.estimate_inner_product(y, encoded)
