"""Stage 2 TurboQuant implementation with unbiased inner-product correction."""

from __future__ import annotations

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.qjl import GaussianSignSketch
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import QuantizedProdBatch, TurboQuantMSEConfig, TurboQuantProdConfig


class TurboQuantProd:
    """`TurboQuantMSE` plus residual QJL correction."""

    def __init__(self, config: TurboQuantProdConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.mse_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=config.dim,
                bits=config.resolved_mse_bits(),
                rotation_seed=config.rotation_seed,
                device=config.device,
                dtype=config.dtype,
            )
        )
        self.qjl = GaussianSignSketch(
            dim=config.dim,
            sketch_dim=config.resolved_qjl_dim(),
            seed=config.qjl_seed,
            device=config.device,
            dtype=config.dtype,
        )

    def quantize(
        self,
        x: torch.Tensor,
        allocation: ChannelBitAllocation | None = None,
    ) -> QuantizedProdBatch:
        mse_encoded = self.mse_quantizer.quantize(x=x, allocation=allocation)
        mse_reconstruction = self.mse_quantizer.dequantize(mse_encoded)
        residual = x.to(device=self.device, dtype=self.dtype) - mse_reconstruction
        qjl_sketch = self.qjl.encode(residual)
        return QuantizedProdBatch(
            mse=mse_encoded,
            qjl=qjl_sketch,
            shape=tuple(x.shape),
            dim=self.config.dim,
        )

    def dequantize(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self.mse_quantizer.dequantize(encoded.mse)

    def estimate_inner_product(self, y: torch.Tensor, encoded: QuantizedProdBatch) -> torch.Tensor:
        y = y.to(device=self.device, dtype=self.dtype)
        mse_part = (y * self.dequantize(encoded)).sum(dim=-1)
        residual_part = self.qjl.estimate(y=y, sketch=encoded.qjl)
        return mse_part + residual_part
