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
        if config.resolved_mse_bits() < 1:
            raise ValueError("TurboQuantProd requires at least 1 MSE bit before the QJL residual step")
        self.mse_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=config.dim,
                bits=config.resolved_mse_bits(),
                rotation_seed=config.rotation_seed,
                rotation_policy=config.rotation_policy,
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
        if x.device != self.device:
            raise ValueError(f"Expected x on {self.device}, got {x.device}")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x dtype {self.dtype}, got {x.dtype}")
        mse_encoded = self.mse_quantizer.quantize(x=x, allocation=allocation)
        mse_reconstruction = self.mse_quantizer.dequantize(mse_encoded)
        residual = x - mse_reconstruction
        qjl_sketch = self.qjl.encode(residual)
        return QuantizedProdBatch(
            mse=mse_encoded,
            qjl=qjl_sketch,
            shape=tuple(x.shape),
            dim=self.config.dim,
        )

    def dequantize(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self.mse_quantizer.dequantize(encoded.mse)

    def transport_decode(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        """Decode a full vector approximation using the Stage 1 path plus a QJL back-projection.

        The paper uses the QJL residual only for unbiased inner-product
        estimation. This transport decode is intentionally research-only and is
        used to test whether a Prod-style residual path is a poor value codec.
        """

        mse = self.dequantize(encoded)
        residual = self.qjl.decode(encoded.qjl)
        return mse + residual

    def estimate_inner_product(self, y: torch.Tensor, encoded: QuantizedProdBatch) -> torch.Tensor:
        if y.device != self.device:
            raise ValueError(f"Expected y on {self.device}, got {y.device}")
        if y.dtype != self.dtype:
            raise ValueError(f"Expected y dtype {self.dtype}, got {y.dtype}")
        mse_part = (y * self.dequantize(encoded)).sum(dim=-1)
        residual_part = self.qjl.estimate(y=y, sketch=encoded.qjl)
        return mse_part + residual_part

    def fit_rotation(
        self,
        x: torch.Tensor,
        *,
        queries: torch.Tensor | None = None,
        steps: int = 60,
        lr: float = 5e-2,
    ) -> torch.Tensor:
        return self.mse_quantizer.fit_rotation(x=x, queries=queries, steps=steps, lr=lr)
