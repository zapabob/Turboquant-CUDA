"""KV codec and attention score estimator."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import QuantizedMSEBatch, QuantizedProdBatch, TurboQuantMSEConfig, TurboQuantProdConfig


@dataclass(slots=True)
class KVCodecConfig:
    head_dim: int
    key_bits: int = 3
    value_bits: int = 3
    rotation_seed: int = 0
    qjl_seed: int = 1
    qjl_dim: int | None = None
    device: str = "cpu"
    dtype: str = "float32"
    mixed_key_bits: float | None = None
    mixed_value_bits: float | None = None


class AttentionScoreEstimator:
    """Compute exact or TurboQuant-estimated attention logits."""

    def __init__(self, key_quantizer: TurboQuantProd) -> None:
        self.key_quantizer = key_quantizer

    @staticmethod
    def exact(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...qd,...sd->...qs", q, k)

    def turboquant(self, q: torch.Tensor, encoded_k: QuantizedProdBatch) -> torch.Tensor:
        q = q.to(device=self.key_quantizer.device, dtype=self.key_quantizer.dtype)
        mse_keys = self.key_quantizer.dequantize(encoded_k)
        mse_scores = self.exact(q=q, k=mse_keys)
        residual_scores = self.key_quantizer.qjl.pairwise_estimate(q=q, sketch=encoded_k.qjl)
        return mse_scores + residual_scores


class KVCodec:
    """Encode KV tensors with paper-faithful stage separation."""

    def __init__(self, config: KVCodecConfig) -> None:
        self.config = config
        self.key_quantizer = TurboQuantProd(
            TurboQuantProdConfig(
                dim=config.head_dim,
                total_bits=config.key_bits,
                qjl_dim=config.qjl_dim,
                rotation_seed=config.rotation_seed,
                qjl_seed=config.qjl_seed,
                device=config.device,
                dtype=config.dtype,
            )
        )
        self.value_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=config.head_dim,
                bits=config.value_bits,
                rotation_seed=config.rotation_seed,
                device=config.device,
                dtype=config.dtype,
            )
        )
        self.estimator = AttentionScoreEstimator(self.key_quantizer)

    def _allocation(self, effective_bits: float | None) -> ChannelBitAllocation | None:
        if effective_bits is None:
            return None
        return ChannelBitAllocation.preset(effective_bits=effective_bits, width=self.config.head_dim)

    def encode_keys(self, keys: torch.Tensor) -> QuantizedProdBatch:
        return self.key_quantizer.quantize(keys, allocation=self._allocation(self.config.mixed_key_bits))

    def encode_values(self, values: torch.Tensor) -> QuantizedMSEBatch:
        return self.value_quantizer.quantize(
            values,
            allocation=self._allocation(self.config.mixed_value_bits),
        )

    def decode_keys(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self.key_quantizer.dequantize(encoded)

    def decode_values(self, encoded: QuantizedMSEBatch) -> torch.Tensor:
        return self.value_quantizer.dequantize(encoded)
