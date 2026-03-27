"""KV codec and attention score estimator."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import (
    MemoryBudgetSpec,
    ProtectedValueBatch,
    QuantizedMSEBatch,
    QuantizedProdBatch,
    RotationPolicy,
    SensitivitySpec,
    TurboQuantMSEConfig,
    TurboQuantProdConfig,
    ValueCodecConfig,
)
from turboquant.value_codec import ProtectedValueCodec


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
    rotation_policy: RotationPolicy = "random_haar"
    mixed_key_bits: float | None = None
    mixed_value_bits: float | None = None
    value_codec: ValueCodecConfig = field(default_factory=ValueCodecConfig)
    sensitivity: SensitivitySpec = field(default_factory=SensitivitySpec)
    memory_budget: MemoryBudgetSpec = field(default_factory=MemoryBudgetSpec)


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
                rotation_policy=config.rotation_policy,
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
                rotation_policy=config.rotation_policy,
                device=config.device,
                dtype=config.dtype,
            )
        )
        self.protected_value_codec = ProtectedValueCodec(
            dim=config.head_dim,
            config=config.value_codec,
            rotation_seed=config.rotation_seed,
            rotation_policy=config.rotation_policy,
            device=config.device,
            dtype=config.dtype,
        )
        self.estimator = AttentionScoreEstimator(self.key_quantizer)

    def _allocation(self, effective_bits: float | None) -> ChannelBitAllocation | None:
        if effective_bits is None:
            return None
        return ChannelBitAllocation.preset(effective_bits=effective_bits, width=self.config.head_dim)

    def encode_keys(self, keys: torch.Tensor) -> QuantizedProdBatch:
        effective_stage1_bits = None
        if self.config.mixed_key_bits is not None:
            effective_stage1_bits = self.config.mixed_key_bits - self.key_quantizer.config.qjl_bits
            if effective_stage1_bits <= 0:
                raise ValueError("mixed_key_bits must exceed the Stage 2 qjl_bits contribution")
        return self.key_quantizer.quantize(keys, allocation=self._allocation(effective_stage1_bits))

    def encode_values(self, values: torch.Tensor) -> QuantizedMSEBatch:
        return self.value_quantizer.quantize(
            values,
            allocation=self._allocation(self.config.mixed_value_bits),
        )

    def calibrate(self, *, keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor) -> None:
        if self.config.rotation_policy == "block_so8_learned":
            self.key_quantizer.fit_rotation(keys, queries=queries)
        exact_logits = self.estimator.exact(queries, keys)
        attention_weights = torch.softmax(exact_logits / (self.config.head_dim**0.5), dim=-1)
        self.protected_value_codec.calibrate(values, attention_weights=attention_weights)

    def decode_keys(self, encoded: QuantizedProdBatch) -> torch.Tensor:
        return self.key_quantizer.dequantize(encoded)

    def decode_values(self, encoded: QuantizedMSEBatch) -> torch.Tensor:
        return self.value_quantizer.dequantize(encoded)

    def encode_protected_values(self, values: torch.Tensor) -> ProtectedValueBatch:
        return self.protected_value_codec.encode(values)

    def decode_protected_values(self, encoded: ProtectedValueBatch) -> torch.Tensor:
        return self.protected_value_codec.decode(encoded)

    def key_storage_bits(self, encoded: QuantizedProdBatch) -> int:
        return encoded.total_bits()

    def value_storage_bits(self, encoded: QuantizedMSEBatch) -> int:
        return encoded.total_bits()

    def protected_value_storage_bits(self, encoded: ProtectedValueBatch) -> int:
        return encoded.total_bits()
