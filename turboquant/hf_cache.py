"""Replaceable cache backends for Hugging Face replay and smoke tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from turboquant.kv_codec import KVCodec
from turboquant.types import QuantizedMSEBatch, QuantizedProdBatch


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def append(self, layer_idx: int, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Append one cache chunk for a layer."""

    @abstractmethod
    def get_keys(self, layer_idx: int) -> torch.Tensor:
        """Return decoded keys for a layer."""

    @abstractmethod
    def get_values(self, layer_idx: int) -> torch.Tensor:
        """Return decoded values for a layer."""

    @abstractmethod
    def estimate_scores(self, layer_idx: int, q_t: torch.Tensor) -> torch.Tensor:
        """Estimate attention scores against the layer cache."""


class ExactCacheBackend(CacheBackend):
    """Reference cache backend storing raw tensors."""

    def __init__(self) -> None:
        self.keys: dict[int, list[torch.Tensor]] = {}
        self.values: dict[int, list[torch.Tensor]] = {}

    def append(self, layer_idx: int, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        self.keys.setdefault(layer_idx, []).append(k_t.detach())
        self.values.setdefault(layer_idx, []).append(v_t.detach())

    def get_keys(self, layer_idx: int) -> torch.Tensor:
        return torch.cat(self.keys.get(layer_idx, []), dim=-2)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        return torch.cat(self.values.get(layer_idx, []), dim=-2)

    def estimate_scores(self, layer_idx: int, q_t: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...qd,...sd->...qs", q_t, self.get_keys(layer_idx))


class TurboQuantCacheBackend(CacheBackend):
    """Cache backend using `KVCodec` for keys and values."""

    def __init__(self, codec: KVCodec, *, quantize_values: bool = False) -> None:
        self.codec = codec
        self.quantize_values = quantize_values
        self.encoded_keys: dict[int, list[QuantizedProdBatch]] = {}
        self.encoded_values: dict[int, list[QuantizedMSEBatch]] = {}
        self.raw_values: dict[int, list[torch.Tensor]] = {}

    def append(self, layer_idx: int, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        self.encoded_keys.setdefault(layer_idx, []).append(self.codec.encode_keys(k_t))
        if self.quantize_values:
            self.encoded_values.setdefault(layer_idx, []).append(self.codec.encode_values(v_t))
        else:
            self.raw_values.setdefault(layer_idx, []).append(v_t.detach())

    def get_keys(self, layer_idx: int) -> torch.Tensor:
        decoded = [self.codec.decode_keys(item) for item in self.encoded_keys.get(layer_idx, [])]
        return torch.cat(decoded, dim=-2)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        if self.quantize_values:
            decoded = [self.codec.decode_values(item) for item in self.encoded_values.get(layer_idx, [])]
            return torch.cat(decoded, dim=-2)
        return torch.cat(self.raw_values.get(layer_idx, []), dim=-2)

    def estimate_scores(self, layer_idx: int, q_t: torch.Tensor) -> torch.Tensor:
        chunks = [
            self.codec.estimator.turboquant(q=q_t, encoded_k=item)
            for item in self.encoded_keys.get(layer_idx, [])
        ]
        return torch.cat(chunks, dim=-1)
