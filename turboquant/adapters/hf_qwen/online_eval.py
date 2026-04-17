"""Minimal Qwen3 online-eval adapter for TurboQuant KV-cache experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DynamicCache,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen35,
    repeat_kv as repeat_kv_qwen35,
)

from turboquant.allocation import ChannelBitAllocation
from turboquant.hf_cache import CacheBackend, ExactCacheBackend, TurboQuantCacheBackend
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.research_extension.k_triality import TrialityRotationArtifact, load_triality_proxy_rotations
from turboquant.research_extension.triality_proxy import TrialityProxyProd
from turboquant.runtime import LOCAL_CAPTURE_MODEL_PATH
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantMSEConfig, TurboQuantProdConfig, ValueCodecConfig

SUPPORTED_QWEN_ONLINE_MODES = (
    "exact",
    "key_only_random",
    "full_kv",
    "asym_q8_turbo4",
    "key_only_block_so8_triality_vector",
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype name: {dtype}") from exc


def _stage1_allocation(bits: float, width: int, *, qjl_bits: int = 1) -> ChannelBitAllocation | None:
    if float(bits).is_integer():
        return None
    effective_bits = bits - qjl_bits
    if effective_bits <= 0.0:
        raise ValueError("effective Stage 1 bits must remain positive")
    return ChannelBitAllocation.preset(effective_bits=effective_bits, width=width)


@dataclass(frozen=True, slots=True)
class QwenOnlineEvalConfig:
    """Configuration for the Qwen3 online TurboQuant evaluation shim."""

    mode: str
    bits: float = 4.0
    model_name_or_path: str = LOCAL_CAPTURE_MODEL_PATH
    tokenizer_name_or_path: str | None = None
    device: str | torch.device = "cpu"
    torch_dtype: str | torch.dtype = "float32"
    weight_load: str = "4bit"
    trust_remote_code: bool = False
    device_map: str | None = None
    triality_rotation_dir: str | Path | None = None

    def resolved_device(self) -> torch.device:
        return torch.device(self.device)

    def resolved_dtype(self) -> torch.dtype:
        return _torch_dtype(self.torch_dtype)


@dataclass(slots=True)
class _Q8ProxyBatch:
    indices: torch.Tensor
    scale: torch.Tensor


def _q8_proxy_quantize(x: torch.Tensor) -> _Q8ProxyBatch:
    if not x.is_floating_point():
        raise ValueError("q8 proxy expects a floating-point tensor")
    if x.shape[-1] <= 0:
        raise ValueError("q8 proxy requires a positive head dimension")
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    indices = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return _Q8ProxyBatch(indices=indices, scale=scale)


def _q8_proxy_dequantize(encoded: _Q8ProxyBatch) -> torch.Tensor:
    return encoded.indices.to(dtype=encoded.scale.dtype) * encoded.scale


class _AsymmetricQ8TurboCacheBackend(CacheBackend):
    """Evaluation-only q8-key / TurboQuant-MSE-value cache backend."""

    def __init__(
        self,
        *,
        layer_idx: int,
        value_bits: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        rotation_seed: int,
    ) -> None:
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        self.encoded_keys: list[_Q8ProxyBatch] = []
        self.encoded_values: list[Any] = []
        self.value_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=head_dim,
                bits=value_bits,
                device=str(device),
                dtype=_dtype_name(dtype),
                rotation_policy="block_so8_learned",
                rotation_seed=rotation_seed,
            )
        )
        self._value_quantizer_fitted = False

    def append(self, layer_idx: int, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        if k_t.device != self.device or v_t.device != self.device:
            raise ValueError(f"Expected cache tensors on {self.device}, got {k_t.device} and {v_t.device}")
        if k_t.dtype != self.dtype or v_t.dtype != self.dtype:
            raise ValueError(f"Expected cache tensors with dtype {self.dtype}, got {k_t.dtype} and {v_t.dtype}")
        self.encoded_keys.append(_q8_proxy_quantize(k_t.detach()))
        if not self._value_quantizer_fitted:
            self.value_quantizer.fit_rotation(v_t)
            self._value_quantizer_fitted = True
        self.encoded_values.append(self.value_quantizer.quantize(v_t.detach()))

    def get_keys(self, layer_idx: int) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        return torch.cat([_q8_proxy_dequantize(item) for item in self.encoded_keys], dim=-2)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        return torch.cat([self.value_quantizer.dequantize(item) for item in self.encoded_values], dim=-2)

    def estimate_scores(self, layer_idx: int, q_t: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        if q_t.device != self.device:
            raise ValueError(f"Expected query tensor on {self.device}, got {q_t.device}")
        if q_t.dtype != self.dtype:
            raise ValueError(f"Expected query dtype {self.dtype}, got {q_t.dtype}")
        return torch.einsum("...qd,...sd->...qs", q_t, self.get_keys(layer_idx))


class _TrialityProxyCacheBackend(CacheBackend):
    """Evaluation-only triality-vector key-only backend using fitted proxy rotations."""

    def __init__(
        self,
        *,
        layer_idx: int,
        proxy: TrialityProxyProd,
        allocation: ChannelBitAllocation | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.layer_idx = layer_idx
        self.proxy = proxy
        self.allocation = allocation
        self.device = device
        self.dtype = dtype
        self.encoded_keys: list[Any] = []
        self.raw_values: list[torch.Tensor] = []

    def append(self, layer_idx: int, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        if k_t.device != self.device or v_t.device != self.device:
            raise ValueError(f"Expected cache tensors on {self.device}, got {k_t.device} and {v_t.device}")
        if k_t.dtype != self.dtype or v_t.dtype != self.dtype:
            raise ValueError(f"Expected cache tensors with dtype {self.dtype}, got {k_t.dtype} and {v_t.dtype}")
        self.encoded_keys.append(self.proxy.quantize(k_t.detach(), allocation=self.allocation))
        self.raw_values.append(v_t.detach())

    def get_keys(self, layer_idx: int) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        return torch.cat([self.proxy.dequantize(item) for item in self.encoded_keys], dim=-2)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        return torch.cat(self.raw_values, dim=-2)

    def estimate_scores(self, layer_idx: int, q_t: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.layer_idx:
            raise ValueError(f"Expected layer {self.layer_idx}, got {layer_idx}")
        if q_t.device != self.device:
            raise ValueError(f"Expected query tensor on {self.device}, got {q_t.device}")
        if q_t.dtype != self.dtype:
            raise ValueError(f"Expected query dtype {self.dtype}, got {q_t.dtype}")
        scores = [self.proxy.pairwise_estimate(q_t, encoded) for encoded in self.encoded_keys]
        return torch.cat(scores, dim=-1)


class TurboQuantQwenCache(Cache):
    """Layer-indexed cache that stores per-layer TurboQuant backends.

    Shapes:
    - key/value tensors appended via ``update``: ``[batch, kv_heads, seq, head_dim]``
    - score tensors returned by backends: ``[batch, kv_heads, query, seq]`` or broadcast-compatible
    """

    def __init__(
        self,
        *,
        backend_factories: dict[int, Callable[[], CacheBackend]],
        num_hidden_layers: int,
    ) -> None:
        super().__init__(layers=[])
        self._backend_factories = backend_factories
        self._num_hidden_layers = num_hidden_layers
        self.reset()

    def __len__(self) -> int:
        return self._num_hidden_layers

    def get_backend(self, layer_idx: int) -> CacheBackend:
        try:
            return self.backends[layer_idx]
        except KeyError as exc:
            raise KeyError(f"No TurboQuant backend for layer {layer_idx}") from exc

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[0] != 1:
            raise ValueError("TurboQuantQwenCache currently supports batch size 1 only")
        if key_states.device != value_states.device:
            raise ValueError(f"Expected key/value on the same device, got {key_states.device} and {value_states.device}")
        if key_states.dtype != value_states.dtype:
            raise ValueError(f"Expected matching key/value dtype, got {key_states.dtype} and {value_states.dtype}")
        backend = self.get_backend(layer_idx)
        backend.append(layer_idx, key_states.detach(), value_states.detach())
        self.seq_lengths[layer_idx] += int(key_states.shape[-2])
        return backend.get_keys(layer_idx), backend.get_values(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.seq_lengths.get(layer_idx, 0))

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def reset(self) -> None:
        self.backends = {layer_idx: factory() for layer_idx, factory in self._backend_factories.items()}
        self.seq_lengths = {layer_idx: 0 for layer_idx in self._backend_factories}

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("TurboQuantQwenCache does not support beam-search cache reordering")

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("TurboQuantQwenCache does not support cache cropping")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        raise NotImplementedError("TurboQuantQwenCache does not support batch selection")


class TurboQuantQwen35Cache(Qwen3_5DynamicCache):
    """Hybrid Qwen3.5 cache: keep linear-attention state exact, quantize full-attention KV."""

    def __init__(
        self,
        *,
        backend_factories: dict[int, Callable[[], CacheBackend]],
        model_config,
    ) -> None:
        super().__init__(model_config)
        self._backend_factories = backend_factories
        self.backends = {layer_idx: factory() for layer_idx, factory in backend_factories.items()}

    def get_backend(self, layer_idx: int) -> CacheBackend:
        try:
            return self.backends[layer_idx]
        except KeyError as exc:
            raise KeyError(f"No TurboQuant backend for layer {layer_idx}") from exc

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_types[layer_idx] != "full_attention":
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        if key_states.shape[0] != 1:
            raise ValueError("TurboQuantQwen35Cache currently supports batch size 1 only")
        if key_states.device != value_states.device:
            raise ValueError(f"Expected key/value on the same device, got {key_states.device} and {value_states.device}")
        if key_states.dtype != value_states.dtype:
            raise ValueError(f"Expected matching key/value dtype, got {key_states.dtype} and {value_states.dtype}")
        backend = self.get_backend(layer_idx)
        backend.append(layer_idx, key_states.detach(), value_states.detach())
        self.key_cache[layer_idx] = backend.get_keys(layer_idx)
        self.value_cache[layer_idx] = backend.get_values(layer_idx)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _mode_seed(layer_idx: int, bits: float, salt: int) -> int:
    return 40_000 + (layer_idx * 1_009) + int(bits * 100) + salt


def _build_codec(
    *,
    head_dim: int,
    bits: float,
    rotation_policy: str,
    device: torch.device,
    dtype: torch.dtype,
    rotation_seed: int,
    qjl_seed: int,
) -> KVCodec:
    integer_bits = int(math.floor(bits))
    return KVCodec(
        KVCodecConfig(
            head_dim=head_dim,
            key_bits=integer_bits,
            value_bits=integer_bits,
            mixed_key_bits=bits if not float(bits).is_integer() else None,
            mixed_value_bits=bits if not float(bits).is_integer() else None,
            device=str(device),
            dtype=_dtype_name(dtype),
            rotation_policy=rotation_policy,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
            value_codec=ValueCodecConfig(base_bits=integer_bits),
        )
    )


def _load_triality_artifacts(config: QwenOnlineEvalConfig) -> dict[tuple[int, float, str], TrialityRotationArtifact] | None:
    if config.mode != "key_only_block_so8_triality_vector":
        return None
    if config.triality_rotation_dir is None:
        raise ValueError("key_only_block_so8_triality_vector requires triality_rotation_dir")
    return load_triality_proxy_rotations(Path(config.triality_rotation_dir))


def _build_backend_factory(
    *,
    config: QwenOnlineEvalConfig,
    layer_idx: int,
    head_dim: int,
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact] | None,
) -> Callable[[], CacheBackend]:
    device = config.resolved_device()
    dtype = config.resolved_dtype()
    bits = float(config.bits)

    if config.mode == "exact":
        return ExactCacheBackend

    if config.mode == "key_only_random":
        rotation_seed = _mode_seed(layer_idx, bits, 11)
        qjl_seed = _mode_seed(layer_idx, bits, 83)

        def factory() -> CacheBackend:
            codec = _build_codec(
                head_dim=head_dim,
                bits=bits,
                rotation_policy="random_haar",
                device=device,
                dtype=dtype,
                rotation_seed=rotation_seed,
                qjl_seed=qjl_seed,
            )
            return TurboQuantCacheBackend(codec, quantize_values=False)

        return factory

    if config.mode == "full_kv":
        rotation_seed = _mode_seed(layer_idx, bits, 11)
        qjl_seed = _mode_seed(layer_idx, bits, 83)

        def factory() -> CacheBackend:
            codec = _build_codec(
                head_dim=head_dim,
                bits=bits,
                rotation_policy="random_haar",
                device=device,
                dtype=dtype,
                rotation_seed=rotation_seed,
                qjl_seed=qjl_seed,
            )
            return TurboQuantCacheBackend(codec, quantize_values=True)

        return factory

    if config.mode == "asym_q8_turbo4":
        rotation_seed = _mode_seed(layer_idx, 4.0, 401)

        def factory() -> CacheBackend:
            return _AsymmetricQ8TurboCacheBackend(
                layer_idx=layer_idx,
                value_bits=4,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
                rotation_seed=rotation_seed,
            )

        return factory

    if config.mode == "key_only_block_so8_triality_vector":
        if artifacts is None:
            raise ValueError("Triality-vector mode requires loaded artifacts")
        key = (layer_idx, bits, "vector")
        if key not in artifacts:
            raise KeyError(f"Missing triality artifact for layer={layer_idx}, bits={bits:g}, view=vector")
        artifact = artifacts[key]
        allocation = _stage1_allocation(bits, width=head_dim)

        def factory() -> CacheBackend:
            quantizer = TurboQuantProd(
                TurboQuantProdConfig(
                    dim=head_dim,
                    total_bits=int(math.floor(bits)),
                    rotation_seed=artifact.rotation_seed,
                    rotation_policy="block_so8_static",
                    qjl_seed=artifact.qjl_seed,
                    device=str(device),
                    dtype=_dtype_name(dtype),
                )
            )
            proxy = TrialityProxyProd(quantizer=quantizer, view="vector")
            proxy.set_rotation(artifact.rotation.to(device=device, dtype=dtype))
            return _TrialityProxyCacheBackend(
                layer_idx=layer_idx,
                proxy=proxy,
                allocation=allocation,
                device=device,
                dtype=dtype,
            )

        return factory

    raise ValueError(f"Unsupported Qwen online mode: {config.mode!r}")


def _estimate_scores_with_gqa(
    *,
    backend: CacheBackend,
    layer_idx: int,
    query_states: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if num_key_value_groups == 1:
        return backend.estimate_scores(layer_idx, query_states)
    batch, num_heads, query_length, head_dim = query_states.shape
    if num_heads % num_key_value_groups != 0:
        raise ValueError(
            f"Expected num_heads divisible by num_key_value_groups, got {num_heads} and {num_key_value_groups}"
        )
    num_key_value_heads = num_heads // num_key_value_groups
    regrouped_queries = query_states.reshape(batch, num_key_value_heads, num_key_value_groups, query_length, head_dim)
    estimated = backend.estimate_scores(layer_idx, regrouped_queries)
    return estimated.reshape(batch, num_heads, query_length, estimated.shape[-1])


def _patched_qwen_attention_forward(
    self: Qwen3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hidden_states.shape[0] != 1:
        raise ValueError("TurboQuant online Qwen evaluation currently supports batch size 1 only")
    if past_key_values is None or not isinstance(past_key_values, TurboQuantQwenCache):
        raise ValueError(
            "TurboQuant online Qwen evaluation requires TurboQuantQwenCache. "
            "Call model.make_turboquant_cache() and pass it as past_key_values."
        )
    if self.layer_idx is None:
        raise ValueError("Qwen3Attention layer_idx must be set for TurboQuant online evaluation")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    past_key_values.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    backend = past_key_values.get_backend(self.layer_idx)
    estimated_scores = _estimate_scores_with_gqa(
        backend=backend,
        layer_idx=self.layer_idx,
        query_states=query_states,
        num_key_value_groups=self.num_key_value_groups,
    )
    estimated_scores = estimated_scores * float(self.scaling)
    if attention_mask is not None:
        estimated_scores = estimated_scores + attention_mask

    attn_weights = torch.softmax(estimated_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, train=self.training)
    value_states = repeat_kv(backend.get_values(self.layer_idx), self.num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _patched_qwen35_attention_forward(
    self: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hidden_states.shape[0] != 1:
        raise ValueError("TurboQuant online Qwen3.5 evaluation currently supports batch size 1 only")
    if past_key_values is None or not isinstance(past_key_values, TurboQuantQwen35Cache):
        raise ValueError(
            "TurboQuant online Qwen3.5 evaluation requires TurboQuantQwen35Cache. "
            "Call model.make_turboquant_cache() and pass it as past_key_values."
        )
    if self.layer_idx is None:
        raise ValueError("Qwen3_5Attention layer_idx must be set for TurboQuant online evaluation")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_qwen35(query_states, key_states, cos, sin)
    past_key_values.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    backend = past_key_values.get_backend(self.layer_idx)
    estimated_scores = _estimate_scores_with_gqa(
        backend=backend,
        layer_idx=self.layer_idx,
        query_states=query_states,
        num_key_value_groups=self.num_key_value_groups,
    )
    estimated_scores = estimated_scores * float(self.scaling)
    if attention_mask is not None:
        estimated_scores = estimated_scores + attention_mask

    attn_weights = torch.softmax(estimated_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, train=self.training)
    value_states = repeat_kv_qwen35(backend.get_values(self.layer_idx), self.num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_qwen_for_online_eval(
    model: PreTrainedModel,
    config: QwenOnlineEvalConfig,
) -> PreTrainedModel:
    """Patch a Qwen3 causal LM in-place for TurboQuant online evaluation."""

    if config.mode not in SUPPORTED_QWEN_ONLINE_MODES:
        raise ValueError(f"Unsupported Qwen online mode: {config.mode!r}")
    model_type = str(getattr(model.config, "model_type", ""))
    if not (model_type == "qwen3" or model_type.startswith("qwen3_5")):
        raise ValueError(f"Expected a Qwen3/Qwen3.5 model, got model_type={model_type!r}")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a decoder-only Qwen3 model with model.layers")

    model.config._attn_implementation = "eager"
    artifacts = _load_triality_artifacts(config)

    backend_factories: dict[int, Callable[[], CacheBackend]] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        backend_factories[layer_idx] = _build_backend_factory(
            config=config,
            layer_idx=layer_idx,
            head_dim=int(attn.head_dim),
            artifacts=artifacts,
        )
        if not getattr(attn, "_turboquant_online_patched", False):
            attn._turboquant_online_patched = True
            attn._turboquant_online_original_forward = attn.forward
            attn.forward = MethodType(
                _patched_qwen_attention_forward if model_type == "qwen3" else _patched_qwen35_attention_forward,
                attn,
            )

    def make_turboquant_cache(self: PreTrainedModel) -> TurboQuantQwenCache:
        if model_type == "qwen3":
            return TurboQuantQwenCache(
                backend_factories=backend_factories,
                num_hidden_layers=int(self.config.num_hidden_layers),
            )
        return TurboQuantQwen35Cache(
            backend_factories=backend_factories,
            model_config=self.config,
        )

    model.make_turboquant_cache = MethodType(make_turboquant_cache, model)
    model._turboquant_online_eval_config = config
    return model


def _build_model_load_kwargs(config: QwenOnlineEvalConfig) -> dict[str, Any]:
    """Build explicit model-loading kwargs for the requested weight path."""

    torch_dtype = config.resolved_dtype()
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }
    if config.weight_load in {"4bit", "8bit"}:
        if config.resolved_device().type != "cuda":
            raise ValueError("4bit/8bit online eval requires a CUDA device")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.weight_load == "4bit",
            load_in_8bit=config.weight_load == "8bit",
            bnb_4bit_compute_dtype=torch_dtype if config.weight_load == "4bit" else None,
        )
        kwargs["device_map"] = config.device_map or "auto"
        kwargs["torch_dtype"] = torch_dtype
        return kwargs
    if config.weight_load == "none":
        kwargs["torch_dtype"] = torch_dtype
        if config.device_map is not None:
            kwargs["device_map"] = config.device_map
        return kwargs
    raise ValueError(f"Unsupported weight_load: {config.weight_load!r}")


def load_qwen_online_eval_model(
    config: QwenOnlineEvalConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load and patch a Qwen3 model/tokenizer pair for online TurboQuant evaluation."""

    tokenizer_source = config.tokenizer_name_or_path or config.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **_build_model_load_kwargs(config),
    )
    if config.weight_load == "none" and config.device_map is None:
        model.to(config.resolved_device())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=config.trust_remote_code,
    )
    return patch_qwen_for_online_eval(model, config), tokenizer


__all__ = [
    "QwenOnlineEvalConfig",
    "SUPPORTED_QWEN_ONLINE_MODES",
    "TurboQuantQwenCache",
    "load_qwen_online_eval_model",
    "patch_qwen_for_online_eval",
]
