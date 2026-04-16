"""Offline KV replay and summary helpers for paper-faithful TurboQuant evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Callable, cast

import pandas as pd
from scipy import stats
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.attention_metrics import summarize_attention_scores
from turboquant.capture import CaptureMetadata, load_capture_metadata
from turboquant.kv_codec import AttentionScoreEstimator, KVCodec, KVCodecConfig
from turboquant.reporting import summarize_metric_trials
from turboquant.types import RotationPolicy, TurboQuantMSEConfig, TurboQuantProdConfig
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import SensitivitySpec
from turboquant.types import ValueCodecConfig
from turboquant.value_codec import ProtectedValueCodec


METRIC_COLUMNS = [
    "logit_cosine_similarity",
    "logit_mae",
    "logit_mse",
    "next_logit_kl",
    "logit_spearman",
    "logit_top1_match",
    "logit_top5_match",
    "logit_top5_overlap",
    "hidden_cosine_similarity",
    "hidden_mae",
    "hidden_mse",
    "attention_output_relative_error",
    "memory_bits",
    "memory_ratio_vs_exact",
    "prefill_seconds",
    "decode_seconds",
    "peak_vram_mb",
]

QWEN_3060_MATRIX_MODES = (
    "exact",
    "key_only_random",
    "full_kv",
    "asym_q8_turbo4",
    "asym_q8_turbo3",
    "multiscreen_relevance",
    "key_only_block_so8_triality_vector",
)

QWEN_3060_STAT_METRICS = (
    "logit_cosine_similarity",
    "next_logit_kl",
    "hidden_cosine_similarity",
    "memory_ratio_vs_exact",
)

QWEN_3060_PAIRWISE_BASELINES = ("exact", "asym_q8_turbo4")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def raw_storage_bits(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size() * 8)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def select_queries(keys: torch.Tensor, seed: int, query_count: int = 4) -> torch.Tensor:
    """Choose representative queries from keys with a small amount of Gaussian noise."""

    if keys.shape[-2] == 0:
        raise ValueError("keys must contain at least one sequence position")
    index_generator = torch.Generator(device="cpu")
    index_generator.manual_seed(seed)
    chosen = min(query_count, keys.shape[-2])
    indices = torch.randperm(keys.shape[-2], generator=index_generator)[:chosen]
    base = keys.index_select(dim=-2, index=indices.to(keys.device))
    noise_generator = torch.Generator(device=keys.device.type if keys.device.type == "cuda" else "cpu")
    noise_generator.manual_seed(seed + 1)
    noise = 0.01 * torch.randn(base.shape, generator=noise_generator, dtype=keys.dtype, device=keys.device)
    return base + noise


def scaled_attention_output(logits: torch.Tensor, values: torch.Tensor, head_dim: int) -> torch.Tensor:
    weights = torch.softmax(logits / math.sqrt(float(head_dim)), dim=-1)
    return torch.einsum("...qs,...sd->...qd", weights, values)


def _seed_value(trial: int, layer_idx: int, salt: int) -> int:
    return 20_000 + (trial * 1_009) + (layer_idx * 131) + salt


def _mse_allocation(bits: float, width: int) -> ChannelBitAllocation | None:
    if float(bits).is_integer():
        return None
    return ChannelBitAllocation.preset(effective_bits=bits, width=width)


def _prod_allocation(bits: float, width: int) -> ChannelBitAllocation | None:
    if float(bits).is_integer():
        return None
    effective_bits = bits - 1.0
    if effective_bits <= 0.0:
        raise ValueError("Prod effective Stage 1 bits must remain positive")
    return ChannelBitAllocation.preset(effective_bits=effective_bits, width=width)


def _base_kv_codec(
    *,
    keys: torch.Tensor,
    bit_value: float,
    rotation_policy: str,
    rotation_seed: int,
    qjl_seed: int,
    value_codec: ValueCodecConfig | None = None,
) -> KVCodec:
    integer_bits = int(math.floor(bit_value))
    return KVCodec(
        KVCodecConfig(
            head_dim=keys.shape[-1],
            key_bits=integer_bits,
            value_bits=integer_bits,
            mixed_key_bits=bit_value if not float(bit_value).is_integer() else None,
            mixed_value_bits=bit_value if not float(bit_value).is_integer() else None,
            device=str(keys.device),
            dtype=dtype_name(keys.dtype),
            rotation_policy=rotation_policy,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
            value_codec=value_codec or ValueCodecConfig(base_bits=integer_bits),
        )
    )


def _attention_weights_from_exact(exact_logits: torch.Tensor, head_dim: int) -> torch.Tensor:
    return torch.softmax(exact_logits / math.sqrt(float(head_dim)), dim=-1)


def _build_value_mse_quantizer(
    *,
    values: torch.Tensor,
    bit_value: float,
    rotation_policy: str,
    rotation_seed: int,
) -> tuple[TurboQuantMSE, ChannelBitAllocation | None]:
    integer_bits = int(math.floor(bit_value))
    quantizer = TurboQuantMSE(
        TurboQuantMSEConfig(
            dim=values.shape[-1],
            bits=integer_bits,
            device=str(values.device),
            dtype=dtype_name(values.dtype),
            rotation_policy=rotation_policy,
            rotation_seed=rotation_seed,
        )
    )
    return quantizer, _mse_allocation(bit_value, values.shape[-1])


def _build_value_prod_quantizer(
    *,
    values: torch.Tensor,
    bit_value: float,
    rotation_policy: str,
    rotation_seed: int,
    qjl_seed: int,
) -> tuple[TurboQuantProd, ChannelBitAllocation | None]:
    integer_bits = int(math.floor(bit_value))
    quantizer = TurboQuantProd(
        TurboQuantProdConfig(
            dim=values.shape[-1],
            total_bits=integer_bits,
            device=str(values.device),
            dtype=dtype_name(values.dtype),
            rotation_policy=rotation_policy,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
        )
    )
    return quantizer, _prod_allocation(bit_value, values.shape[-1])


def _build_protected_value_codec(
    *,
    values: torch.Tensor,
    base_bits: int,
    rotation_policy: str,
    rotation_seed: int,
    protected_fraction: float,
    secondary_fraction: float,
    high_bits: int,
    low_rank_rank: int,
    score_source: str,
    channel_group_size: int = 8,
) -> ProtectedValueCodec:
    return ProtectedValueCodec(
        dim=values.shape[-1],
        config=ValueCodecConfig(
            base_bits=base_bits,
            protected_fraction=protected_fraction,
            secondary_fraction=secondary_fraction,
            high_bits=high_bits,
            low_rank_rank=low_rank_rank,
            channel_group_size=channel_group_size,
            sensitivity=SensitivitySpec(score_source=score_source),
        ),
        rotation_seed=rotation_seed,
        rotation_policy=rotation_policy,
        device=str(values.device),
        dtype=dtype_name(values.dtype),
    )


def melt_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    present_metrics = [column for column in METRIC_COLUMNS if column in frame.columns]
    id_vars = [column for column in frame.columns if column not in present_metrics]
    return frame.melt(id_vars=id_vars, value_vars=present_metrics, var_name="metric", value_name="value")


def summarize_trial_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    long_frame = melt_metric_rows(frame)
    return summarize_metric_trials(
        long_frame,
        group_columns=["dataset", "mode", "bit_setting", "bits", "metric"],
    )


def summarize_layer_thresholds(frame: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["dataset", "mode", "bit_setting", "bits", "trial"]
    for optional in ("capture_id", "prompt_label", "prompt_hash"):
        if optional in frame.columns:
            group_columns.append(optional)
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        ordered = group.sort_values("layer")
        mask = ordered[metric] < threshold
        first_layer = int(ordered.loc[mask, "layer"].iloc[0]) if mask.any() else -1
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        record.update(
            {
                "metric": metric,
                "threshold": threshold,
                "trial": int(record["trial"]),
                "value": first_layer,
            }
        )
        rows.append(record)
    threshold_frame = pd.DataFrame(rows)
    if threshold_frame.empty:
        return threshold_frame
    return summarize_metric_trials(
        threshold_frame,
        group_columns=[column for column in threshold_frame.columns if column not in {"trial", "value"}],
    )


@dataclass(slots=True)
class CapturedLayerBundle:
    metadata: CaptureMetadata
    capture_dir: Path
    layer_idx: int
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True)
class Q8ProxyBatch:
    """Simple per-vector symmetric int8 proxy used for the 3060 asymmetric offline lane."""

    indices: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, ...]

    def total_bits(self) -> int:
        return int(self.indices.numel() * 8) + int(self.scale.numel() * self.scale.element_size() * 8)


def _q8_proxy_quantize(x: torch.Tensor) -> Q8ProxyBatch:
    """Quantize ``x`` with a per-vector symmetric int8 proxy.

    Shapes:
    - ``x``: ``[..., dim]``
    - ``scale``: ``[..., 1]``
    - ``indices``: ``[..., dim]`` packed as int8
    """

    if not x.is_floating_point():
        raise ValueError("q8 proxy expects a floating-point tensor")
    if x.shape[-1] <= 0:
        raise ValueError("q8 proxy requires a positive head dimension")
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    indices = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return Q8ProxyBatch(indices=indices, scale=scale, shape=tuple(x.shape))


def _q8_proxy_dequantize(encoded: Q8ProxyBatch) -> torch.Tensor:
    return encoded.indices.to(dtype=encoded.scale.dtype) * encoded.scale


def _evaluate_mode(
    *,
    dataset: str,
    layer_idx: int,
    trial: int,
    bit_setting: str,
    bits: float | None,
    mode: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    codec: KVCodec | None,
    value_mode: str,
    calibrate_codec: bool,
    mode_metadata: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    device = keys.device
    exact_logits = codec.estimator.exact(queries, keys) if codec is not None else torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=keys.shape[-1])
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    if mode == "exact":
        _sync_if_cuda(device)
        _reset_peak_if_cuda(device)
        started = time.perf_counter()
        measured_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
        measured_hidden = scaled_attention_output(measured_logits, values, head_dim=keys.shape[-1])
        _sync_if_cuda(device)
        prefill_seconds = time.perf_counter() - started

        _sync_if_cuda(device)
        decode_started = time.perf_counter()
        last_query = queries[..., -1:, :]
        decode_logits = torch.einsum("...qd,...sd->...qs", last_query, keys)
        _ = scaled_attention_output(decode_logits, values, head_dim=keys.shape[-1])
        _sync_if_cuda(device)
        decode_seconds = time.perf_counter() - decode_started
        row = {
            "dataset": dataset,
            "trial": trial,
            "layer": layer_idx,
            "mode": mode,
            "bit_setting": bit_setting,
            "bits": float("nan"),
            "logit_cosine_similarity": 1.0,
            "logit_mae": 0.0,
            "logit_mse": 0.0,
            "next_logit_kl": 0.0,
            "logit_spearman": 1.0,
            "logit_top1_match": 1.0,
            "logit_top5_match": 1.0,
            "logit_top5_overlap": 1.0,
            "hidden_cosine_similarity": 1.0,
            "hidden_mae": 0.0,
            "hidden_mse": 0.0,
            "attention_output_relative_error": 0.0,
            "memory_bits": float(exact_memory_bits),
            "memory_ratio_vs_exact": 1.0,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "peak_vram_mb": _peak_vram_mb(device),
        }
        if mode_metadata is not None:
            row.update(mode_metadata)
        return row

    if codec is None:
        raise ValueError("codec is required for quantized modes")

    if calibrate_codec:
        codec.calibrate(keys=keys, values=values, queries=queries)

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = codec.encode_keys(keys)
    estimated_logits = codec.estimator.turboquant(queries, encoded_keys)
    decoded_values = values
    encoded_values = None
    protected_values = None
    if value_mode == "full_kv":
        encoded_values = codec.encode_values(values)
        decoded_values = codec.decode_values(encoded_values)
    elif value_mode == "protected":
        protected_values = codec.encode_protected_values(values)
        decoded_values = codec.decode_protected_values(protected_values)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = codec.estimator.turboquant(last_query, encoded_keys)
    if value_mode == "full_kv" and encoded_values is not None:
        _ = scaled_attention_output(decode_logits, decoded_values, head_dim=keys.shape[-1])
    elif value_mode == "protected" and protected_values is not None:
        _ = scaled_attention_output(decode_logits, decoded_values, head_dim=keys.shape[-1])
    else:
        _ = scaled_attention_output(decode_logits, values, head_dim=keys.shape[-1])
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, decoded_values, head_dim=keys.shape[-1])
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)

    memory_bits = codec.key_storage_bits(encoded_keys) + raw_storage_bits(values)
    if value_mode == "full_kv" and encoded_values is not None:
        memory_bits = codec.key_storage_bits(encoded_keys) + codec.value_storage_bits(encoded_values)
    elif value_mode == "protected" and protected_values is not None:
        memory_bits = codec.key_storage_bits(encoded_keys) + codec.protected_value_storage_bits(protected_values)

    row = {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": mode,
        "bit_setting": bit_setting,
        "bits": float(bits) if bits is not None else float("nan"),
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "next_logit_kl": logit_metrics["kl_divergence"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "attention_output_relative_error": hidden_metrics["relative_fro_error"],
        "memory_bits": float(memory_bits),
        "memory_ratio_vs_exact": float(memory_bits) / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
    }
    if mode_metadata is not None:
        row.update(mode_metadata)
    return row


def evaluate_asymmetric_q8_value_attention_row(
    *,
    dataset: str,
    trial: int,
    layer_idx: int,
    mode: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor | None = None,
    bit_setting: str | None = None,
    bits: float | None = None,
    eval_device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Evaluate the 3060 V-first asymmetric proxy modes.

    ``asym_q8_turbo4`` and ``asym_q8_turbo3`` use:
    - K side: per-vector symmetric int8 proxy to approximate a q8 cache path
    - V side: Stage-1 TurboQuant reconstruction with learned block-SO(8) rotation
    """

    if mode not in {"asym_q8_turbo4", "asym_q8_turbo3"}:
        raise ValueError(f"Unsupported asymmetric mode: {mode!r}")

    target_device = torch.device(eval_device) if eval_device is not None else keys.device
    if keys.device != target_device:
        keys = keys.to(target_device)
    if values.device != target_device:
        values = values.to(target_device)

    if queries is None:
        query_seed = _seed_value(trial, layer_idx, 503 if mode == "asym_q8_turbo4" else 509)
        queries = select_queries(keys, seed=query_seed)

    value_bits = 4 if mode == "asym_q8_turbo4" else 3
    comparison_bits = float(value_bits) if bits is None else float(bits)
    comparison_bit_setting = f"{value_bits:g}" if bit_setting is None else bit_setting
    rotation_seed = _seed_value(trial, layer_idx, 401 if value_bits == 4 else 307)

    device = keys.device
    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=keys.shape[-1])
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    q8_keys = _q8_proxy_quantize(keys)
    value_quantizer = TurboQuantMSE(
        TurboQuantMSEConfig(
            dim=values.shape[-1],
            bits=value_bits,
            device=str(device),
            dtype=dtype_name(values.dtype),
            rotation_policy="block_so8_learned",
            rotation_seed=rotation_seed,
        )
    )
    value_quantizer.fit_rotation(values, queries=queries)

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    decoded_keys = _q8_proxy_dequantize(q8_keys)
    encoded_values = value_quantizer.quantize(values)
    decoded_values = value_quantizer.dequantize(encoded_values)
    estimated_logits = torch.einsum("...qd,...sd->...qs", queries, decoded_keys)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = torch.einsum("...qd,...sd->...qs", last_query, decoded_keys)
    _ = scaled_attention_output(decode_logits, decoded_values, head_dim=keys.shape[-1])
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, decoded_values, head_dim=keys.shape[-1])
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)

    memory_bits = float(q8_keys.total_bits() + encoded_values.total_bits())
    return {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": mode,
        "bit_setting": comparison_bit_setting,
        "bits": comparison_bits,
        "mode_reference_bits": float(value_bits),
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "next_logit_kl": logit_metrics["kl_divergence"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "attention_output_relative_error": hidden_metrics["relative_fro_error"],
        "memory_bits": memory_bits,
        "memory_ratio_vs_exact": memory_bits / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
        "key_mode": "q8_proxy",
        "value_mode": mode,
        "value_rotation_policy": "block_so8_learned",
        "rotation_seed": rotation_seed,
        "qjl_seed": 0,
    }


def evaluate_multiscreen_relevance_attention_row(
    *,
    dataset: str,
    trial: int,
    layer_idx: int,
    bit_value: float,
    keys: torch.Tensor,  # [batch, heads, seq, head_dim]
    values: torch.Tensor,
    queries: torch.Tensor,
    allocation: ChannelBitAllocation,
    rotation_policy: str,
    rotation_seed: int,
    qjl_seed: int,
) -> dict[str, float | int | str]:
    """Key-only paper-style attention row with Multiscreen-derived per-position Stage-1 bitwidths."""

    from turboquant.research_extension.multiscreen_kv import (
        compute_k_relevance,
        expand_relevance_bitwidths_to_key_shape,
    )

    device = keys.device
    head_dim = keys.shape[-1]
    integer_bits = int(math.floor(bit_value))
    key_prod = TurboQuantProd(
        TurboQuantProdConfig(
            dim=head_dim,
            total_bits=integer_bits,
            rotation_seed=rotation_seed,
            rotation_policy=cast(RotationPolicy, rotation_policy),
            qjl_seed=qjl_seed,
            device=str(device),
            dtype=dtype_name(keys.dtype),
        )
    )
    relevance = compute_k_relevance(queries, keys)  # [batch, heads, seq]
    bitwidths = expand_relevance_bitwidths_to_key_shape(relevance, allocation, head_dim)

    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=head_dim)
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    estimator = AttentionScoreEstimator(key_prod)
    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = key_prod.quantize_with_bitwidths(keys, bitwidths)
    estimated_logits = estimator.turboquant(queries, encoded_keys)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = estimator.turboquant(last_query, encoded_keys)
    _ = scaled_attention_output(decode_logits, values, head_dim=head_dim)
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, values, head_dim=head_dim)
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)

    memory_bits = float(encoded_keys.total_bits() + raw_storage_bits(values))
    bit_setting = f"{bit_value:g}"
    return {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": "multiscreen_relevance",
        "bit_setting": bit_setting,
        "bits": float(bit_value),
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "next_logit_kl": logit_metrics["kl_divergence"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "attention_output_relative_error": hidden_metrics["relative_fro_error"],
        "memory_bits": memory_bits,
        "memory_ratio_vs_exact": memory_bits / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
        "key_mode": "multiscreen_relevance",
        "value_mode": "exact",
        "value_rotation_policy": "exact",
        "rotation_seed": rotation_seed,
        "qjl_seed": qjl_seed,
        "multiscreen_regular_bits": allocation.regular_bits,
        "multiscreen_outlier_bits": allocation.outlier_bits,
        "multiscreen_outlier_count": allocation.outlier_count,
    }


def _evaluate_value_decoder_mode(
    *,
    dataset: str,
    layer_idx: int,
    trial: int,
    bit_setting: str,
    bits: float,
    mode: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    key_codec: KVCodec,
    value_strategy: str,
    value_rotation_policy: str,
    rotation_seed: int,
    qjl_seed: int,
    low_rank_rank: int = 0,
    protected_fraction: float = 0.10,
    secondary_fraction: float = 0.10,
    high_bits: int = 8,
    score_source: str = "attention-output-sensitivity",
    channel_group_size: int = 8,
) -> dict[str, float | int | str]:
    device = keys.device
    exact_logits = key_codec.estimator.exact(queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=keys.shape[-1])
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)
    attention_weights = _attention_weights_from_exact(exact_logits, keys.shape[-1])

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = key_codec.encode_keys(keys)
    estimated_logits = key_codec.estimator.turboquant(queries, encoded_keys)

    if value_strategy == "mse":
        value_quantizer, allocation = _build_value_mse_quantizer(
            values=values,
            bit_value=bits,
            rotation_policy=value_rotation_policy,
            rotation_seed=rotation_seed,
        )
        if value_rotation_policy == "block_so8_learned":
            value_quantizer.fit_rotation(values, queries=queries)
        encoded_values = value_quantizer.quantize(values, allocation=allocation)
        decoded_values = value_quantizer.dequantize(encoded_values)
        value_bits = encoded_values.total_bits()
    elif value_strategy == "prod":
        value_quantizer, allocation = _build_value_prod_quantizer(
            values=values,
            bit_value=bits,
            rotation_policy=value_rotation_policy,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
        )
        if value_rotation_policy == "block_so8_learned":
            value_quantizer.fit_rotation(values, queries=queries)
        encoded_values = value_quantizer.quantize(values, allocation=allocation)
        decoded_values = value_quantizer.transport_decode(encoded_values)
        value_bits = encoded_values.total_bits()
    elif value_strategy == "protected":
        value_codec = _build_protected_value_codec(
            values=values,
            base_bits=int(math.floor(bits)),
            rotation_policy=value_rotation_policy,
            rotation_seed=rotation_seed,
            protected_fraction=protected_fraction,
            secondary_fraction=secondary_fraction,
            high_bits=high_bits,
            low_rank_rank=low_rank_rank,
            score_source=score_source,
            channel_group_size=channel_group_size,
        )
        value_codec.calibrate(values, attention_weights=attention_weights)
        encoded_values = value_codec.encode(values)
        decoded_values = value_codec.decode(encoded_values)
        value_bits = value_codec.storage_bits(encoded_values)
    else:
        raise ValueError(f"Unsupported value_strategy={value_strategy!r}")

    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = key_codec.estimator.turboquant(last_query, encoded_keys)
    _ = scaled_attention_output(decode_logits, decoded_values, head_dim=keys.shape[-1])
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, decoded_values, head_dim=keys.shape[-1])
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)

    row = {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": mode,
        "bit_setting": bit_setting,
        "bits": float(bits),
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "next_logit_kl": logit_metrics["kl_divergence"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "attention_output_relative_error": hidden_metrics["relative_fro_error"],
        "memory_bits": float(key_codec.key_storage_bits(encoded_keys) + value_bits),
        "memory_ratio_vs_exact": float(key_codec.key_storage_bits(encoded_keys) + value_bits) / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
        "key_mode": "key_only_block_so8_learned",
        "value_mode": value_strategy,
        "value_rotation_policy": value_rotation_policy,
        "rotation_seed": rotation_seed,
        "qjl_seed": qjl_seed,
    }
    return row


def evaluate_layer_grid(
    *,
    dataset: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    trial: int,
    layer_idx: int,
    bit_grid: list[float],
    key_only_default: bool = True,
    eval_device: str | torch.device | None = None,
    progress_callback: Callable[[dict[str, float | int | str]], None] | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device) if eval_device is not None else keys.device
    if keys.device != target_device:
        keys = keys.to(target_device)
    if values.device != target_device:
        values = values.to(target_device)
    queries = select_queries(keys, seed=10_000 + (trial * 257) + layer_idx)
    exact_row = _evaluate_mode(
        dataset=dataset,
        layer_idx=layer_idx,
        trial=trial,
        bit_setting="exact",
        bits=None,
        mode="exact",
        keys=keys,
        values=values,
        queries=queries,
        codec=None,
        value_mode="exact",
        calibrate_codec=False,
    )
    rows.append(exact_row)
    if progress_callback is not None:
        progress_callback(exact_row)
    for bit_value in bit_grid:
        bit_setting = f"{bit_value:g}"
        random_seed = _seed_value(trial, layer_idx, 11 + int(bit_value * 10))
        static_seed = _seed_value(trial, layer_idx, 29 + int(bit_value * 10))
        learned_seed = _seed_value(trial, layer_idx, 47 + int(bit_value * 10))
        qjl_seed = _seed_value(trial, layer_idx, 83 + int(bit_value * 10))

        key_only_specs = [
            ("key_only_random", "random_haar", "exact", False, 0, random_seed, qjl_seed),
            ("key_only_block_so8_static", "block_so8_static", "exact", False, 0, static_seed, qjl_seed),
            ("key_only_block_so8_learned", "block_so8_learned", "exact", True, 0, learned_seed, qjl_seed),
            ("full_kv", "random_haar", "full_kv", False, 0, random_seed, qjl_seed),
        ]
        for mode, rotation_policy, value_mode, calibrate_codec, low_rank_rank, rotation_seed, mode_qjl_seed in key_only_specs:
            codec = _base_kv_codec(
                keys=keys,
                bit_value=bit_value,
                rotation_policy=rotation_policy,
                rotation_seed=rotation_seed,
                qjl_seed=mode_qjl_seed,
                value_codec=ValueCodecConfig(base_bits=int(math.floor(bit_value)), low_rank_rank=low_rank_rank),
            )
            mode_row = _evaluate_mode(
                dataset=dataset,
                layer_idx=layer_idx,
                trial=trial,
                bit_setting=bit_setting,
                bits=bit_value,
                mode=mode,
                keys=keys,
                values=values,
                queries=queries,
                codec=codec,
                value_mode=value_mode,
                calibrate_codec=calibrate_codec,
                mode_metadata={
                    "key_mode": mode if mode.startswith("key_only") else "full_kv",
                    "value_mode": value_mode,
                    "value_rotation_policy": rotation_policy if value_mode == "full_kv" else "exact",
                    "rotation_seed": rotation_seed,
                    "qjl_seed": mode_qjl_seed,
                },
            )
            rows.append(mode_row)
            if progress_callback is not None:
                progress_callback(mode_row)

        learned_key_codec = _base_kv_codec(
            keys=keys,
            bit_value=bit_value,
            rotation_policy="block_so8_learned",
            rotation_seed=learned_seed,
            qjl_seed=qjl_seed,
            value_codec=ValueCodecConfig(base_bits=int(math.floor(bit_value))),
        )
        learned_key_codec.calibrate(keys=keys, values=values, queries=queries)

        value_specs = [
            ("v_mse_random", "mse", "random_haar", 0),
            ("v_mse_block_so8", "mse", "block_so8_static", 0),
            ("v_prod_random", "prod", "random_haar", 0),
            ("v_prod_block_so8", "prod", "block_so8_static", 0),
            ("protected_v", "protected", "block_so8_learned", 0),
            ("protected_v_lowrank", "protected", "block_so8_learned", 4),
        ]
        for mode, value_strategy, value_rotation_policy, low_rank_rank in value_specs:
            mode_row = _evaluate_value_decoder_mode(
                dataset=dataset,
                layer_idx=layer_idx,
                trial=trial,
                bit_setting=bit_setting,
                bits=bit_value,
                mode=mode,
                keys=keys,
                values=values,
                queries=queries,
                key_codec=learned_key_codec,
                value_strategy=value_strategy,
                value_rotation_policy=value_rotation_policy,
                rotation_seed=learned_seed,
                qjl_seed=qjl_seed,
                low_rank_rank=low_rank_rank,
            )
            rows.append(mode_row)
            if progress_callback is not None:
                progress_callback(mode_row)
    return rows


def compute_value_sensitivity_rows(
    *,
    dataset: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    trial: int,
    layer_idx: int,
    eval_device: str | torch.device | None = None,
    group_size: int = 8,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device) if eval_device is not None else keys.device
    if keys.device != target_device:
        keys = keys.to(target_device)
    if values.device != target_device:
        values = values.to(target_device)
    queries = select_queries(keys, seed=40_000 + (trial * 257) + layer_idx)
    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    attention_weights = _attention_weights_from_exact(exact_logits, keys.shape[-1])

    for score_source in ("attention-output-sensitivity", "teacher-gradient-proxy"):
        codec = _build_protected_value_codec(
            values=values,
            base_bits=2,
            rotation_policy="block_so8_learned",
            rotation_seed=_seed_value(trial, layer_idx, 191),
            protected_fraction=0.10,
            secondary_fraction=0.10,
            high_bits=8,
            low_rank_rank=0,
            score_source=score_source,
            channel_group_size=group_size,
        )
        codec.calibrate(values, attention_weights=attention_weights)
        channel_scores = codec.channel_sensitivity()
        group_scores = codec.group_sensitivity()
        if channel_scores is None or group_scores is None:
            continue
        layer_score = float(channel_scores.mean().item())
        rows.append(
            {
                "dataset": dataset,
                "trial": trial,
                "layer": layer_idx,
                "score_source": score_source,
                "granularity": "per-layer",
                "head_index": -1,
                "channel_index": -1,
                "group_index": -1,
                "score": layer_score,
            }
        )
        for head_index in range(channel_scores.shape[0]):
            rows.append(
                {
                    "dataset": dataset,
                    "trial": trial,
                    "layer": layer_idx,
                    "score_source": score_source,
                    "granularity": "per-head",
                    "head_index": head_index,
                    "channel_index": -1,
                    "group_index": -1,
                    "score": float(channel_scores[head_index].mean().item()),
                }
            )
            for group_index in range(group_scores.shape[-1]):
                rows.append(
                    {
                        "dataset": dataset,
                        "trial": trial,
                        "layer": layer_idx,
                        "score_source": score_source,
                        "granularity": "per-group",
                        "head_index": head_index,
                        "channel_index": -1,
                        "group_index": group_index,
                        "score": float(group_scores[head_index, group_index].item()),
                    }
                )
            for channel_index in range(channel_scores.shape[-1]):
                rows.append(
                    {
                        "dataset": dataset,
                        "trial": trial,
                        "layer": layer_idx,
                        "score_source": score_source,
                        "granularity": "per-channel",
                        "head_index": head_index,
                        "channel_index": channel_index,
                        "group_index": channel_index // group_size,
                        "score": float(channel_scores[head_index, channel_index].item()),
                    }
                )
    return rows


def summarize_value_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return summarize_metric_trials(
        frame.rename(columns={"score": "value"}),
        group_columns=["dataset", "score_source", "granularity", "layer", "head_index", "channel_index", "group_index"],
    )


def compose_sensitive_layer_policy_rows(
    *,
    trial_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    top_fraction: float = 0.25,
) -> pd.DataFrame:
    if trial_frame.empty or sensitivity_frame.empty:
        return pd.DataFrame()
    layer_scores = sensitivity_frame.loc[
        (sensitivity_frame["score_source"] == "teacher-gradient-proxy")
        & (sensitivity_frame["granularity"] == "per-layer")
    ].copy()
    if layer_scores.empty:
        return pd.DataFrame()
    ranked = layer_scores.groupby("layer", as_index=False)["score"].mean().sort_values("score", ascending=False)
    top_count = max(1, int(math.ceil(len(ranked) * top_fraction)))
    selected_layers = set(int(value) for value in ranked.head(top_count)["layer"].tolist())

    rows: list[dict[str, float | int | str]] = []
    exact_rows_all = trial_frame.loc[trial_frame["mode"] == "exact"]
    full_rows_all = trial_frame.loc[trial_frame["mode"] == "full_kv"]
    for bit_setting, bit_group in full_rows_all.groupby("bit_setting", sort=True):
        exact_rows = exact_rows_all.loc[exact_rows_all["layer"].isin(selected_layers)]
        full_rows = bit_group.loc[~bit_group["layer"].isin(selected_layers)]
        combined = pd.concat([exact_rows, full_rows], ignore_index=True)
        if combined.empty:
            continue
        row: dict[str, float | int | str] = {
            "dataset": combined["dataset"].iloc[0],
            "mode": "sensitive_layers_only_exact_v",
            "bit_setting": bit_setting,
            "bits": float(combined["bits"].dropna().iloc[0]) if combined["bits"].notna().any() else float("nan"),
            "selected_layer_count": len(selected_layers),
            "selected_layers": ",".join(str(value) for value in sorted(selected_layers)),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = float(combined[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _bit_setting_sort_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("inf")


def _paired_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["dataset", "trial", "layer"]
    for optional in ("capture_id", "prompt_hash", "prompt_label", "lane_name"):
        if optional in frame.columns:
            columns.append(optional)
    return columns


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = (total - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def compute_qwen_3060_multigroup_statistics(
    trial_frame: pd.DataFrame,
    *,
    modes: tuple[str, ...] = QWEN_3060_MATRIX_MODES,
    metrics: tuple[str, ...] = QWEN_3060_STAT_METRICS,
    baseline_modes: tuple[str, ...] = QWEN_3060_PAIRWISE_BASELINES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Friedman and pairwise Wilcoxon-Holm statistics for the 3060 matrix."""

    if trial_frame.empty:
        raise ValueError("compute_qwen_3060_multigroup_statistics received an empty trial_frame")
    required = {"mode", "bit_setting", "bits", *metrics}
    missing = sorted(required.difference(trial_frame.columns))
    if missing:
        raise ValueError(f"trial_frame is missing required columns: {missing}")

    pair_columns = _paired_columns(trial_frame)
    bit_settings = sorted({str(value) for value in trial_frame["bit_setting"].unique()}, key=_bit_setting_sort_key)

    friedman_rows: list[dict[str, float | int | str]] = []
    pairwise_rows: list[dict[str, float | int | str | bool]] = []

    for bit_setting in bit_settings:
        subset = trial_frame.loc[trial_frame["bit_setting"].astype(str) == bit_setting]
        for metric in metrics:
            mode_to_values: dict[str, list[float]] = {mode: [] for mode in modes}
            for _, group in subset.groupby(pair_columns, dropna=False, sort=True):
                block_values: list[float] = []
                skip_block = False
                for mode in modes:
                    mode_rows = group.loc[group["mode"] == mode]
                    if len(mode_rows) != 1:
                        skip_block = True
                        break
                    block_values.append(float(mode_rows.iloc[0][metric]))
                if skip_block:
                    continue
                for mode, value in zip(modes, block_values, strict=True):
                    mode_to_values[mode].append(value)

            n_blocks = len(mode_to_values[modes[0]])
            if n_blocks >= 2:
                result = stats.friedmanchisquare(*(mode_to_values[mode] for mode in modes))
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            else:
                statistic = float("nan")
                p_value = 1.0
            friedman_rows.append(
                {
                    "metric": metric,
                    "bit_setting": bit_setting,
                    "test": "friedman",
                    "n_blocks": n_blocks,
                    "n_modes": len(modes),
                    "statistic": statistic,
                    "p_value": p_value,
                    "modes": ",".join(modes),
                }
            )

            for baseline_mode in baseline_modes:
                raw_rows: list[dict[str, float | int | str | bool]] = []
                for candidate_mode in modes:
                    if candidate_mode == baseline_mode:
                        continue
                    paired = subset.loc[subset["mode"].isin((baseline_mode, candidate_mode))].copy()
                    pivot = paired.pivot_table(
                        index=pair_columns,
                        columns="mode",
                        values=metric,
                        aggfunc="first",
                    ).dropna()
                    if len(pivot) < 3:
                        continue
                    baseline_values = pivot[baseline_mode].astype(float)
                    candidate_values = pivot[candidate_mode].astype(float)
                    result = stats.wilcoxon(
                        baseline_values,
                        candidate_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                        correction=False,
                        method="auto",
                    )
                    raw_rows.append(
                        {
                            "metric": metric,
                            "bit_setting": bit_setting,
                            "baseline_mode": baseline_mode,
                            "candidate_mode": candidate_mode,
                            "test": "wilcoxon_signed_rank",
                            "n_pairs": int(len(pivot)),
                            "statistic": float(result.statistic),
                            "p_value": float(result.pvalue),
                            "baseline_mean": float(baseline_values.mean()),
                            "candidate_mean": float(candidate_values.mean()),
                            "delta_candidate_minus_baseline": float(candidate_values.mean() - baseline_values.mean()),
                        }
                    )
                if raw_rows:
                    adjusted = _holm_bonferroni([float(row["p_value"]) for row in raw_rows])
                    for row, adj_p in zip(raw_rows, adjusted, strict=True):
                        row["p_value_holm"] = adj_p
                        row["significant_0_05"] = adj_p < 0.05
                    pairwise_rows.extend(raw_rows)

    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


def evaluate_qwen_3060_matrix_rows(
    *,
    bundle: CapturedLayerBundle,
    trial: int,
    bit_grid: list[float],
    eval_device: str | torch.device = "cpu",
    triality_artifacts: dict | None = None,
    multiscreen_allocation: ChannelBitAllocation | None = None,
    rotation_dir: Path | None = None,
) -> list[dict[str, float | int | str]]:
    """Compose the 12GB-only Qwen comparison matrix for one captured layer bundle."""

    from turboquant.research_extension.captured_kv_modes import eval_captured_key_mode_row

    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device)
    keys = bundle.keys.to(target_device)
    values = bundle.values.to(target_device)
    dataset = f"qwen3060:{bundle.metadata.prompt_label or 'capture'}"

    for bit_value in bit_grid:
        bit_setting = f"{bit_value:g}"
        grid_rows = evaluate_layer_grid(
            dataset=dataset,
            keys=keys,
            values=values,
            trial=trial,
            layer_idx=bundle.layer_idx,
            bit_grid=[bit_value],
            eval_device=target_device,
        )
        for row in grid_rows:
            if str(row["mode"]) not in {"exact", "key_only_random", "full_kv"}:
                continue
            row["bit_setting"] = bit_setting
            row["bits"] = float(bit_value)
            row["mode_reference_bits"] = float(bit_value) if row["mode"] != "exact" else float("nan")
            rows.append(row)

        rows.append(
            evaluate_asymmetric_q8_value_attention_row(
                dataset=dataset,
                trial=trial,
                layer_idx=bundle.layer_idx,
                mode="asym_q8_turbo4",
                keys=keys,
                values=values,
                bit_setting=bit_setting,
                bits=bit_value,
                eval_device=target_device,
            )
        )
        rows.append(
            evaluate_asymmetric_q8_value_attention_row(
                dataset=dataset,
                trial=trial,
                layer_idx=bundle.layer_idx,
                mode="asym_q8_turbo3",
                keys=keys,
                values=values,
                bit_setting=bit_setting,
                bits=bit_value,
                eval_device=target_device,
            )
        )

        if multiscreen_allocation is not None:
            rows.append(
                eval_captured_key_mode_row(
                    mode="multiscreen_relevance",
                    bundle=CapturedLayerBundle(
                        metadata=bundle.metadata,
                        capture_dir=bundle.capture_dir,
                        layer_idx=bundle.layer_idx,
                        keys=keys,
                        values=values,
                    ),
                    trial=trial,
                    bit_value=float(bit_value),
                    eval_device=target_device,
                    rotation_dir=None,
                    triality_artifacts=None,
                    ms_alloc=multiscreen_allocation,
                )
            )

        if triality_artifacts is not None and rotation_dir is not None:
            rows.append(
                eval_captured_key_mode_row(
                    mode="key_only_block_so8_triality_vector",
                    bundle=CapturedLayerBundle(
                        metadata=bundle.metadata,
                        capture_dir=bundle.capture_dir,
                        layer_idx=bundle.layer_idx,
                        keys=keys,
                        values=values,
                    ),
                    trial=trial,
                    bit_value=float(bit_value),
                    eval_device=target_device,
                    rotation_dir=rotation_dir,
                    triality_artifacts=triality_artifacts,
                    ms_alloc=None,
                )
            )
    return rows


def evaluate_value_protection_grid(
    *,
    dataset: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    trial: int,
    layer_idx: int,
    eval_device: str | torch.device | None = None,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
    betas: tuple[float, ...] = (0.00, 0.10, 0.20),
    high_bits_grid: tuple[int, ...] = (4, 8),
    low_bits_grid: tuple[int, ...] = (2, 3),
    ranks: tuple[int, ...] = (0, 2, 4, 8),
    score_source: str = "teacher-gradient-proxy",
    channel_group_size: int = 8,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device) if eval_device is not None else keys.device
    if keys.device != target_device:
        keys = keys.to(target_device)
    if values.device != target_device:
        values = values.to(target_device)
    queries = select_queries(keys, seed=60_000 + (trial * 257) + layer_idx)
    rotation_seed = _seed_value(trial, layer_idx, 271)
    qjl_seed = _seed_value(trial, layer_idx, 313)
    key_codec = _base_kv_codec(
        keys=keys,
        bit_value=4.0,
        rotation_policy="block_so8_learned",
        rotation_seed=rotation_seed,
        qjl_seed=qjl_seed,
        value_codec=ValueCodecConfig(base_bits=2),
    )
    key_codec.calibrate(keys=keys, values=values, queries=queries)
    exact_bits = raw_storage_bits(keys) + raw_storage_bits(values)
    encoded_keys = key_codec.encode_keys(keys)
    key_only_ratio = float(key_codec.key_storage_bits(encoded_keys) + raw_storage_bits(values)) / float(exact_bits)

    for low_bits in low_bits_grid:
        low_quantizer = TurboQuantMSE(
            TurboQuantMSEConfig(
                dim=values.shape[-1],
                bits=low_bits,
                device=str(values.device),
                dtype=dtype_name(values.dtype),
                rotation_policy="block_so8_static",
                rotation_seed=rotation_seed,
            )
        )
        low_encoded = low_quantizer.quantize(values)
        low_ratio = float(key_codec.key_storage_bits(encoded_keys) + low_encoded.total_bits()) / float(exact_bits)
        for high_bits in high_bits_grid:
            high_quantizer = TurboQuantMSE(
                TurboQuantMSEConfig(
                    dim=values.shape[-1],
                    bits=high_bits,
                    device=str(values.device),
                    dtype=dtype_name(values.dtype),
                    rotation_policy="block_so8_static",
                    rotation_seed=rotation_seed,
                )
            )
            high_encoded = high_quantizer.quantize(values)
            high_ratio = float(key_codec.key_storage_bits(encoded_keys) + high_encoded.total_bits()) / float(exact_bits)
            for alpha in alphas:
                for beta in betas:
                    if alpha + beta > 1.0:
                        continue
                    for rank in ranks:
                        row = _evaluate_value_decoder_mode(
                            dataset=dataset,
                            layer_idx=layer_idx,
                            trial=trial,
                            bit_setting=f"{low_bits:g}",
                            bits=float(low_bits),
                            mode="protected_v_grid" if rank == 0 else "protected_v_lowrank_grid",
                            keys=keys,
                            values=values,
                            queries=queries,
                            key_codec=key_codec,
                            value_strategy="protected",
                            value_rotation_policy="block_so8_learned",
                            rotation_seed=rotation_seed,
                            qjl_seed=qjl_seed,
                            low_rank_rank=rank,
                            protected_fraction=alpha,
                            secondary_fraction=beta,
                            high_bits=high_bits,
                            score_source=score_source,
                            channel_group_size=channel_group_size,
                        )
                        meta_ratio = float((2 * values.shape[1] * values.shape[-1]) / max(exact_bits, 1))
                        row["protected_fraction"] = alpha
                        row["secondary_fraction"] = beta
                        row["high_bits"] = high_bits
                        row["low_bits"] = low_bits
                        row["low_rank_rank"] = rank
                        row["memory_ratio_approx"] = low_ratio + (alpha * (key_only_ratio - low_ratio)) + (beta * (high_ratio - low_ratio)) + meta_ratio
                        rows.append(row)
    return rows


def synthetic_kv(seed: int, batch: int, heads: int, seq_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keys = torch.randn((batch, heads, seq_len, dim), generator=generator, dtype=torch.float32)
    values = torch.randn((batch, heads, seq_len, dim), generator=generator, dtype=torch.float32)
    return keys, values


def load_captured_layers(kv_dir: Path) -> tuple[CaptureMetadata | None, list[tuple[int, torch.Tensor, torch.Tensor]]]:
    manifest_path = kv_dir / "capture_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing capture manifest: {manifest_path}")
    metadata = load_capture_metadata(manifest_path)
    layers: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for record in metadata.layers:
        key_path = kv_dir / record.key_file
        value_path = kv_dir / record.value_file
        if not key_path.exists():
            raise FileNotFoundError(f"Missing captured key tensor: {key_path}")
        if not value_path.exists():
            raise FileNotFoundError(f"Missing captured value tensor: {value_path}")
        layers.append(
            (
                int(record.layer_index),
                torch.load(key_path, map_location="cpu"),
                torch.load(value_path, map_location="cpu"),
            )
        )
    return metadata, layers


def load_captured_runs(kv_root: Path) -> list[CapturedLayerBundle]:
    if (kv_root / "capture_manifest.json").exists():
        capture_dirs = [kv_root]
    else:
        capture_dirs = sorted(path for path in kv_root.iterdir() if path.is_dir() and (path / "capture_manifest.json").exists())
    if not capture_dirs:
        raise FileNotFoundError(
            f"No prompt-scoped capture directories found under {kv_root}. "
            "Run scripts/capture_qwen_kv.py first."
        )
    bundles: list[CapturedLayerBundle] = []
    for capture_dir in capture_dirs:
        metadata, layers = load_captured_layers(capture_dir)
        if metadata is None:
            raise FileNotFoundError(f"Missing capture manifest for {capture_dir}")
        for layer_idx, keys, values in layers:
            bundles.append(
                CapturedLayerBundle(
                    metadata=metadata,
                    capture_dir=capture_dir,
                    layer_idx=layer_idx,
                    keys=keys,
                    values=values,
                )
            )
    return bundles
