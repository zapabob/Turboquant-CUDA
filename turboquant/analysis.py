"""Offline KV replay and summary helpers for paper-faithful TurboQuant evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from turboquant.attention_metrics import summarize_attention_scores
from turboquant.capture import CaptureMetadata, load_capture_metadata
from turboquant.kv_codec import KVCodec, KVCodecConfig
from turboquant.reporting import summarize_metric_trials
from turboquant.types import ValueCodecConfig


METRIC_COLUMNS = [
    "logit_cosine_similarity",
    "logit_mae",
    "logit_mse",
    "logit_spearman",
    "logit_top1_match",
    "logit_top5_match",
    "logit_top5_overlap",
    "hidden_cosine_similarity",
    "hidden_mae",
    "hidden_mse",
    "memory_bits",
    "memory_ratio_vs_exact",
    "prefill_seconds",
    "decode_seconds",
    "peak_vram_mb",
]


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


def melt_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    id_vars = [column for column in frame.columns if column not in METRIC_COLUMNS]
    return frame.melt(id_vars=id_vars, value_vars=METRIC_COLUMNS, var_name="metric", value_name="value")


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
        return {
            "dataset": dataset,
            "trial": trial,
            "layer": layer_idx,
            "mode": mode,
            "bit_setting": bit_setting,
            "bits": float("nan"),
            "logit_cosine_similarity": 1.0,
            "logit_mae": 0.0,
            "logit_mse": 0.0,
            "logit_spearman": 1.0,
            "logit_top1_match": 1.0,
            "logit_top5_match": 1.0,
            "logit_top5_overlap": 1.0,
            "hidden_cosine_similarity": 1.0,
            "hidden_mae": 0.0,
            "hidden_mse": 0.0,
            "memory_bits": float(exact_memory_bits),
            "memory_ratio_vs_exact": 1.0,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "peak_vram_mb": _peak_vram_mb(device),
        }

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

    return {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": mode,
        "bit_setting": bit_setting,
        "bits": float(bits) if bits is not None else float("nan"),
        "logit_cosine_similarity": logit_metrics["cosine_similarity"],
        "logit_mae": logit_metrics["mae"],
        "logit_mse": logit_metrics["mse"],
        "logit_spearman": logit_metrics["spearman"],
        "logit_top1_match": logit_metrics["top1_match"],
        "logit_top5_match": logit_metrics["top5_match"],
        "logit_top5_overlap": logit_metrics["top5_overlap"],
        "hidden_cosine_similarity": hidden_metrics["cosine_similarity"],
        "hidden_mae": hidden_metrics["mae"],
        "hidden_mse": hidden_metrics["mse"],
        "memory_bits": float(memory_bits),
        "memory_ratio_vs_exact": float(memory_bits) / float(exact_memory_bits),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_vram_mb": _peak_vram_mb(device),
    }


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
        key_bits = int(math.floor(bit_value))
        bit_setting = f"{bit_value:g}"
        base_kwargs = dict(
            head_dim=keys.shape[-1],
            key_bits=key_bits,
            value_bits=key_bits,
            mixed_key_bits=bit_value if not float(bit_value).is_integer() else None,
            mixed_value_bits=bit_value if not float(bit_value).is_integer() else None,
            device=str(keys.device),
            dtype=dtype_name(keys.dtype),
        )
        specs = [
            ("key_only_random", "random_haar", "exact", False, 0),
            ("key_only_block_so8_static", "block_so8_static", "exact", False, 0),
            ("key_only_block_so8_learned", "block_so8_learned", "exact", True, 0),
            ("protected_v", "block_so8_learned", "protected", True, 0),
            ("protected_v_lowrank", "block_so8_learned", "protected", True, 4),
            ("full_kv", "random_haar", "full_kv", False, 0),
        ]
        for mode, rotation_policy, value_mode, calibrate_codec, low_rank_rank in specs:
            value_codec = ValueCodecConfig(
                base_bits=key_bits,
                low_rank_rank=low_rank_rank,
            )
            codec = KVCodec(
                KVCodecConfig(
                    **base_kwargs,
                    rotation_policy=rotation_policy,
                    value_codec=value_codec,
                )
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
            )
            rows.append(mode_row)
            if progress_callback is not None:
                progress_callback(mode_row)
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
