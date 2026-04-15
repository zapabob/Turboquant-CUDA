"""Research-only K-side triality proxy training and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.io_utils import ensure_dir
from turboquant.analysis import (
    _peak_vram_mb,
    _reset_peak_if_cuda,
    _sync_if_cuda,
    evaluate_layer_grid,
    load_captured_runs,
    raw_storage_bits,
    scaled_attention_output,
    select_queries,
    summarize_trial_metrics,
)
from turboquant.attention_metrics import summarize_attention_scores
from turboquant.rotation import so8_block_diagonal_rotation_metrics
from turboquant.research_extension.multiscreen_kv import compute_k_relevance, expand_relevance_bitwidths_to_key_shape
from turboquant.research_extension.triality_proxy import TRIALITY_PROXY_VIEWS, TrialityProxyProd
from turboquant.schema import (
    DEFAULT_BITWIDTH_PAYLOAD_DTYPE,
    DEFAULT_SIGN_PACK_FORMAT,
    build_turboquant_artifact_metadata,
    validate_turboquant_artifact_metadata,
)
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import TurboQuantProdConfig


TRIALITY_MODE_BY_VIEW = {
    "vector": "key_only_block_so8_triality_vector",
    "spinor_plus_proxy": "key_only_block_so8_triality_plus",
    "spinor_minus_proxy": "key_only_block_so8_triality_minus",
}
# These runtime mode strings are legacy identifiers. The explicit ABI / manifest
# label for the current implementation is `tq_triality_mode="triality_proxy"`.

# Subset of evaluate_layer_grid modes replayed alongside triality-proxy rows (paper baselines + static SO8).
TRIALITY_BASELINE_GRID_MODES: frozenset[str] = frozenset(
    {
        "exact",
        "key_only_random",
        "key_only_block_so8_static",
        "key_only_block_so8_learned",
        "full_kv",
    }
)

# Paired multi-group comparison order (random Haar, static SO8, learned SO8, triality views, full-KV).
ROTATION_COMPARE_MODES: tuple[str, ...] = (
    "key_only_random",
    "key_only_block_so8_static",
    "key_only_block_so8_learned",
    "key_only_block_so8_triality_vector",
    "key_only_block_so8_triality_plus",
    "key_only_block_so8_triality_minus",
    "full_kv",
)
TRIALITY_SELECTOR_MODE = "key_only_block_so8_triality_best_per_layer"

# --- Production canonical K-side TurboQuant (実用正系) ---------------------------------
# Offline replay / shipping recommendation: per-layer fitted block-SO(8) proxy rotations
# with the **vector** triality view, Stage 1+2 TurboQuantProd path (`k_triality._evaluate_triality_proxy_mode`).
# Paper modes (`key_only_random`, static SO8 without triality, etc.) remain ablation / reproducibility baselines.
PRODUCTION_K_TURBOQUANT_VIEW: str = "vector"
PRODUCTION_K_TURBOQUANT_MODE: str = "key_only_block_so8_triality_vector"
DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR: str = "artifacts/research_extension/triality_full_train/rotations"

# Learned Triality (vector) rotations + Multiscreen relevance mixed-bit Stage-1 keys (combined eval row).
MULTISCREEN_TRIALITY_VECTOR_MODE: str = "multiscreen_triality_vector"


def bit_setting_sort_key(label: str) -> tuple[int, float]:
    """Sort key for ``bit_setting`` labels (numeric first, then lexicographic fallback)."""

    try:
        return (0, float(label))
    except ValueError:
        return (1, 0.0)


def _canonical_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    return str(device)


@dataclass(slots=True)
class TrialityRotationArtifact:
    layer_idx: int
    bits: float
    view: str
    rotation: torch.Tensor
    rotation_seed: int
    qjl_seed: int
    metadata: dict[str, object]


def triality_mode_name(view: str) -> str:
    try:
        return TRIALITY_MODE_BY_VIEW[view]
    except KeyError as exc:
        raise ValueError(f"Unsupported triality view: {view!r}") from exc


def _seed_value(layer_idx: int, bit_value: float, view_index: int, salt: int) -> int:
    return 70_000 + (layer_idx * 2_003) + int(bit_value * 100) + (view_index * 257) + salt


def _stage1_allocation(bits: float, qjl_bits: int = 1, width: int = 128) -> ChannelBitAllocation | None:
    if float(bits).is_integer():
        return None
    effective_bits = bits - qjl_bits
    if effective_bits <= 0:
        raise ValueError("effective stage-1 bits must remain positive")
    return ChannelBitAllocation.preset(effective_bits=effective_bits, width=width)


def _bundle_groups(kv_root: Path, max_layers: int = 0) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for bundle in load_captured_runs(kv_root):
        if max_layers > 0 and bundle.layer_idx >= max_layers:
            continue
        grouped.setdefault(bundle.layer_idx, []).append(bundle)
    return grouped


def _rotation_step_trace_callback(
    rotation_fit_trace: list[dict[str, float | int | str]],
    *,
    layer_idx: int,
    bit_value: float,
    view: str,
) -> Callable[[int, torch.Tensor], None]:
    bit_setting = f"{bit_value:g}"

    def callback(step: int, rotation: torch.Tensor) -> None:
        ortho, det_err = so8_block_diagonal_rotation_metrics(rotation)
        rotation_fit_trace.append(
            {
                "layer": layer_idx,
                "bits": float(bit_value),
                "bit_setting": bit_setting,
                "view": view,
                "step": step,
                "orthogonality_error": ortho,
                "rotation_determinant_error_max": det_err,
            }
        )

    return callback


def _stack_training_keys_and_queries(
    bundles: list,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys_list: list[torch.Tensor] = []
    queries_list: list[torch.Tensor] = []
    for bundle_index, bundle in enumerate(bundles):
        keys = bundle.keys.to(device)
        keys_list.append(keys)
        queries_list.append(select_queries(keys, seed=90_000 + (bundle.layer_idx * 131) + bundle_index))
    keys = torch.cat(keys_list, dim=-2)
    queries = torch.cat(queries_list, dim=-2)
    return keys, queries


def fit_triality_proxy_rotations(
    *,
    kv_root: Path,
    bit_grid: list[float],
    max_layers: int = 0,
    steps: int = 60,
    lr: float = 5e-2,
    device: str | torch.device = "cpu",
    rotation_fit_trace: list[dict[str, float | int | str]] | None = None,
) -> tuple[list[TrialityRotationArtifact], pd.DataFrame]:
    target_device = torch.device(device)
    device_name = _canonical_device_name(target_device)
    artifacts: list[TrialityRotationArtifact] = []
    rows: list[dict[str, float | int | str]] = []
    grouped = _bundle_groups(kv_root, max_layers=max_layers)
    for layer_idx, bundles in sorted(grouped.items()):
        keys, queries = _stack_training_keys_and_queries(bundles, device=target_device)
        for bit_value in bit_grid:
            total_bits = int(math.floor(bit_value))
            allocation = _stage1_allocation(bit_value, width=keys.shape[-1])
            for view_index, view in enumerate(TRIALITY_PROXY_VIEWS):
                rotation_seed = _seed_value(layer_idx, bit_value, view_index, 17)
                qjl_seed = _seed_value(layer_idx, bit_value, view_index, 71)
                metadata = build_turboquant_artifact_metadata(
                    total_bits=float(bit_value),
                    qjl_bits=1,
                    qjl_dim=keys.shape[-1],
                    rotation_policy="block_so8_learned",
                    rotation_seed=rotation_seed,
                    qjl_seed=qjl_seed,
                    triality_mode="triality_proxy",
                    triality_view=view,
                    width=keys.shape[-1],
                    allocation=allocation,
                    bitwidth_payload_dtype=DEFAULT_BITWIDTH_PAYLOAD_DTYPE,
                    norm_dtype=str(keys.dtype).split(".")[-1],
                    sign_pack_format=DEFAULT_SIGN_PACK_FORMAT,
                )
                quantizer = TurboQuantProd(
                    TurboQuantProdConfig(
                        dim=keys.shape[-1],
                        total_bits=total_bits,
                        rotation_seed=rotation_seed,
                        rotation_policy="block_so8_learned",
                        qjl_seed=qjl_seed,
                        device=device_name,
                        dtype=str(keys.dtype).split(".")[-1],
                    )
                )
                proxy = TrialityProxyProd(quantizer=quantizer, view=view)
                step_cb = (
                    _rotation_step_trace_callback(
                        rotation_fit_trace,
                        layer_idx=layer_idx,
                        bit_value=bit_value,
                        view=view,
                    )
                    if rotation_fit_trace is not None
                    else None
                )
                proxy.fit_rotation(keys, queries=queries, steps=steps, lr=lr, step_metrics_callback=step_cb)
                encoded = proxy.quantize(keys, allocation=allocation)
                estimated_logits = proxy.pairwise_estimate(queries, encoded)
                exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
                metrics = summarize_attention_scores(exact_logits, estimated_logits)
                rotation = proxy.quantizer.mse_quantizer.rotation.detach().cpu()
                artifacts.append(
                    TrialityRotationArtifact(
                        layer_idx=layer_idx,
                        bits=float(bit_value),
                        view=view,
                        rotation=rotation,
                        rotation_seed=rotation_seed,
                        qjl_seed=qjl_seed,
                        metadata=metadata,
                    )
                )
                ortho_err, det_err_max = so8_block_diagonal_rotation_metrics(rotation)
                row = {
                    "layer": layer_idx,
                    "bits": float(bit_value),
                    "bit_setting": f"{bit_value:g}",
                    "view": view,
                    "mode": triality_mode_name(view),
                    "rotation_seed": rotation_seed,
                    "qjl_seed": qjl_seed,
                    "prompt_count": len(bundles),
                    "token_count": int(keys.shape[-2]),
                    "orthogonality_error": ortho_err,
                    "rotation_determinant_error_max": det_err_max,
                    "train_logit_cosine_similarity": metrics["cosine_similarity"],
                    "train_logit_mse": metrics["mse"],
                }
                row.update(metadata)
                rows.append(row)
    return artifacts, pd.DataFrame(rows)


def save_triality_proxy_rotations(artifacts: list[TrialityRotationArtifact], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        file_name = f"layer_{artifact.layer_idx:02d}_bits_{artifact.bits:g}_{artifact.view}.pt"
        path = output_dir / file_name
        torch.save(
            {
                "layer": artifact.layer_idx,
                "bits": artifact.bits,
                "view": artifact.view,
                "rotation": artifact.rotation,
                "rotation_seed": artifact.rotation_seed,
                "qjl_seed": artifact.qjl_seed,
                "metadata": artifact.metadata,
            },
            path,
        )
        row = {
            "layer": artifact.layer_idx,
            "bits": artifact.bits,
            "bit_setting": f"{artifact.bits:g}",
            "view": artifact.view,
            "mode": triality_mode_name(artifact.view),
            "rotation_path": str(path).replace("\\", "/"),
            "rotation_seed": artifact.rotation_seed,
            "qjl_seed": artifact.qjl_seed,
        }
        row.update(artifact.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def load_triality_proxy_rotations(rotation_dir: Path) -> dict[tuple[int, float, str], TrialityRotationArtifact]:
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact] = {}
    for path in sorted(rotation_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata")
        if metadata is None:
            rotation = payload["rotation"]
            metadata = build_turboquant_artifact_metadata(
                total_bits=float(payload["bits"]),
                qjl_bits=1,
                qjl_dim=int(rotation.shape[-1]),
                rotation_policy="block_so8_learned",
                rotation_seed=int(payload["rotation_seed"]),
                qjl_seed=int(payload["qjl_seed"]),
                triality_mode="triality_proxy",
                triality_view=str(payload["view"]),
                width=int(rotation.shape[-1]),
                allocation=_stage1_allocation(float(payload["bits"]), width=int(rotation.shape[-1])),
                bitwidth_payload_dtype=DEFAULT_BITWIDTH_PAYLOAD_DTYPE,
                norm_dtype=str(rotation.dtype).split(".")[-1],
                sign_pack_format=DEFAULT_SIGN_PACK_FORMAT,
            )
            metadata["tq_metadata_inferred_from_legacy_payload"] = True
        validate_turboquant_artifact_metadata(metadata)
        artifact = TrialityRotationArtifact(
            layer_idx=int(payload["layer"]),
            bits=float(payload["bits"]),
            view=str(payload["view"]),
            rotation=payload["rotation"].to(dtype=torch.float32),
            rotation_seed=int(payload["rotation_seed"]),
            qjl_seed=int(payload["qjl_seed"]),
            metadata=metadata,
        )
        artifacts[(artifact.layer_idx, artifact.bits, artifact.view)] = artifact
    if not artifacts:
        raise FileNotFoundError(f"No triality rotation artifacts found under {rotation_dir}")
    return artifacts


def _validate_triality_artifacts(
    *,
    bundles: list,
    bit_grid: list[float],
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact],
    max_layers: int = 0,
) -> None:
    expected: set[tuple[int, float, str]] = set()
    included_layers = sorted({bundle.layer_idx for bundle in bundles if max_layers <= 0 or bundle.layer_idx < max_layers})
    for layer_idx in included_layers:
        for bit_value in bit_grid:
            for view in TRIALITY_PROXY_VIEWS:
                expected.add((layer_idx, float(bit_value), view))
    missing = sorted(expected.difference(artifacts.keys()))
    if missing:
        preview = ", ".join(f"(layer={layer}, bits={bits:g}, view={view})" for layer, bits, view in missing[:12])
        suffix = "" if len(missing) <= 12 else f" ... and {len(missing) - 12} more"
        raise KeyError(f"Missing triality rotation artifacts: {preview}{suffix}")


def _validate_triality_rotation_key_dims(
    *,
    bundles_filtered: list,
    artifacts: dict[tuple[int, float, str], TrialityRotationArtifact],
    rotation_dir: Path,
) -> None:
    """Ensure each loaded rotation is (D, D) with D equal to captured keys' last dimension."""

    if not bundles_filtered:
        return
    dims = [int(b.keys.shape[-1]) for b in bundles_filtered]
    unique_dims = sorted(set(dims))
    if len(unique_dims) != 1:
        raise ValueError(
            "Captured KV bundles use inconsistent key head dimensions (last axis): "
            f"{unique_dims}. All bundles must share the same D for triality eval."
        )
    expected = unique_dims[0]
    mismatches: list[str] = []
    for (layer, bits, view), art in sorted(artifacts.items(), key=lambda item: item[0]):
        shape_tuple = tuple(art.rotation.shape)
        if shape_tuple != (expected, expected):
            mismatches.append(
                f"(layer={layer}, bits={bits:g}, view={view}): rotation {shape_tuple}, need ({expected}, {expected})"
            )
    if mismatches:
        preview = "\n  ".join(mismatches[:12])
        suffix = f"\n  ... and {len(mismatches) - 12} more" if len(mismatches) > 12 else ""
        raise ValueError(
            "Triality rotation matrices under "
            f"{rotation_dir} do not match captured key head dimension D={expected}. "
            "Train with scripts/research_train_k_triality.py using the same --kv-dir, or choose a rotation_dir "
            f"trained on KV with the same head size.\n  {preview}{suffix}"
        )


def _evaluate_triality_proxy_mode(
    *,
    dataset: str,
    trial: int,
    layer_idx: int,
    bit_value: float,
    view: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    artifact: TrialityRotationArtifact,
) -> dict[str, float | int | str]:
    device = keys.device
    allocation = _stage1_allocation(bit_value, width=keys.shape[-1])
    quantizer = TurboQuantProd(
        TurboQuantProdConfig(
            dim=keys.shape[-1],
            total_bits=int(math.floor(bit_value)),
            rotation_seed=artifact.rotation_seed,
            rotation_policy="block_so8_static",
            qjl_seed=artifact.qjl_seed,
            device=_canonical_device_name(torch.device(device)),
            dtype=str(keys.dtype).split(".")[-1],
        )
    )
    proxy = TrialityProxyProd(quantizer=quantizer, view=view)
    proxy.set_rotation(artifact.rotation.to(device=device, dtype=keys.dtype))

    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=keys.shape[-1])
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = proxy.quantize(keys, allocation=allocation)
    estimated_logits = proxy.pairwise_estimate(queries, encoded_keys)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = proxy.pairwise_estimate(last_query, encoded_keys)
    _ = scaled_attention_output(decode_logits, values, head_dim=keys.shape[-1])
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started

    hidden = scaled_attention_output(estimated_logits, values, head_dim=keys.shape[-1])
    logit_metrics = summarize_attention_scores(exact_logits, estimated_logits)
    hidden_metrics = summarize_attention_scores(exact_hidden, hidden)
    memory_bits = float(encoded_keys.total_bits() + raw_storage_bits(values))
    return {
        "dataset": dataset,
        "trial": trial,
        "layer": layer_idx,
        "mode": triality_mode_name(view),
        "bit_setting": f"{bit_value:g}",
        "bits": float(bit_value),
        "view": view,
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
        "key_mode": triality_mode_name(view),
        "value_mode": "exact",
        "value_rotation_policy": "exact",
        "rotation_seed": artifact.rotation_seed,
        "qjl_seed": artifact.qjl_seed,
    }


def evaluate_multiscreen_triality_vector_row(
    *,
    dataset: str,
    trial: int,
    layer_idx: int,
    bit_value: float,
    keys: torch.Tensor,  # [batch, heads, seq, head_dim]
    values: torch.Tensor,
    queries: torch.Tensor,
    allocation: ChannelBitAllocation,
    artifact: TrialityRotationArtifact,
) -> dict[str, float | int | str]:
    """Key-only row: Multiscreen relevance bit map + fitted Triality vector proxy (learned SO(8) rotation)."""

    device = keys.device
    head_dim = keys.shape[-1]
    n_rel = int(keys.shape[0] * keys.shape[1] * keys.shape[2])
    oc = min(allocation.outlier_count, n_rel)
    alloc_use = (
        allocation
        if oc == allocation.outlier_count
        else ChannelBitAllocation.from_multiscreen_relevance(
            regular_bits=allocation.regular_bits,
            outlier_bits=allocation.outlier_bits,
            outlier_count=oc,
        )
    )

    quantizer = TurboQuantProd(
        TurboQuantProdConfig(
            dim=head_dim,
            total_bits=int(math.floor(bit_value)),
            rotation_seed=artifact.rotation_seed,
            rotation_policy="block_so8_static",
            qjl_seed=artifact.qjl_seed,
            device=_canonical_device_name(torch.device(device)),
            dtype=str(keys.dtype).split(".")[-1],
        )
    )
    proxy = TrialityProxyProd(quantizer=quantizer, view="vector")
    proxy.set_rotation(artifact.rotation.to(device=device, dtype=keys.dtype))

    relevance = compute_k_relevance(queries, keys)
    bitwidths = expand_relevance_bitwidths_to_key_shape(relevance, alloc_use, head_dim)

    exact_logits = torch.einsum("...qd,...sd->...qs", queries, keys)
    exact_hidden = scaled_attention_output(exact_logits, values, head_dim=head_dim)
    exact_memory_bits = raw_storage_bits(keys) + raw_storage_bits(values)

    _sync_if_cuda(device)
    _reset_peak_if_cuda(device)
    started = time.perf_counter()
    encoded_keys = proxy.quantize_with_bitwidths(keys, bitwidths)
    estimated_logits = proxy.pairwise_estimate(queries, encoded_keys)
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - started

    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    last_query = queries[..., -1:, :]
    decode_logits = proxy.pairwise_estimate(last_query, encoded_keys)
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
        "mode": MULTISCREEN_TRIALITY_VECTOR_MODE,
        "bit_setting": bit_setting,
        "bits": float(bit_value),
        "view": "vector",
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
        "key_mode": MULTISCREEN_TRIALITY_VECTOR_MODE,
        "value_mode": "exact",
        "value_rotation_policy": "exact",
        "rotation_seed": artifact.rotation_seed,
        "qjl_seed": artifact.qjl_seed,
        "multiscreen_regular_bits": alloc_use.regular_bits,
        "multiscreen_outlier_bits": alloc_use.outlier_bits,
        "multiscreen_outlier_count": alloc_use.outlier_count,
    }


def _stable_sort_bundles(bundles: list) -> list:
    return sorted(bundles, key=lambda b: (str(b.capture_dir), int(b.layer_idx)))


def _bundle_keys_filtered(bundles_filtered: list) -> list[str]:
    # Resolve so bundle_keys match resume state regardless of relative vs absolute --kv-dir.
    return [f"{b.capture_dir.resolve().as_posix()}:{b.layer_idx}" for b in bundles_filtered]


def triality_eval_config_fingerprint(
    *,
    kv_root: Path,
    rotation_dir: Path,
    trial_count: int,
    bit_grid: list[float],
    max_layers: int,
    eval_device: str | torch.device | None,
    bundles_filtered: list,
) -> str:
    device_str: str | None
    if eval_device is None:
        device_str = None
    else:
        device_str = str(eval_device)
    payload = {
        "baseline_grid_modes": sorted(TRIALITY_BASELINE_GRID_MODES),
        "bit_grid": bit_grid,
        "bundle_keys": _bundle_keys_filtered(bundles_filtered),
        "eval_device": device_str,
        "kv_root": str(kv_root.resolve()),
        "max_layers": max_layers,
        "rotation_dir": str(rotation_dir.resolve()),
        "trial_count": trial_count,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _next_triality_position(
    trial: int, bundle_index: int, *, trial_count: int, n_bundles: int
) -> tuple[int, int]:
    if bundle_index + 1 < n_bundles:
        return trial, bundle_index + 1
    if trial + 1 < trial_count:
        return trial + 1, 0
    return trial_count, 0


def _write_triality_partial_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".partial.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _save_triality_resume_state(
    path: Path,
    *,
    fingerprint: str,
    bundle_keys: list[str],
    next_trial: int,
    next_bundle_index: int,
    trial_count: int,
    n_bundles: int,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "bundle_keys": bundle_keys,
        "config_fingerprint": fingerprint,
        "n_bundles": n_bundles,
        "next_bundle_index": next_bundle_index,
        "next_trial": next_trial,
        "trial_count": trial_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_triality_proxy_captured(
    *,
    kv_root: Path,
    trial_count: int,
    bit_grid: list[float],
    rotation_dir: Path,
    max_layers: int = 0,
    eval_device: str | torch.device | None = None,
    metrics_dir: Path | None = None,
    resume: bool = False,
    resume_state_path: Path | None = None,
    partial_csv_path: Path | None = None,
    force_fresh: bool = False,
    on_bundle_done: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_triality_proxy_rotations(rotation_dir)
    rows: list[dict[str, float | int | str]] = []
    target_device = torch.device(eval_device) if eval_device is not None else None
    bundles = _stable_sort_bundles(load_captured_runs(kv_root))
    bundles_filtered = [b for b in bundles if max_layers <= 0 or b.layer_idx < max_layers]
    n_bundles = len(bundles_filtered)
    bundle_keys = _bundle_keys_filtered(bundles_filtered)
    fingerprint = triality_eval_config_fingerprint(
        kv_root=kv_root,
        rotation_dir=rotation_dir,
        trial_count=trial_count,
        bit_grid=bit_grid,
        max_layers=max_layers,
        eval_device=eval_device,
        bundles_filtered=bundles_filtered,
    )
    _validate_triality_rotation_key_dims(
        bundles_filtered=bundles_filtered,
        artifacts=artifacts,
        rotation_dir=rotation_dir,
    )
    _validate_triality_artifacts(
        bundles=bundles,
        bit_grid=bit_grid,
        artifacts=artifacts,
        max_layers=max_layers,
    )

    partial_path = partial_csv_path
    state_path = resume_state_path
    if metrics_dir is not None:
        ensure_dir(metrics_dir)
        if partial_path is None:
            partial_path = metrics_dir / "triality_trials_partial.csv"
        if state_path is None:
            state_path = metrics_dir / "eval_resume_state.json"

    start_trial = 0
    start_bundle = 0
    resume_effective = bool(resume) and not force_fresh
    if resume_effective and partial_path is not None and state_path is not None:
        if not partial_path.exists() or not state_path.exists():
            warnings.warn(
                "Resume requested but partial CSV or eval_resume_state.json is missing; starting fresh.",
                UserWarning,
                stacklevel=2,
            )
        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            fp_ok = state.get("config_fingerprint") == fingerprint
            keys_ok = state.get("bundle_keys") == bundle_keys
            if not fp_ok:
                warnings.warn(
                    "eval_resume_state.json fingerprint mismatch (KV/rotation/trials/bits changed); starting fresh.",
                    UserWarning,
                    stacklevel=2,
                )
            elif not keys_ok:
                warnings.warn(
                    "eval_resume_state.json bundle_keys mismatch; starting fresh.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                rows = pd.read_csv(partial_path).to_dict("records")
                start_trial = int(state["next_trial"])
                start_bundle = int(state["next_bundle_index"])
    elif resume and not force_fresh and metrics_dir is None:
        warnings.warn("Resume requested but metrics_dir is unset; starting fresh.", UserWarning, stacklevel=2)

    if start_trial >= trial_count:
        trial_frame = pd.DataFrame(rows)
        if trial_frame.empty:
            raise ValueError(
                "Triality resume produced no rows "
                f"(trial_count={trial_count}, n_bundles={n_bundles}, bits={bit_grid})"
            )
        return trial_frame, summarize_trial_metrics(trial_frame)

    for trial in range(trial_count):
        for b_idx, bundle in enumerate(bundles_filtered):
            if trial < start_trial or (trial == start_trial and b_idx < start_bundle):
                continue
            keys = bundle.keys.to(target_device) if target_device is not None else bundle.keys
            values = bundle.values.to(target_device) if target_device is not None else bundle.values
            rows.extend(
                {
                    **row,
                    "capture_id": bundle.metadata.capture_id or bundle.capture_dir.name,
                    "prompt_label": bundle.metadata.prompt_label or "unknown",
                    "prompt_hash": bundle.metadata.prompt_hash,
                }
                for row in evaluate_layer_grid(
                    dataset=f"research_captured:{bundle.metadata.prompt_label}",
                    keys=keys,
                    values=values,
                    trial=trial,
                    layer_idx=bundle.layer_idx,
                    bit_grid=bit_grid,
                    eval_device=target_device,
                )
                if row["mode"] in TRIALITY_BASELINE_GRID_MODES
            )
            queries = select_queries(keys, seed=10_000 + (trial * 257) + bundle.layer_idx)
            for bit_value in bit_grid:
                for view in TRIALITY_PROXY_VIEWS:
                    artifact = artifacts[(bundle.layer_idx, float(bit_value), view)]
                    row = _evaluate_triality_proxy_mode(
                        dataset=f"research_captured:{bundle.metadata.prompt_label}",
                        trial=trial,
                        layer_idx=bundle.layer_idx,
                        bit_value=bit_value,
                        view=view,
                        keys=keys,
                        values=values,
                        queries=queries,
                        artifact=artifact,
                    )
                    row["capture_id"] = bundle.metadata.capture_id or bundle.capture_dir.name
                    row["prompt_label"] = bundle.metadata.prompt_label or "unknown"
                    row["prompt_hash"] = bundle.metadata.prompt_hash
                    rows.append(row)

            if partial_path is not None and state_path is not None:
                next_trial, next_bundle = _next_triality_position(
                    trial, b_idx, trial_count=trial_count, n_bundles=n_bundles
                )
                _write_triality_partial_csv(partial_path, rows)
                _save_triality_resume_state(
                    state_path,
                    fingerprint=fingerprint,
                    bundle_keys=bundle_keys,
                    next_trial=next_trial,
                    next_bundle_index=next_bundle,
                    trial_count=trial_count,
                    n_bundles=n_bundles,
                )
            if on_bundle_done is not None:
                on_bundle_done()

    trial_frame = pd.DataFrame(rows)
    if trial_frame.empty:
        raise ValueError(
            "Triality replay produced no rows "
            f"(bundles={n_bundles}, bits={bit_grid}, max_layers={max_layers}, artifacts={len(artifacts)})"
        )
    return trial_frame, summarize_trial_metrics(trial_frame)


def triality_pairing_columns(trial_frame: pd.DataFrame) -> list[str]:
    pair_columns = ["dataset", "trial", "layer", "bit_setting", "bits"]
    for optional in ("capture_id", "prompt_label", "prompt_hash"):
        if optional in trial_frame.columns:
            pair_columns.append(optional)
    return pair_columns


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = (total - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def compute_friedman_rotation_mode_statistics(
    trial_frame: pd.DataFrame,
    *,
    modes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Friedman test on paired blocks (same trial/layer/capture/bit) across rotation modes."""
    if trial_frame.empty:
        return pd.DataFrame()
    mode_order = modes or ROTATION_COMPARE_MODES
    pair_columns = triality_pairing_columns(trial_frame)
    metrics = ("hidden_cosine_similarity", "next_logit_kl")
    rows_out: list[dict[str, float | int | str]] = []
    bit_settings = sorted(
        {str(x) for x in trial_frame["bit_setting"].unique()} - {"exact"},
        key=bit_setting_sort_key,
    )
    for metric in metrics:
        for bit_setting in bit_settings:
            sub = trial_frame.loc[
                (trial_frame["mode"].isin(mode_order))
                & (trial_frame["bit_setting"].astype(str) == str(bit_setting))
            ]
            if sub.empty:
                continue
            blocks: list[list[float]] = []
            grouped = sub.groupby(pair_columns, dropna=False, sort=True)
            for _, group in grouped:
                if len(group) != len(mode_order) or group["mode"].astype(str).nunique() != len(mode_order):
                    continue
                mode_to_val = {str(row["mode"]): float(row[metric]) for _, row in group.iterrows()}
                if all(m in mode_to_val for m in mode_order):
                    blocks.append([float(mode_to_val[m]) for m in mode_order])
            if len(blocks) < 2:
                continue
            k = len(mode_order)
            n_b = len(blocks)
            columns = [[blocks[r][c] for r in range(n_b)] for c in range(k)]
            try:
                result = stats.friedmanchisquare(*columns)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                statistic = float("nan")
                p_value = 1.0
            rows_out.append(
                {
                    "metric": metric,
                    "bit_setting": bit_setting,
                    "test": "friedman",
                    "n_blocks": n_b,
                    "n_modes": k,
                    "statistic": statistic,
                    "p_value": p_value,
                    "modes": ",".join(mode_order),
                }
            )
    return pd.DataFrame(rows_out)


def compute_pairwise_wilcoxon_rotation_modes(
    trial_frame: pd.DataFrame,
    *,
    modes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Pairwise two-sided Wilcoxon signed-rank on paired blocks; Holm adjustment within each metric x bit_setting."""
    if trial_frame.empty:
        return pd.DataFrame()
    mode_order = modes or ROTATION_COMPARE_MODES
    pair_columns = triality_pairing_columns(trial_frame)
    metrics = ("hidden_cosine_similarity", "next_logit_kl")
    rows: list[dict[str, float | int | str | bool]] = []
    bit_settings = sorted(
        {str(x) for x in trial_frame["bit_setting"].unique()} - {"exact"},
        key=bit_setting_sort_key,
    )
    for metric in metrics:
        for bit_setting in bit_settings:
            block_rows: list[dict[str, float | int | str | bool]] = []
            for mode_a, mode_b in combinations(mode_order, 2):
                left = trial_frame.loc[
                    (trial_frame["mode"] == mode_a) & (trial_frame["bit_setting"].astype(str) == str(bit_setting))
                ][pair_columns + [metric]].rename(columns={metric: "a"})
                right = trial_frame.loc[
                    (trial_frame["mode"] == mode_b) & (trial_frame["bit_setting"].astype(str) == str(bit_setting))
                ][pair_columns + [metric]].rename(columns={metric: "b"})
                paired = left.merge(right, on=pair_columns, how="inner")
                if len(paired) < 3:
                    continue
                try:
                    result = stats.wilcoxon(
                        paired["a"],
                        paired["b"],
                        alternative="two-sided",
                        zero_method="wilcox",
                        correction=False,
                        method="auto",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0
                block_rows.append(
                    {
                        "metric": metric,
                        "bit_setting": bit_setting,
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        "test": "wilcoxon",
                        "alternative": "two-sided",
                        "n_pairs": int(len(paired)),
                        "statistic": statistic,
                        "p_value": p_value,
                        "mean_a": float(paired["a"].mean()),
                        "mean_b": float(paired["b"].mean()),
                    }
                )
            if not block_rows:
                continue
            p_values = [float(r["p_value"]) for r in block_rows]
            adjusted = holm_bonferroni(p_values)
            for row, adj_p in zip(block_rows, adjusted, strict=True):
                row["p_value_holm"] = adj_p
                row["significant_0_05"] = adj_p < 0.05
            rows.extend(block_rows)
    return pd.DataFrame(rows)


def compute_triality_statistics(trial_frame: pd.DataFrame) -> pd.DataFrame:
    if trial_frame.empty:
        raise ValueError("compute_triality_statistics received an empty trial_frame")
    required_columns = {"mode", "dataset", "trial", "layer", "bit_setting", "bits"}
    missing_columns = sorted(required_columns.difference(trial_frame.columns))
    if missing_columns:
        raise ValueError(f"trial_frame is missing required columns: {missing_columns}")
    pair_columns = triality_pairing_columns(trial_frame)
    metrics = (
        ("hidden_cosine_similarity", "greater"),
        ("next_logit_kl", "less"),
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for metric, alternative in metrics:
        p_values: list[float] = []
        row_indices: list[int] = []
        baseline = trial_frame.loc[trial_frame["mode"] == "key_only_block_so8_learned", pair_columns + [metric]].rename(
            columns={metric: "baseline_value"}
        )
        for view in TRIALITY_PROXY_VIEWS:
            mode = triality_mode_name(view)
            candidate = trial_frame.loc[trial_frame["mode"] == mode, pair_columns + [metric]].rename(
                columns={metric: "candidate_value"}
            )
            paired = baseline.merge(candidate, on=pair_columns, how="inner")
            if paired.empty:
                continue
            try:
                result = stats.wilcoxon(
                    paired["candidate_value"],
                    paired["baseline_value"],
                    alternative=alternative,
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            row = {
                "metric": metric,
                "view": view,
                "mode": mode,
                "test": "wilcoxon",
                "alternative": alternative,
                "n_pairs": int(len(paired)),
                "statistic": statistic,
                "p_value": p_value,
                "candidate_mean": float(paired["candidate_value"].mean()),
                "baseline_mean": float(paired["baseline_value"].mean()),
                "delta_candidate_minus_baseline": float(
                    paired["candidate_value"].mean() - paired["baseline_value"].mean()
                ),
            }
            rows.append(row)
            p_values.append(p_value)
            row_indices.append(len(rows) - 1)
        adjusted = holm_bonferroni(p_values)
        for row_index, adjusted_p in zip(row_indices, adjusted, strict=True):
            rows[row_index]["p_value_holm"] = adjusted_p
            rows[row_index]["significant_0_05"] = adjusted_p < 0.05
    return pd.DataFrame(rows)


def build_best_per_layer_selector(trial_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best Triality view for each layer/bit by mean hidden cosine."""
    if trial_frame.empty:
        raise ValueError("build_best_per_layer_selector received an empty trial_frame")
    required_columns = {
        "mode",
        "view",
        "dataset",
        "trial",
        "layer",
        "bit_setting",
        "bits",
        "hidden_cosine_similarity",
    }
    missing_columns = sorted(required_columns.difference(trial_frame.columns))
    if missing_columns:
        raise ValueError(f"trial_frame is missing required columns for selector build: {missing_columns}")
    triality_rows = trial_frame.loc[trial_frame["mode"].isin(TRIALITY_MODE_BY_VIEW.values())].copy()
    if triality_rows.empty:
        raise ValueError("build_best_per_layer_selector found no triality rows in trial_frame")
    selection_rows: list[dict[str, float | int | str]] = []
    selected_groups: list[pd.DataFrame] = []
    for (layer, bit_setting), group in triality_rows.groupby(["layer", "bit_setting"], dropna=False, sort=True):
        view_summary = (
            group.groupby("view", dropna=False)["hidden_cosine_similarity"]
            .mean()
            .sort_values(ascending=False, kind="stable")
        )
        selected_view = str(view_summary.index[0])
        selected_mode = triality_mode_name(selected_view)
        selection_rows.append(
            {
                "layer": int(layer),
                "bit_setting": str(bit_setting),
                "bits": float(group.loc[group["view"] == selected_view, "bits"].iloc[0]),
                "selected_view": selected_view,
                "selected_mode": selected_mode,
                "selector_policy": "best_per_layer_hidden_cosine",
                "selected_hidden_cosine_mean": float(view_summary.iloc[0]),
            }
        )
        chosen = group.loc[group["view"] == selected_view].copy()
        chosen["mode"] = TRIALITY_SELECTOR_MODE
        chosen["key_mode"] = TRIALITY_SELECTOR_MODE
        chosen["selector_policy"] = "best_per_layer_hidden_cosine"
        chosen["selected_view"] = selected_view
        selected_groups.append(chosen)
    manifest = pd.DataFrame(selection_rows).sort_values(
        by=["bits", "layer"],
        key=lambda series: series.map(bit_setting_sort_key) if series.name == "bit_setting" else series,
    )
    selected_frame = pd.concat(selected_groups, ignore_index=True)
    return manifest, selected_frame


def compute_triality_selector_statistics(
    selected_frame: pd.DataFrame,
    trial_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Paired Wilcoxon gate comparing best-per-layer selector vs learned SO(8)."""
    if selected_frame.empty:
        raise ValueError("compute_triality_selector_statistics received an empty selected_frame")
    pair_columns = triality_pairing_columns(selected_frame)
    baseline = trial_frame.loc[
        trial_frame["mode"] == "key_only_block_so8_learned",
        pair_columns + ["hidden_cosine_similarity", "next_logit_kl", "memory_ratio_vs_exact"],
    ].rename(
        columns={
            "hidden_cosine_similarity": "baseline_hidden_cosine_similarity",
            "next_logit_kl": "baseline_next_logit_kl",
            "memory_ratio_vs_exact": "baseline_memory_ratio_vs_exact",
        }
    )
    candidate = selected_frame[pair_columns + ["hidden_cosine_similarity", "next_logit_kl", "memory_ratio_vs_exact"]].rename(
        columns={
            "hidden_cosine_similarity": "candidate_hidden_cosine_similarity",
            "next_logit_kl": "candidate_next_logit_kl",
            "memory_ratio_vs_exact": "candidate_memory_ratio_vs_exact",
        }
    )
    paired = baseline.merge(candidate, on=pair_columns, how="inner")
    if paired.empty:
        raise ValueError("No paired rows available for triality selector statistics")
    metric_specs = (
        ("hidden_cosine_similarity", "greater"),
        ("next_logit_kl", "less"),
    )
    rows: list[dict[str, float | int | str | bool]] = []
    raw_p_values: list[float] = []
    for metric, alternative in metric_specs:
        cand_col = f"candidate_{metric}"
        base_col = f"baseline_{metric}"
        try:
            result = stats.wilcoxon(
                paired[cand_col],
                paired[base_col],
                alternative=alternative,
                zero_method="wilcox",
                correction=False,
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        except ValueError:
            statistic = 0.0
            p_value = 1.0
        rows.append(
            {
                "selector_policy": "best_per_layer_hidden_cosine",
                "metric": metric,
                "alternative": alternative,
                "n_pairs": int(len(paired)),
                "statistic": statistic,
                "p_value": p_value,
                "candidate_mean": float(paired[cand_col].mean()),
                "baseline_mean": float(paired[base_col].mean()),
                "delta_candidate_minus_baseline": float(paired[cand_col].mean() - paired[base_col].mean()),
            }
        )
        raw_p_values.append(p_value)
    adjusted = holm_bonferroni(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_p
        row["significant_0_05"] = adjusted_p < 0.05
    memory_delta = float(
        paired["candidate_memory_ratio_vs_exact"].mean() - paired["baseline_memory_ratio_vs_exact"].mean()
    )
    hidden_gate = next(
        (
            bool(row["significant_0_05"]) and float(row["delta_candidate_minus_baseline"]) > 0
            for row in rows
            if row["metric"] == "hidden_cosine_similarity"
        ),
        False,
    )
    kl_gate = next(
        (
            float(row["delta_candidate_minus_baseline"]) <= 1e-9
            for row in rows
            if row["metric"] == "next_logit_kl"
        ),
        False,
    )
    promotion_row = {
        "selector_policy": "best_per_layer_hidden_cosine",
        "metric": "promotion_gate",
        "alternative": "compound",
        "n_pairs": int(len(paired)),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "candidate_mean": float("nan"),
        "baseline_mean": float("nan"),
        "delta_candidate_minus_baseline": memory_delta,
        "p_value_holm": float("nan"),
        "significant_0_05": bool(hidden_gate and kl_gate and abs(memory_delta) <= 1e-9),
        "hidden_gate_pass": hidden_gate,
        "next_logit_kl_gate_pass": kl_gate,
        "memory_ratio_delta": memory_delta,
    }
    rows.append(promotion_row)
    return pd.DataFrame(rows)
