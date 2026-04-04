"""Captured KV evaluation modes shared by multiscreen / VRAM comparison scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import (
    _base_kv_codec,
    _evaluate_mode,
    _seed_value,
    evaluate_multiscreen_relevance_attention_row,
    select_queries,
)
from turboquant.research_extension.k_triality import (
    MULTISCREEN_TRIALITY_VECTOR_MODE,
    PRODUCTION_K_TURBOQUANT_MODE,
    _evaluate_triality_proxy_mode,
    evaluate_multiscreen_triality_vector_row,
    load_triality_proxy_rotations,
)

CAPTURED_KEY_EVAL_MODES = frozenset(
    {
        "exact",
        "key_only_random",
        "key_only_block_so8_static",
        "key_only_block_so8_triality_vector",
        MULTISCREEN_TRIALITY_VECTOR_MODE,
        "multiscreen_relevance",
    }
)


def dataset_label_captured_kv(bundle) -> str:
    label = bundle.metadata.prompt_label or "capture"
    return f"multiscreen_kv:{label}"


def clamp_multiscreen_allocation_for_keys(
    keys: torch.Tensor,
    alloc: ChannelBitAllocation,
) -> ChannelBitAllocation:
    """Clamp ``outlier_count`` to at most ``batch * heads * seq`` relevance cells."""

    n = int(keys.shape[0] * keys.shape[1] * keys.shape[2])
    oc = min(alloc.outlier_count, n)
    if oc == alloc.outlier_count:
        return alloc
    return ChannelBitAllocation.from_multiscreen_relevance(
        regular_bits=alloc.regular_bits,
        outlier_bits=alloc.outlier_bits,
        outlier_count=oc,
    )


def eval_captured_key_mode_row(
    *,
    mode: str,
    bundle,
    trial: int,
    bit_value: float,
    eval_device: torch.device,
    rotation_dir: Path | None,
    triality_artifacts: dict | None,
    ms_alloc: ChannelBitAllocation | None,
) -> dict[str, float | int | str]:
    """Single trial × layer row for a captured KV bundle (paper-style metrics + peak VRAM when on CUDA)."""

    keys = bundle.keys.to(eval_device)
    values = bundle.values.to(eval_device)
    layer_idx = bundle.layer_idx
    dataset = dataset_label_captured_kv(bundle)
    queries = select_queries(keys, seed=_seed_value(trial, layer_idx, 11 + int(bit_value * 10)))
    qjl_seed = _seed_value(trial, layer_idx, 83 + int(bit_value * 10))

    if mode == "exact":
        return _evaluate_mode(
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

    if mode == "key_only_random":
        rot = "random_haar"
        rot_seed = _seed_value(trial, layer_idx, 11 + int(bit_value * 10))
    elif mode == "key_only_block_so8_static":
        rot = "block_so8_static"
        rot_seed = _seed_value(trial, layer_idx, 29 + int(bit_value * 10))
    else:
        rot = ""
        rot_seed = 0

    if mode in ("key_only_random", "key_only_block_so8_static"):
        codec = _base_kv_codec(
            keys=keys,
            bit_value=bit_value,
            rotation_policy=rot,
            rotation_seed=rot_seed,
            qjl_seed=qjl_seed,
        )
        return _evaluate_mode(
            dataset=dataset,
            layer_idx=layer_idx,
            trial=trial,
            bit_setting=f"{bit_value:g}",
            bits=bit_value,
            mode=mode,
            keys=keys,
            values=values,
            queries=queries,
            codec=codec,
            value_mode="exact",
            calibrate_codec=False,
            mode_metadata={
                "key_mode": mode,
                "value_mode": "exact",
                "value_rotation_policy": "exact",
                "rotation_seed": rot_seed,
                "qjl_seed": qjl_seed,
            },
        )

    if mode == "key_only_block_so8_triality_vector":
        if triality_artifacts is None or rotation_dir is None:
            raise ValueError("key_only_block_so8_triality_vector requires rotation_dir and loaded artifacts")
        key = (layer_idx, float(bit_value), "vector")
        if key not in triality_artifacts:
            raise KeyError(
                f"Missing triality artifact for layer={layer_idx}, bits={bit_value:g}, view=vector under {rotation_dir}"
            )
        artifact = triality_artifacts[key]
        return _evaluate_triality_proxy_mode(
            dataset=dataset,
            trial=trial,
            layer_idx=layer_idx,
            bit_value=bit_value,
            view="vector",
            keys=keys,
            values=values,
            queries=queries,
            artifact=artifact,
        )

    if mode == "multiscreen_relevance":
        if ms_alloc is None:
            raise ValueError("ms_alloc required for multiscreen_relevance")
        alloc_use = clamp_multiscreen_allocation_for_keys(keys, ms_alloc)
        return evaluate_multiscreen_relevance_attention_row(
            dataset=dataset,
            trial=trial,
            layer_idx=layer_idx,
            bit_value=bit_value,
            keys=keys,
            values=values,
            queries=queries,
            allocation=alloc_use,
            rotation_policy="block_so8_static",
            rotation_seed=_seed_value(trial, layer_idx, 29 + int(bit_value * 10)),
            qjl_seed=qjl_seed,
        )

    if mode == MULTISCREEN_TRIALITY_VECTOR_MODE:
        if ms_alloc is None or triality_artifacts is None or rotation_dir is None:
            raise ValueError(
                f"{MULTISCREEN_TRIALITY_VECTOR_MODE} requires ms_alloc, rotation_dir, and triality artifacts"
            )
        key = (layer_idx, float(bit_value), "vector")
        if key not in triality_artifacts:
            raise KeyError(
                f"Missing triality artifact for layer={layer_idx}, bits={bit_value:g}, view=vector under {rotation_dir}"
            )
        artifact = triality_artifacts[key]
        alloc_use = clamp_multiscreen_allocation_for_keys(keys, ms_alloc)
        return evaluate_multiscreen_triality_vector_row(
            dataset=dataset,
            trial=trial,
            layer_idx=layer_idx,
            bit_value=bit_value,
            keys=keys,
            values=values,
            queries=queries,
            allocation=alloc_use,
            artifact=artifact,
        )

    raise ValueError(f"Unsupported mode: {mode!r}")


__all__ = [
    "CAPTURED_KEY_EVAL_MODES",
    "MULTISCREEN_TRIALITY_VECTOR_MODE",
    "PRODUCTION_K_TURBOQUANT_MODE",
    "clamp_multiscreen_allocation_for_keys",
    "dataset_label_captured_kv",
    "eval_captured_key_mode_row",
]
