"""Offline GGUF weight conversion helpers for experimental TQ4_1S exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
from math import prod
from pathlib import Path
import re
from typing import Any

import numpy as np

from turboquant.gguf_profiles import import_vendor_gguf
from turboquant.triality_contract import (
    build_default_weight_plan,
    build_triality_metadata,
    build_triality_payload,
    payload_json_dumps,
    validate_triality_payload,
)


_BLOCK_LAYER_RE = re.compile(r"^blk\.(\d+)\.")
_QWEN_9B_HINTS = ("qwen3.5-9b", "qwen35-9b", "qwen35_9b")
_QWEN_27B_HINTS = ("qwen3.5-27b", "qwen35-27b", "qwen35_27b")
_GEMMA_E2B_HINTS = ("gemma-4-e2b", "gemma4-e2b", "gemma 4 e2b")
_GEMMA_E4B_HINTS = ("gemma-4-e4b", "gemma4-e4b", "gemma 4 e4b")
_GEMMA_A4B_HINTS = ("gemma-4-26b-a4b", "gemma4-26b-a4b", "gemma 4 26b a4b", "gemma4-a4b")
_GEMMA4_ALWAYS_PRESERVE_PATTERNS = (
    "per_layer_token_embd.weight",
    "per_layer_model_proj.weight",
    "per_layer_proj_norm.weight",
    "blk.*.inp_gate.weight",
    "blk.*.proj.weight",
    "blk.*.layer_output_scale.weight",
)


@dataclass(frozen=True, slots=True)
class WeightGGUFConversionSummary:
    """Summary of an offline GGUF weight conversion run."""

    source_path: Path
    output_path: Path
    model_family: str
    converted_tensor_count: int
    preserved_tensor_count: int
    converted_tensor_names: tuple[str, ...]
    weight_plan: dict[str, Any]


def convert_weight_turboquant_gguf(
    *,
    source_path: Path,
    output_path: Path,
    model_family: str | None = None,
    mode: str = "triality-proxy-so8-pareto",
    weight_plan: dict[str, Any] | None = None,
) -> WeightGGUFConversionSummary:
    """Convert selected GGUF Q8_0 weight tensors into offline TQ4_1S blocks.

    The output file is a real tensor-byte rewrite, not only metadata. Runtime
    execution is intentionally deferred; this function only produces a GGUF
    artifact whose tensor storage reflects the requested offline codec plan.
    """

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(source_path)
    _ensure_no_existing_turboquant_namespace(reader=reader)

    arch = _require_source_architecture(reader)
    resolved_model_family = model_family or _infer_model_family(reader=reader, source_path=source_path)
    if resolved_model_family is None:
        raise ValueError(
            "Unable to infer model family from GGUF metadata or filename; pass model_family explicitly"
        )

    num_layers = _require_positive_int(reader=reader, key=f"{arch}.block_count")
    head_dim = _infer_head_dim(reader=reader, arch=arch)
    num_kv_heads = _infer_kv_head_count(reader=reader, arch=arch)

    default_weight_plan = build_default_weight_plan(
        model_family=resolved_model_family,
        num_layers=num_layers,
        source_ftype="q8_0",
    )
    resolved_weight_plan = default_weight_plan if weight_plan is None else weight_plan
    actual_weight_plan = _resolve_actual_weight_plan(resolved_weight_plan)

    payload = build_triality_payload(
        mode=mode,
        head_dim=head_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        model_family=resolved_model_family,
        weight_source_ftype=str(actual_weight_plan["source_ftype"]),
        weight_policy=str(actual_weight_plan["policy"]),
        weight_protected_roles=list(actual_weight_plan.get("protected_roles", [])),
        weight_protected_layers=[int(v) for v in actual_weight_plan.get("protected_layers", [])],
        modality_scope=str(actual_weight_plan.get("modality_scope", "text-only")),
    )
    payload["weight_plan"] = actual_weight_plan
    validate_triality_payload(payload)
    payload_json = payload_json_dumps(payload)
    metadata = build_triality_metadata(
        mode=mode,
        payload_json=payload_json,
        weight_plan=actual_weight_plan,
    )

    writer = gguf.GGUFWriter(output_path, arch=arch, use_temp_file=False)
    _copy_source_kv_metadata(reader=reader, writer=writer, gguf=gguf)
    writer.add_uint32("general.file_type", int(gguf.LlamaFileType.GUESSED))
    writer.add_string("hypura.turboquant.weight.generated_at_utc", datetime.now(timezone.utc).isoformat())
    for key, value in metadata.items():
        _add_metadata_value(writer=writer, gguf=gguf, key=key, value=value)

    converted_tensor_names: list[str] = []
    tensor_actions: list[tuple[str, Any]] = []
    for tensor in reader.tensors:
        tensor_array = np.array(tensor.data, copy=False)
        action = _plan_tensor_action(
            tensor=tensor,
            tensor_array=tensor_array,
            weight_plan=actual_weight_plan,
            model_family=resolved_model_family,
            gguf=gguf,
        )
        tensor_actions.append((tensor.name, action))
        if action["kind"] == "convert":
            byte_shape = tuple(
                int(v)
                for v in gguf.quant_shape_to_byte_shape(action["logical_shape"], action["target_qtype"])
            )
            writer.add_tensor_info(
                name=tensor.name,
                tensor_shape=byte_shape,
                tensor_dtype=np.dtype(np.uint8),
                tensor_nbytes=int(prod(byte_shape)),
                raw_dtype=action["target_qtype"],
            )
            converted_tensor_names.append(tensor.name)
        else:
            writer.add_tensor_info(
                name=tensor.name,
                tensor_shape=tuple(int(v) for v in tensor_array.shape),
                tensor_dtype=tensor_array.dtype,
                tensor_nbytes=int(tensor_array.nbytes),
                raw_dtype=tensor.tensor_type,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    action_map = dict(tensor_actions)
    for tensor in reader.tensors:
        action = action_map[tensor.name]
        tensor_array = np.array(tensor.data, copy=False)
        if action["kind"] == "convert":
            dequantized = gguf.dequantize(tensor_array, action["source_qtype"])
            converted = gguf.quantize(dequantized, action["target_qtype"])
            writer.write_tensor_data(converted, tensor_endianess=reader.endianess)
        else:
            writer.write_tensor_data(tensor_array, tensor_endianess=reader.endianess)
    writer.close()

    return WeightGGUFConversionSummary(
        source_path=source_path,
        output_path=output_path,
        model_family=resolved_model_family,
        converted_tensor_count=len(converted_tensor_names),
        preserved_tensor_count=len(reader.tensors) - len(converted_tensor_names),
        converted_tensor_names=tuple(converted_tensor_names),
        weight_plan=actual_weight_plan,
    )


def _resolve_actual_weight_plan(weight_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize the runtime plan to codecs the offline converter can emit today."""

    actual_plan = dict(weight_plan)
    tensor_plan = dict(weight_plan.get("tensor_plan", {}))
    actual_tensor_plan: dict[str, str] = {}
    source_ftype = str(weight_plan.get("source_ftype", "q8_0")).strip().lower()
    for tensor_name, tensor_codec in tensor_plan.items():
        normalized_codec = str(tensor_codec).strip().lower()
        actual_tensor_plan[str(tensor_name)] = "tq4_1s" if normalized_codec == "tq4_1s" else source_ftype
    actual_plan["tensor_plan"] = actual_tensor_plan
    return actual_plan


def _plan_tensor_action(
    *,
    tensor: Any,
    tensor_array: np.ndarray,
    weight_plan: dict[str, Any],
    model_family: str,
    gguf: Any,
) -> dict[str, Any]:
    target_codec = _target_codec_for_tensor(
        tensor_name=tensor.name,
        weight_plan=weight_plan,
        model_family=model_family,
    )
    if target_codec is None:
        return {"kind": "preserve"}
    if target_codec != "tq4_1s":
        return {"kind": "preserve"}
    if tensor.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError(
            f"Tensor {tensor.name} matched tq4_1s conversion but source dtype is {tensor.tensor_type.name}, expected Q8_0"
        )
    logical_shape = tuple(
        int(v) for v in gguf.dequantize(tensor_array, gguf.GGMLQuantizationType.Q8_0).shape
    )
    if len(logical_shape) < 2:
        raise ValueError(f"Tensor {tensor.name} matched tq4_1s conversion but shape {logical_shape} is not matrix-like")
    if logical_shape[-1] % 32 != 0:
        raise ValueError(
            f"Tensor {tensor.name} matched tq4_1s conversion but last dimension {logical_shape[-1]} is not divisible by 32"
        )
    return {
        "kind": "convert",
        "source_qtype": gguf.GGMLQuantizationType.Q8_0,
        "target_qtype": gguf.GGMLQuantizationType.TQ4_1S,
        "logical_shape": logical_shape,
    }


def _target_codec_for_tensor(
    *,
    tensor_name: str,
    weight_plan: dict[str, Any],
    model_family: str,
) -> str | None:
    if _is_gemma4_family(model_family) and any(
        fnmatch.fnmatchcase(tensor_name, pattern) for pattern in _GEMMA4_ALWAYS_PRESERVE_PATTERNS
    ):
        return None
    layer_idx = _tensor_layer_index(tensor_name)
    protected_layers = {int(v) for v in weight_plan.get("protected_layers", [])}
    if layer_idx is not None and layer_idx in protected_layers:
        return None
    for pattern, codec in weight_plan.get("tensor_plan", {}).items():
        if fnmatch.fnmatchcase(tensor_name, str(pattern)):
            return str(codec).strip().lower()
    return None


def _tensor_layer_index(tensor_name: str) -> int | None:
    match = _BLOCK_LAYER_RE.match(tensor_name)
    if match is None:
        return None
    return int(match.group(1))


def _copy_source_kv_metadata(*, reader: Any, writer: Any, gguf: Any) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue
        if key.startswith("hypura.turboquant."):
            raise ValueError(
                "Source GGUF already contains Hypura TurboQuant metadata. Refusing to overwrite an existing namespace."
            )
        writer.add_key_value(
            key,
            field.contents(),
            field.types[0],
            field.types[-1] if field.types and field.types[0] == gguf.GGUFValueType.ARRAY else None,
        )


def _add_metadata_value(*, writer: Any, gguf: Any, key: str, value: Any) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
        return
    if isinstance(value, int) and not isinstance(value, bool):
        writer.add_int64(key, value)
        return
    if isinstance(value, float):
        writer.add_float64(key, value)
        return
    if isinstance(value, str):
        writer.add_string(key, value)
        return
    if isinstance(value, list):
        writer.add_array(key, value)
        return
    raise TypeError(f"Unsupported metadata type for {key}: {type(value)!r}")


def _ensure_no_existing_turboquant_namespace(*, reader: Any) -> None:
    for key in reader.fields:
        if key.startswith("hypura.turboquant."):
            raise ValueError(
                "Source GGUF already contains hypura.turboquant metadata. Refusing to overwrite an existing namespace."
            )


def _require_source_architecture(reader: Any) -> str:
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError("GGUF metadata is missing general.architecture")
    arch = str(arch_field.contents())
    if not arch:
        raise ValueError("general.architecture must not be empty")
    return arch


def _require_positive_int(*, reader: Any, key: str) -> int:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"Missing GGUF metadata field: {key}")
    value = int(field.contents())
    if value <= 0:
        raise ValueError(f"GGUF metadata field must be positive: {key}={value}")
    return value


def _infer_head_dim(*, reader: Any, arch: str) -> int:
    key_length_field = reader.get_field(f"{arch}.attention.key_length")
    if key_length_field is not None:
        head_dim = int(key_length_field.contents())
        if head_dim <= 0:
            raise ValueError(f"Invalid {arch}.attention.key_length: {head_dim}")
        return head_dim
    embedding_length = _require_positive_int(reader=reader, key=f"{arch}.embedding_length")
    head_count = _require_positive_int(reader=reader, key=f"{arch}.attention.head_count")
    if embedding_length % head_count != 0:
        raise ValueError(
            f"{arch}.embedding_length must be divisible by {arch}.attention.head_count; got {embedding_length} and {head_count}"
        )
    return embedding_length // head_count


def _infer_kv_head_count(*, reader: Any, arch: str) -> int:
    kv_field = reader.get_field(f"{arch}.attention.head_count_kv")
    if kv_field is not None:
        kv_heads = int(kv_field.contents())
        if kv_heads <= 0:
            raise ValueError(f"Invalid {arch}.attention.head_count_kv: {kv_heads}")
        return kv_heads
    return _require_positive_int(reader=reader, key=f"{arch}.attention.head_count")


def _infer_model_family(*, reader: Any, source_path: Path) -> str | None:
    candidate_values = [
        _read_optional_string(reader, "general.name"),
        _read_optional_string(reader, "general.basename"),
        source_path.stem,
        str(source_path.parent.name),
    ]
    lowered = " ".join(value.lower() for value in candidate_values if value)
    if any(hint in lowered for hint in _GEMMA_A4B_HINTS):
        return "google/gemma-4-26b-a4b-it"
    if any(hint in lowered for hint in _GEMMA_E4B_HINTS):
        return "google/gemma-4-e4b-it"
    if any(hint in lowered for hint in _GEMMA_E2B_HINTS):
        return "google/gemma-4-e2b-it"
    if any(hint in lowered for hint in _QWEN_27B_HINTS):
        return "Qwen/Qwen3.5-27B"
    if any(hint in lowered for hint in _QWEN_9B_HINTS):
        return "Qwen/Qwen3.5-9B"
    return None


def _is_gemma4_family(model_family: str) -> bool:
    lowered = model_family.strip().lower()
    return "gemma-4" in lowered or "gemma4" in lowered


def _read_optional_string(reader: Any, key: str) -> str | None:
    field = reader.get_field(key)
    if field is None:
        return None
    value = str(field.contents()).strip()
    return value or None


__all__ = [
    "WeightGGUFConversionSummary",
    "convert_weight_turboquant_gguf",
]
