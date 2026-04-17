"""JSON schema helpers for paper-baseline and research-extension artifacts."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any
import json

from turboquant.allocation import ChannelBitAllocation
from turboquant.io_utils import write_json
from turboquant.paper_baseline.types import PaperMixedBitPolicy
from turboquant.research_extension.types import KeyResearchConfig, ValueResearchConfig


PAPER_SCHEMA_KIND = "paper_baseline"
RESEARCH_SCHEMA_KIND = "research_extension"
SCHEMA_VERSION = 2
ARTIFACT_METADATA_SCHEMA_VERSION = 1
CAPTURE_QUANTIZATION_CONFIG_SCHEMA_VERSION = 1
PAPER_MODE_NAMES = ("exact", "key_only_random", "full_kv")
DEFAULT_SIGN_PACK_FORMAT = "int8_unpacked_binary"
DEFAULT_BITWIDTH_PAYLOAD_DTYPE = "uint8"
TURBOQUANT_REFERENCE_PAPER_URL = "https://arxiv.org/abs/2504.19874"
TURBOQUANT_GGUF_FLOAT_KEYS = (
    "tq_total_bits",
    "tq_runtime_bits_per_channel",
    "tq_stage1_effective_bits",
)
TURBOQUANT_GGUF_U32_KEYS = (
    "tq_qjl_bits",
    "tq_qjl_dim",
    "tq_rotation_seed",
    "tq_qjl_seed",
)
TURBOQUANT_GGUF_STRING_KEYS = (
    "tq_rotation_policy",
    "tq_triality_mode",
    "tq_triality_view",
    "tq_stage1_allocation_scheme",
    "tq_stage1_bitwidth_payload_dtype",
    "tq_norm_dtype",
    "tq_sign_pack_format",
)


def _require(mapping: dict[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def _validate_paper_mode_config(mode: str, payload: dict[str, Any]) -> None:
    _require(
        payload,
        (
            "mode",
            "bits_total",
            "mse_bits",
            "qjl_bits",
            "rotation_policy",
            "rotation_seed",
            "norm_mode",
            "codebook_kind",
            "mixed_bit_policy",
        ),
        context=f"paper mode config {mode}",
    )
    if payload["mode"] != mode:
        raise ValueError(f"paper mode config key {mode!r} must match payload mode {payload['mode']!r}")
    if payload["rotation_policy"] != "random_haar":
        raise ValueError("paper schema only supports rotation_policy='random_haar'")
    if payload["norm_mode"] != "explicit":
        raise ValueError("paper schema only supports norm_mode='explicit'")
    mixed = payload["mixed_bit_policy"]
    _require(
        mixed,
        ("enabled", "low_bits", "high_bits", "high_count", "selector"),
        context=f"paper mixed-bit policy {mode}",
    )
    if mixed["selector"] != "paper_outlier_magnitude":
        raise ValueError("paper schema only supports selector='paper_outlier_magnitude'")


def build_paper_turboquant_config(
    *,
    bit_grid: list[float],
    dim: int,
    rotation_seed: int = 0,
    qjl_seed: int = 1,
    artifact_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    evaluation_grid = {
        "modes": list(PAPER_MODE_NAMES),
        "bit_grid": list(bit_grid),
        "dim": dim,
        "qjl_seed": qjl_seed,
    }
    mode_configs: dict[str, dict[str, Any]] = {}
    for mode in PAPER_MODE_NAMES:
        mode_configs[mode] = {
            "mode": mode,
            "bits_total": None,
            "mse_bits": None,
            "qjl_bits": 1 if mode != "exact" else None,
            "rotation_policy": "random_haar",
            "rotation_seed": rotation_seed,
            "norm_mode": "explicit",
            "codebook_kind": "sphere-lloyd-max",
            "mixed_bit_policy": {
                "enabled": mode != "exact",
                "low_bits": None,
                "high_bits": None,
                "high_count": None,
                "selector": "paper_outlier_magnitude",
                "presets": {
                    "2.5": asdict(PaperMixedBitPolicy.for_total_bits(2.5, dim)),
                    "3.5": asdict(PaperMixedBitPolicy.for_total_bits(3.5, dim)),
                },
            },
        }
    payload = {
        "schema_kind": PAPER_SCHEMA_KIND,
        "version": SCHEMA_VERSION,
        "runtime_target": "hf_qwen_captured_replay",
        "evaluation_grid": evaluation_grid,
        "mode_configs": mode_configs,
        "artifact_refs": artifact_refs or {},
    }
    validate_paper_turboquant_config(payload)
    return payload


def build_research_turboquant_config(
    *,
    key_config: KeyResearchConfig,
    value_config: ValueResearchConfig,
    artifact_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    qjl_dim = key_config.qjl_dim if key_config.qjl_dim is not None else key_config.head_dim
    stage1_effective_bits = float(key_config.bits_total - key_config.qjl_bits)
    if stage1_effective_bits < 0:
        raise ValueError("bits_total must be >= qjl_bits")
    payload = {
        "schema_kind": RESEARCH_SCHEMA_KIND,
        "version": SCHEMA_VERSION,
        "k_codec": {
            "bits_total": float(key_config.bits_total),
            "mse_bits": key_config.mse_bits,
            "qjl_bits": key_config.qjl_bits,
            "qjl_dim": qjl_dim,
            "stage1_effective_bits": stage1_effective_bits,
            "rotation_policy": key_config.rotation_policy,
            "rotation_seed": key_config.rotation_seed,
            "qjl_seed": key_config.qjl_seed,
            "head_dim": key_config.head_dim,
            "view_mode": key_config.view_mode,
            "view_selection": key_config.view_selection,
            "triality_mode": "triality_proxy" if key_config.view_mode == "triality_proxy" else "single_view",
            "triality_view": key_config.views[0] if key_config.views else "",
            "views": list(key_config.views),
            "stage1_allocation_scheme": "uniform",
            "stage1_bitwidth_payload_dtype": DEFAULT_BITWIDTH_PAYLOAD_DTYPE,
            "norm_dtype": key_config.dtype,
            "sign_pack_format": DEFAULT_SIGN_PACK_FORMAT,
        },
        "v_codec": {
            "base_bits": value_config.base_bits,
            "high_bits": value_config.high_bits,
            "protected_fraction": value_config.protected_fraction,
            "secondary_fraction": value_config.secondary_fraction,
            "low_rank_rank": value_config.low_rank_rank,
        },
        "channel_group_size": value_config.channel_group_size,
        "sensitivity_source": value_config.sensitivity_source,
        "protected_fraction": value_config.protected_fraction,
        "secondary_fraction": value_config.secondary_fraction,
        "low_rank_rank": value_config.low_rank_rank,
        "rotation_policy": key_config.rotation_policy,
        "view_mode": key_config.view_mode,
        "view_selection": key_config.view_selection,
        "views": list(key_config.views),
        "artifact_refs": artifact_refs or {},
    }
    validate_research_turboquant_config(payload)
    return payload


def validate_paper_turboquant_config(payload: dict[str, Any]) -> None:
    _require(payload, ("schema_kind", "version", "evaluation_grid", "mode_configs", "artifact_refs"), context="paper config")
    if payload["schema_kind"] != PAPER_SCHEMA_KIND:
        raise ValueError(f"Expected schema_kind={PAPER_SCHEMA_KIND!r}, got {payload['schema_kind']!r}")
    evaluation_grid = payload["evaluation_grid"]
    _require(evaluation_grid, ("modes", "bit_grid", "dim", "qjl_seed"), context="paper evaluation_grid")
    mode_configs = payload["mode_configs"]
    for mode in PAPER_MODE_NAMES:
        if mode not in mode_configs:
            raise ValueError(f"Missing paper mode config: {mode}")
        _validate_paper_mode_config(mode, mode_configs[mode])


def validate_research_turboquant_config(payload: dict[str, Any]) -> None:
    _require(
        payload,
        (
            "schema_kind",
            "version",
            "k_codec",
            "v_codec",
            "channel_group_size",
            "sensitivity_source",
            "protected_fraction",
            "secondary_fraction",
            "low_rank_rank",
            "rotation_policy",
            "view_mode",
            "view_selection",
            "views",
            "artifact_refs",
        ),
        context="research config",
    )
    if payload["schema_kind"] != RESEARCH_SCHEMA_KIND:
        raise ValueError(f"Expected schema_kind={RESEARCH_SCHEMA_KIND!r}, got {payload['schema_kind']!r}")
    k_codec = payload["k_codec"]
    _require(
        k_codec,
        (
            "bits_total",
            "mse_bits",
            "qjl_bits",
            "qjl_dim",
            "stage1_effective_bits",
            "rotation_policy",
            "rotation_seed",
            "qjl_seed",
            "head_dim",
            "view_mode",
            "view_selection",
            "triality_mode",
            "triality_view",
            "views",
            "stage1_allocation_scheme",
            "stage1_bitwidth_payload_dtype",
            "norm_dtype",
            "sign_pack_format",
        ),
        context="research k_codec",
    )
    if float(k_codec["bits_total"]) < float(k_codec["qjl_bits"]):
        raise ValueError("research k_codec bits_total must be >= qjl_bits")


def build_capture_quantization_config(
    *,
    weight_load: str,
    requested_dtype: str,
    trust_remote_code: bool,
    max_length: int,
    device_map: str = "auto",
) -> dict[str, Any]:
    """Build a reproducible capture-time quantization payload."""

    payload = {
        "schema_version": CAPTURE_QUANTIZATION_CONFIG_SCHEMA_VERSION,
        "weight_load": weight_load,
        "requested_dtype": requested_dtype,
        "quantization_backend": "bitsandbytes" if weight_load in {"4bit", "8bit"} else "none",
        "load_in_4bit": weight_load == "4bit",
        "load_in_8bit": weight_load == "8bit",
        "device_map": device_map,
        "trust_remote_code": bool(trust_remote_code),
        "max_length": int(max_length),
    }
    validate_capture_quantization_config(payload)
    return payload


def validate_capture_quantization_config(payload: dict[str, Any]) -> None:
    """Validate capture-time quantization metadata."""

    _require(
        payload,
        (
            "schema_version",
            "weight_load",
            "requested_dtype",
            "quantization_backend",
            "load_in_4bit",
            "load_in_8bit",
            "device_map",
            "trust_remote_code",
            "max_length",
        ),
        context="capture quantization config",
    )
    weight_load = payload["weight_load"]
    if weight_load not in {"4bit", "8bit", "none"}:
        raise ValueError(f"Unsupported capture weight_load: {weight_load!r}")
    if int(payload["max_length"]) <= 0:
        raise ValueError(f"capture max_length must be positive, got {payload['max_length']!r}")
    expected_backend = "bitsandbytes" if weight_load in {"4bit", "8bit"} else "none"
    if payload["quantization_backend"] != expected_backend:
        raise ValueError(
            "capture quantization backend is inconsistent: "
            f"expected {expected_backend!r} for weight_load={weight_load!r}"
        )
    if bool(payload["load_in_4bit"]) != (weight_load == "4bit"):
        raise ValueError("capture quantization config has inconsistent load_in_4bit state")
    if bool(payload["load_in_8bit"]) != (weight_load == "8bit"):
        raise ValueError("capture quantization config has inconsistent load_in_8bit state")


def build_turboquant_artifact_metadata(
    *,
    total_bits: float,
    qjl_bits: int,
    qjl_dim: int,
    rotation_policy: str,
    rotation_seed: int,
    qjl_seed: int,
    triality_mode: str,
    triality_view: str,
    width: int,
    allocation: ChannelBitAllocation | None,
    bitwidth_payload_dtype: str = DEFAULT_BITWIDTH_PAYLOAD_DTYPE,
    norm_dtype: str = "float32",
    sign_pack_format: str = DEFAULT_SIGN_PACK_FORMAT,
) -> dict[str, Any]:
    """Build an explicit metadata contract for stored TurboQuant artifacts.

    The metadata separates the user-facing total bits/channel label from the
    Stage 1 allocation details and the Stage 2 QJL contribution so downstream
    loaders do not have to infer mixed-bit behavior from implicit floor rules.
    ``tq_runtime_bits_per_channel`` records the actual average bits implied by
    the stored allocation plus QJL.
    """

    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if qjl_bits < 0:
        raise ValueError(f"qjl_bits must be non-negative, got {qjl_bits}")
    if qjl_dim <= 0:
        raise ValueError(f"qjl_dim must be positive, got {qjl_dim}")

    if allocation is None:
        stage1_effective_bits = float(total_bits - qjl_bits)
        if not float(stage1_effective_bits).is_integer():
            raise ValueError(
                "Non-integer stage1_effective_bits require an explicit ChannelBitAllocation; "
                f"got total_bits={total_bits}, qjl_bits={qjl_bits}"
            )
        stage1_allocation_scheme = "uniform"
        stage1_regular_bits = int(round(stage1_effective_bits))
        stage1_outlier_bits = stage1_regular_bits
        stage1_outlier_count = 0
        stage1_outlier_ratio = 0.0
    else:
        stage1_effective_bits = float(allocation.effective_bits(width))
        stage1_allocation_scheme = allocation.selection_policy
        stage1_regular_bits = allocation.regular_bits
        stage1_outlier_bits = allocation.outlier_bits
        stage1_outlier_count = allocation.outlier_count
        stage1_outlier_ratio = allocation.outlier_ratio(width)

    if stage1_effective_bits < 0:
        raise ValueError("Stage 1 effective bits must be non-negative")

    payload = {
        "tq_schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "tq_total_bits": float(total_bits),
        "tq_runtime_bits_per_channel": float(stage1_effective_bits + qjl_bits),
        "tq_stage1_effective_bits": stage1_effective_bits,
        "tq_qjl_bits": int(qjl_bits),
        "tq_qjl_dim": int(qjl_dim),
        "tq_rotation_policy": rotation_policy,
        "tq_rotation_seed": int(rotation_seed),
        "tq_qjl_seed": int(qjl_seed),
        "tq_triality_mode": triality_mode,
        "tq_triality_view": triality_view,
        "tq_stage1_allocation_scheme": stage1_allocation_scheme,
        "tq_stage1_regular_bits": int(stage1_regular_bits),
        "tq_stage1_outlier_bits": int(stage1_outlier_bits),
        "tq_stage1_outlier_count": int(stage1_outlier_count),
        "tq_stage1_outlier_ratio": float(stage1_outlier_ratio),
        "tq_stage1_width": int(width),
        "tq_stage1_bitwidth_payload_dtype": bitwidth_payload_dtype,
        "tq_norm_dtype": norm_dtype,
        "tq_sign_pack_format": sign_pack_format,
    }
    validate_turboquant_artifact_metadata(payload)
    return payload


def validate_turboquant_artifact_metadata(payload: dict[str, Any]) -> None:
    """Validate the explicit ABI metadata stored alongside TurboQuant artifacts."""

    _require(
        payload,
        (
            "tq_schema_version",
            "tq_total_bits",
            "tq_runtime_bits_per_channel",
            "tq_stage1_effective_bits",
            "tq_qjl_bits",
            "tq_qjl_dim",
            "tq_rotation_policy",
            "tq_rotation_seed",
            "tq_qjl_seed",
            "tq_triality_mode",
            "tq_triality_view",
            "tq_stage1_allocation_scheme",
            "tq_stage1_bitwidth_payload_dtype",
            "tq_norm_dtype",
            "tq_sign_pack_format",
        ),
        context="turboquant artifact metadata",
    )
    total_bits = float(payload["tq_total_bits"])
    runtime_bits_per_channel = float(payload["tq_runtime_bits_per_channel"])
    stage1_effective_bits = float(payload["tq_stage1_effective_bits"])
    qjl_bits = int(payload["tq_qjl_bits"])
    qjl_dim = int(payload["tq_qjl_dim"])
    if total_bits < 0:
        raise ValueError(f"tq_total_bits must be non-negative, got {total_bits}")
    if runtime_bits_per_channel < 0:
        raise ValueError(f"tq_runtime_bits_per_channel must be non-negative, got {runtime_bits_per_channel}")
    if stage1_effective_bits < 0:
        raise ValueError(f"tq_stage1_effective_bits must be non-negative, got {stage1_effective_bits}")
    if qjl_bits < 0:
        raise ValueError(f"tq_qjl_bits must be non-negative, got {qjl_bits}")
    if qjl_dim <= 0:
        raise ValueError(f"tq_qjl_dim must be positive, got {qjl_dim}")
    if not math.isclose(stage1_effective_bits + qjl_bits, runtime_bits_per_channel, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Artifact metadata is inconsistent: "
            f"tq_stage1_effective_bits + tq_qjl_bits != tq_runtime_bits_per_channel "
            f"({stage1_effective_bits} + {qjl_bits} != {runtime_bits_per_channel})"
        )


def build_turboquant_gguf_contract(layer_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the canonical top-level GGUF `tq_*` contract from per-layer artifact metadata."""

    if not layer_metadata:
        raise ValueError("layer_metadata must contain at least one per-layer metadata payload")

    normalized: list[dict[str, Any]] = []
    schema_version: int | None = None
    for index, metadata in enumerate(layer_metadata):
        validate_turboquant_artifact_metadata(metadata)
        current_schema_version = int(metadata["tq_schema_version"])
        if schema_version is None:
            schema_version = current_schema_version
        elif current_schema_version != schema_version:
            raise ValueError(
                "All per-layer artifact metadata must agree on tq_schema_version; "
                f"layer 0 has {schema_version}, layer {index} has {current_schema_version}"
            )
        normalized.append(metadata)

    assert schema_version is not None
    payload: dict[str, Any] = {
        "tq_schema_version": schema_version,
    }
    for key in TURBOQUANT_GGUF_FLOAT_KEYS:
        payload[key] = [float(metadata[key]) for metadata in normalized]
    for key in TURBOQUANT_GGUF_U32_KEYS:
        payload[key] = [int(metadata[key]) for metadata in normalized]
    for key in TURBOQUANT_GGUF_STRING_KEYS:
        payload[key] = [str(metadata[key]) for metadata in normalized]

    validate_turboquant_gguf_contract(payload, expected_len=len(normalized))
    return payload


def build_uniform_turboquant_gguf_contract(*, artifact_metadata: dict[str, Any], num_layers: int) -> dict[str, Any]:
    """Expand one artifact metadata payload across `num_layers` for fixture-style GGUF exports."""

    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    validate_turboquant_artifact_metadata(artifact_metadata)
    return build_turboquant_gguf_contract([artifact_metadata.copy() for _ in range(num_layers)])


def validate_turboquant_gguf_contract(payload: dict[str, Any], *, expected_len: int | None = None) -> None:
    """Validate the canonical top-level GGUF `tq_*` contract."""

    _require(
        payload,
        ("tq_schema_version", *TURBOQUANT_GGUF_FLOAT_KEYS, *TURBOQUANT_GGUF_U32_KEYS, *TURBOQUANT_GGUF_STRING_KEYS),
        context="turboquant gguf contract",
    )

    schema_version = int(payload["tq_schema_version"])
    if schema_version <= 0:
        raise ValueError(f"tq_schema_version must be positive, got {schema_version}")

    lengths: set[int] = set()
    for key in (*TURBOQUANT_GGUF_FLOAT_KEYS, *TURBOQUANT_GGUF_U32_KEYS, *TURBOQUANT_GGUF_STRING_KEYS):
        values = payload[key]
        if not isinstance(values, list):
            raise ValueError(f"GGUF contract key {key!r} must be a list")
        lengths.add(len(values))

    if not lengths or 0 in lengths or len(lengths) != 1:
        raise ValueError("All GGUF contract arrays must share the same positive length")

    array_len = next(iter(lengths))
    if expected_len is not None and array_len != expected_len:
        raise ValueError(f"GGUF contract arrays must have length {expected_len}, got {array_len}")

    for index in range(array_len):
        validate_turboquant_artifact_metadata(
            {
                "tq_schema_version": schema_version,
                **{key: float(payload[key][index]) for key in TURBOQUANT_GGUF_FLOAT_KEYS},
                **{key: int(payload[key][index]) for key in TURBOQUANT_GGUF_U32_KEYS},
                **{key: str(payload[key][index]) for key in TURBOQUANT_GGUF_STRING_KEYS},
            }
        )


def read_turboquant_config(path: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = payload.get("schema_kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"Expected schema kind {expected_kind!r}, got {kind!r}")
    if kind == PAPER_SCHEMA_KIND:
        validate_paper_turboquant_config(payload)
    elif kind == RESEARCH_SCHEMA_KIND:
        validate_research_turboquant_config(payload)
    else:
        raise ValueError(f"Unknown turboquant schema kind: {kind!r}")
    return payload


def write_turboquant_config(path: Path, payload: dict[str, Any]) -> None:
    kind = payload.get("schema_kind")
    if kind == PAPER_SCHEMA_KIND:
        validate_paper_turboquant_config(payload)
    elif kind == RESEARCH_SCHEMA_KIND:
        validate_research_turboquant_config(payload)
    else:
        raise ValueError(f"Unknown turboquant schema kind: {kind!r}")
    write_json(path, payload)


__all__ = [
    "ARTIFACT_METADATA_SCHEMA_VERSION",
    "CAPTURE_QUANTIZATION_CONFIG_SCHEMA_VERSION",
    "DEFAULT_BITWIDTH_PAYLOAD_DTYPE",
    "DEFAULT_SIGN_PACK_FORMAT",
    "PAPER_SCHEMA_KIND",
    "RESEARCH_SCHEMA_KIND",
    "SCHEMA_VERSION",
    "TURBOQUANT_REFERENCE_PAPER_URL",
    "TURBOQUANT_GGUF_FLOAT_KEYS",
    "TURBOQUANT_GGUF_STRING_KEYS",
    "TURBOQUANT_GGUF_U32_KEYS",
    "build_turboquant_gguf_contract",
    "build_uniform_turboquant_gguf_contract",
    "build_capture_quantization_config",
    "build_turboquant_artifact_metadata",
    "build_paper_turboquant_config",
    "build_research_turboquant_config",
    "read_turboquant_config",
    "validate_turboquant_gguf_contract",
    "validate_capture_quantization_config",
    "validate_turboquant_artifact_metadata",
    "validate_paper_turboquant_config",
    "validate_research_turboquant_config",
    "write_turboquant_config",
]
