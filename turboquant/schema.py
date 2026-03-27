"""JSON schema helpers for paper-baseline and research-extension artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

from turboquant.io_utils import write_json
from turboquant.paper_baseline.types import PaperMixedBitPolicy
from turboquant.research_extension.types import KeyResearchConfig, ValueResearchConfig


PAPER_SCHEMA_KIND = "paper_baseline"
RESEARCH_SCHEMA_KIND = "research_extension"
SCHEMA_VERSION = 1
PAPER_MODE_NAMES = ("exact", "key_only_random", "full_kv")


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
    payload = {
        "schema_kind": RESEARCH_SCHEMA_KIND,
        "version": SCHEMA_VERSION,
        "k_codec": {
            "bits_total": key_config.bits_total,
            "mse_bits": key_config.mse_bits,
            "qjl_bits": key_config.qjl_bits,
            "rotation_policy": key_config.rotation_policy,
            "rotation_seed": key_config.rotation_seed,
            "qjl_seed": key_config.qjl_seed,
            "head_dim": key_config.head_dim,
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
            "artifact_refs",
        ),
        context="research config",
    )
    if payload["schema_kind"] != RESEARCH_SCHEMA_KIND:
        raise ValueError(f"Expected schema_kind={RESEARCH_SCHEMA_KIND!r}, got {payload['schema_kind']!r}")


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
    "PAPER_SCHEMA_KIND",
    "RESEARCH_SCHEMA_KIND",
    "SCHEMA_VERSION",
    "build_paper_turboquant_config",
    "build_research_turboquant_config",
    "read_turboquant_config",
    "validate_paper_turboquant_config",
    "validate_research_turboquant_config",
    "write_turboquant_config",
]
