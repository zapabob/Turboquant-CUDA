"""Triality Platform GGUF contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from turboquant.schema import build_paper_turboquant_config

TrialityPublicMode = Literal["paper-faithful", "triality-so8-pareto"]

TRIALITY_GGUF_SCHEMA_VERSION = 1
TRIALITY_GGUF_PAYLOAD_FORMAT = "json-inline-v1"
TRIALITY_GGUF_NAMESPACE = "hypura.turboquant"
TRIALITY_ALLOWED_MODES: tuple[TrialityPublicMode, ...] = (
    "paper-faithful",
    "triality-so8-pareto",
)
TRIALITY_REQUIRED_KEYS = (
    "hypura.turboquant.schema_version",
    "hypura.turboquant.enabled",
    "hypura.turboquant.mode",
    "hypura.turboquant.rotation_policy",
    "hypura.turboquant.rotation_seed",
    "hypura.turboquant.triality_view",
    "hypura.turboquant.triality_mix",
    "hypura.turboquant.paper_fidelity",
    "hypura.turboquant.k_bits",
    "hypura.turboquant.v_bits",
    "hypura.turboquant.payload_format",
    "hypura.turboquant.payload_bytes",
)


@dataclass(frozen=True, slots=True)
class TrialityModeSpec:
    mode: TrialityPublicMode
    runtime_mode: str
    rotation_policy: str
    rotation_seed: int
    triality_view: str
    triality_mix: float
    paper_fidelity: bool
    k_bits: float
    v_bits: float


def resolve_triality_mode_spec(mode: str) -> TrialityModeSpec:
    if mode == "paper-faithful":
        return TrialityModeSpec(
            mode="paper-faithful",
            runtime_mode="paper-key-only",
            rotation_policy="random_haar",
            rotation_seed=0,
            triality_view="none",
            triality_mix=0.0,
            paper_fidelity=True,
            k_bits=3.5,
            v_bits=16.0,
        )
    if mode == "triality-so8-pareto":
        return TrialityModeSpec(
            mode="triality-so8-pareto",
            runtime_mode="research-kv-split",
            rotation_policy="block_so8_learned",
            rotation_seed=17,
            triality_view="vector",
            triality_mix=0.75,
            paper_fidelity=False,
            k_bits=3.5,
            v_bits=8.0,
        )
    raise ValueError(
        f"Unsupported Triality mode {mode!r}; expected one of {', '.join(TRIALITY_ALLOWED_MODES)}"
    )


def build_triality_payload(
    *,
    mode: str,
    head_dim: int,
    num_layers: int,
    num_kv_heads: int,
    rotation_seed: int | None = None,
    source_manifest: dict[str, Any] | None = None,
    offline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = resolve_triality_mode_spec(mode)
    resolved_rotation_seed = spec.rotation_seed if rotation_seed is None else rotation_seed

    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")

    payload: dict[str, Any] = {
        "schema_kind": "triality_gguf_payload",
        "schema_version": TRIALITY_GGUF_SCHEMA_VERSION,
        "mode": spec.mode,
        "runtime_mode": spec.runtime_mode,
        "head_dim": int(head_dim),
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "rotation_policy": spec.rotation_policy,
        "rotation_seed": int(resolved_rotation_seed),
        "triality_view": spec.triality_view,
        "triality_mix": float(spec.triality_mix),
        "paper_fidelity": spec.paper_fidelity,
        "k_bits": float(spec.k_bits),
        "v_bits": float(spec.v_bits),
        "offline_metrics": offline_metrics or {},
    }

    if spec.paper_fidelity:
        payload["paper_config"] = build_paper_turboquant_config(
            bit_grid=[float(spec.k_bits)],
            dim=head_dim,
            rotation_seed=resolved_rotation_seed,
            qjl_seed=1,
        )
    else:
        payload["pareto_profile"] = {
            "frontier_label": "triality-so8-pareto",
            "selection_policy": "latency_quality_balanced",
            "rotation_family": "block-so8",
            "view_family": spec.triality_view,
        }

    if source_manifest is not None:
        payload["source_manifest"] = source_manifest

    validate_triality_payload(payload)
    return payload


def payload_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_triality_metadata(
    *,
    mode: str,
    payload_json: str,
    rotation_policy: str | None = None,
    rotation_seed: int | None = None,
    triality_view: str | None = None,
    triality_mix: float | None = None,
    paper_fidelity: bool | None = None,
    k_bits: float | None = None,
    v_bits: float | None = None,
    runtime_mode: str | None = None,
    source_profile: str | None = None,
) -> dict[str, Any]:
    spec = resolve_triality_mode_spec(mode)
    metadata = {
        "hypura.turboquant.schema_version": TRIALITY_GGUF_SCHEMA_VERSION,
        "hypura.turboquant.enabled": True,
        "hypura.turboquant.mode": spec.mode,
        "hypura.turboquant.rotation_policy": rotation_policy or spec.rotation_policy,
        "hypura.turboquant.rotation_seed": int(
            spec.rotation_seed if rotation_seed is None else rotation_seed
        ),
        "hypura.turboquant.triality_view": triality_view or spec.triality_view,
        "hypura.turboquant.triality_mix": float(
            spec.triality_mix if triality_mix is None else triality_mix
        ),
        "hypura.turboquant.paper_fidelity": spec.paper_fidelity
        if paper_fidelity is None
        else bool(paper_fidelity),
        "hypura.turboquant.k_bits": float(spec.k_bits if k_bits is None else k_bits),
        "hypura.turboquant.v_bits": float(spec.v_bits if v_bits is None else v_bits),
        "hypura.turboquant.payload_format": TRIALITY_GGUF_PAYLOAD_FORMAT,
        "hypura.turboquant.payload_bytes": len(payload_json.encode("utf-8")),
        "hypura.turboquant.payload_json": payload_json,
        "hypura.turboquant.runtime_mode": runtime_mode or spec.runtime_mode,
    }
    if source_profile:
        metadata["hypura.turboquant.source_profile"] = source_profile

    validate_triality_metadata(metadata)
    return metadata


def validate_triality_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_kind") != "triality_gguf_payload":
        raise ValueError("payload schema_kind must be 'triality_gguf_payload'")
    if int(payload.get("schema_version", 0)) != TRIALITY_GGUF_SCHEMA_VERSION:
        raise ValueError(
            f"payload schema_version must be {TRIALITY_GGUF_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    resolve_triality_mode_spec(str(payload.get("mode")))


def validate_triality_metadata(metadata: dict[str, Any]) -> None:
    missing = [key for key in TRIALITY_REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Missing Triality metadata keys: {', '.join(missing)}")

    mode = metadata["hypura.turboquant.mode"]
    spec = resolve_triality_mode_spec(str(mode))

    schema_version = int(metadata["hypura.turboquant.schema_version"])
    if schema_version != TRIALITY_GGUF_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Triality schema_version {schema_version}; expected {TRIALITY_GGUF_SCHEMA_VERSION}"
        )

    payload_format = str(metadata["hypura.turboquant.payload_format"])
    if payload_format != TRIALITY_GGUF_PAYLOAD_FORMAT:
        raise ValueError(
            f"Unsupported Triality payload_format {payload_format!r}; expected {TRIALITY_GGUF_PAYLOAD_FORMAT!r}"
        )

    payload_json = metadata.get("hypura.turboquant.payload_json")
    payload_bytes = int(metadata["hypura.turboquant.payload_bytes"])
    if payload_json is not None:
        actual = len(str(payload_json).encode("utf-8"))
        if actual != payload_bytes:
            raise ValueError(
                "hypura.turboquant.payload_bytes does not match payload_json length: "
                f"{payload_bytes} != {actual}"
            )

    paper_fidelity = bool(metadata["hypura.turboquant.paper_fidelity"])
    if paper_fidelity != spec.paper_fidelity:
        raise ValueError(
            f"Mode {mode!r} expects paper_fidelity={spec.paper_fidelity}, got {paper_fidelity}"
        )


__all__ = [
    "TRIALITY_ALLOWED_MODES",
    "TRIALITY_GGUF_NAMESPACE",
    "TRIALITY_GGUF_PAYLOAD_FORMAT",
    "TRIALITY_GGUF_SCHEMA_VERSION",
    "TRIALITY_REQUIRED_KEYS",
    "TrialityModeSpec",
    "build_triality_metadata",
    "build_triality_payload",
    "payload_json_dumps",
    "resolve_triality_mode_spec",
    "validate_triality_metadata",
    "validate_triality_payload",
]
