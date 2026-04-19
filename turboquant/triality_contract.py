"""Legacy Hypura bridge compatibility helpers for Triality Platform GGUF exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from turboquant.schema import build_paper_turboquant_config

TRIALITY_PROXY_PARETO_MODE = "triality-proxy-so8-pareto"
TRIALITY_PROXY_PARETO_LEGACY_ALIAS = "triality-so8-pareto"
TRIALITY_RUNTIME_MODE = "key_only_block_so8_triality_vector"
TRIALITY_RUNTIME_MODE_ALIASES = (
    TRIALITY_RUNTIME_MODE,
    "triality_vector",
    "triality-vector",
    "research-kv-split",
)
TRIALITY_VIEW_ALIASES = {
    "vector": "vector",
    "spinor_plus_proxy": "spinor_plus_proxy",
    "plus": "spinor_plus_proxy",
    "spinor_minus_proxy": "spinor_minus_proxy",
    "minus": "spinor_minus_proxy",
}
TRIALITY_ROTATION_BLOCK_SIZE = 8

TrialityPublicMode = Literal["paper-faithful", "triality-proxy-so8-pareto", "triality-so8-pareto"]

TRIALITY_GGUF_SCHEMA_VERSION = 1
TRIALITY_GGUF_PAYLOAD_FORMAT = "json-inline-v1"
TRIALITY_GGUF_NAMESPACE = "hypura.turboquant"
TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES = ("bf16", "f16", "q8_0")
TRIALITY_WEIGHT_ALLOWED_TENSOR_CODECS = ("tq4_1s", "q4_k", "q8_0")
TRIALITY_ALLOWED_MODES: tuple[TrialityPublicMode, ...] = (
    "paper-faithful",
    TRIALITY_PROXY_PARETO_MODE,
    TRIALITY_PROXY_PARETO_LEGACY_ALIAS,
)
TRIALITY_REQUIRED_KEYS = (
    "hypura.turboquant.schema_version",
    "hypura.turboquant.enabled",
    "hypura.turboquant.mode",
    "hypura.turboquant.codec",
    "hypura.turboquant.rotation_policy",
    "hypura.turboquant.rotation_block_size",
    "hypura.turboquant.rotation_seed",
    "hypura.turboquant.triality_view",
    "hypura.turboquant.triality_mix",
    "hypura.turboquant.view_bundle_complete",
    "hypura.turboquant.orthogonality_error",
    "hypura.turboquant.determinant_error_max",
    "hypura.turboquant.paper_fidelity",
    "hypura.turboquant.k_bits",
    "hypura.turboquant.v_bits",
    "hypura.turboquant.payload_format",
    "hypura.turboquant.payload_bytes",
    "hypura.turboquant.weight.enabled",
    "hypura.turboquant.weight.codec",
    "hypura.turboquant.weight.source_ftype",
    "hypura.turboquant.weight.policy",
    "hypura.turboquant.weight.protected_roles",
    "hypura.turboquant.weight.protected_layers",
    "hypura.turboquant.weight.modality_scope",
    "hypura.turboquant.weight.payload_format",
    "hypura.turboquant.weight.payload_bytes",
)

TRIALITY_FIXTURE_MANIFEST_VERSION = 2


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


def normalize_triality_view(view: str) -> str:
    normalized_view = view.strip().lower().replace("-", "_")
    try:
        return TRIALITY_VIEW_ALIASES[normalized_view]
    except KeyError as exc:
        supported = ", ".join(sorted(TRIALITY_VIEW_ALIASES))
        raise ValueError(f"Unsupported triality_view {view!r}; expected one of {supported}") from exc


def normalize_triality_runtime_mode(runtime_mode: str) -> str:
    normalized_mode = runtime_mode.strip().lower().replace("-", "_")
    alias_map = {alias.replace("-", "_"): TRIALITY_RUNTIME_MODE for alias in TRIALITY_RUNTIME_MODE_ALIASES}
    return alias_map.get(normalized_mode, runtime_mode)


def normalize_model_family(model_family: str) -> str:
    return model_family.strip().lower().replace("\\", "/")


def _is_qwen35_family(model_family: str) -> bool:
    normalized_family = normalize_model_family(model_family)
    return "qwen" in normalized_family and "3.5" in normalized_family


def _is_gemma4_family(model_family: str) -> bool:
    normalized_family = normalize_model_family(model_family)
    return "gemma-4" in normalized_family or ("gemma" in normalized_family and " 4" in normalized_family)


def _is_gemma4_multimodal_family(model_family: str) -> bool:
    if not _is_gemma4_family(model_family):
        return False
    normalized_family = normalize_model_family(model_family)
    return any(tag in normalized_family for tag in ("-e2b", "-e4b", "-a4b"))


def _default_tq4_1s_tensor_plan() -> dict[str, str]:
    return {
        "blk.*.attn_q.weight": "tq4_1s",
        "blk.*.attn_k.weight": "tq4_1s",
        "blk.*.attn_v.weight": "tq4_1s",
        "blk.*.attn_output.weight": "tq4_1s",
        "blk.*.ffn_gate.weight": "tq4_1s",
        "blk.*.ffn_up.weight": "tq4_1s",
        "blk.*.ffn_down.weight": "q4_k",
    }


def expected_modalities(
    *,
    model_family: str,
    modality_scope: str | None = None,
) -> list[str]:
    normalized_family = normalize_model_family(model_family)
    resolved_scope = (modality_scope or "").strip().lower()
    if resolved_scope == "full-multimodal" or _is_gemma4_multimodal_family(normalized_family):
        return ["text", "image", "audio"]
    return ["text"]


def build_sample_env(
    *,
    model_family: str,
    modality_scope: str | None = None,
) -> dict[str, str]:
    modalities = expected_modalities(
        model_family=model_family,
        modality_scope=modality_scope,
    )
    normalized_family = normalize_model_family(model_family)
    if modalities == ["text"] and "qwen" in normalized_family:
        return {
            "text_model": "TRIALITY_QWEN_SMOKE_MODEL",
        }
    if modalities != ["text"] and "gemma" in normalized_family:
        return {
            "text_model": "TRIALITY_GEMMA_SMOKE_MODEL",
            "mmproj_model": "TRIALITY_GEMMA_MMPROJ_MODEL",
            "image": "TRIALITY_GEMMA_IMAGE_SAMPLE",
            "audio": "TRIALITY_GEMMA_AUDIO_SAMPLE",
        }
    return {
        "text_model": "TRIALITY_SMOKE_MODEL",
    }


def build_triality_fixture_manifest(
    *,
    mode: str,
    model_family: str,
    source_ftype: str,
    generated_at_utc: str,
    payload_path: str,
    metadata_path: str,
    metrics_path: str,
    text_model_path: str,
    payload_hash: str,
    metadata_hash: str,
    modality_scope: str | None = None,
    mmproj_model_path: str | None = None,
) -> dict[str, Any]:
    modalities = expected_modalities(
        model_family=model_family,
        modality_scope=modality_scope,
    )
    mmproj_required = len(modalities) > 1
    if mmproj_required and not mmproj_model_path:
        raise ValueError(
            "mmproj_model_path is required for full-multimodal Triality fixture manifests"
        )

    paths: dict[str, str | None] = {
        "payload": payload_path,
        "metadata": metadata_path,
        "offline_metrics": metrics_path,
        "gguf": text_model_path,
        "text_model": text_model_path,
        "mmproj_model": mmproj_model_path,
    }

    return {
        "schema_version": TRIALITY_FIXTURE_MANIFEST_VERSION,
        "fixture_kind": "triality-fixture-bundle",
        "mode": resolve_triality_mode_spec(mode).mode,
        "model_family": model_family,
        "source_ftype": source_ftype,
        "generated_at_utc": generated_at_utc,
        "text_model_path": text_model_path,
        "mmproj_model_path": mmproj_model_path,
        "mmproj_required": mmproj_required,
        "modalities": modalities,
        "sample_env": build_sample_env(
            model_family=model_family,
            modality_scope=modality_scope,
        ),
        "paths": paths,
        "hashes": {
            "payload_sha256": payload_hash,
            "metadata_sha256": metadata_hash,
        },
    }


def _boundary_layers(num_layers: int) -> list[int]:
    if num_layers <= 0:
        return []
    layers = {0, min(1, num_layers - 1), max(0, num_layers - 2), num_layers - 1}
    return sorted(layers)


def build_default_weight_plan(
    *,
    model_family: str,
    num_layers: int,
    source_ftype: str = "q8_0",
    policy: str | None = None,
    protected_roles: list[str] | None = None,
    protected_layers: list[int] | None = None,
    modality_scope: str | None = None,
) -> dict[str, Any]:
    normalized_family = normalize_model_family(model_family)
    normalized_source_ftype = source_ftype.strip().lower()
    if normalized_source_ftype not in TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES:
        raise ValueError(
            "Unsupported weight source_ftype "
            f"{source_ftype!r}; expected one of {', '.join(TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES)}"
        )

    if _is_qwen35_family(normalized_family):
        resolved_policy = policy or "qwen35-config-i"
        resolved_roles = protected_roles or [
            "embedding",
            "norm",
            "output_head",
            "recurrent_state",
        ]
        resolved_modality_scope = modality_scope or "text-only"
    elif _is_gemma4_family(normalized_family):
        resolved_policy = policy or "gemma4-kv-first-multimodal-safe"
        resolved_roles = protected_roles or [
            "vision_encoder",
            "audio_encoder",
            "projector",
            "per_layer_multimodal_embedding",
            "embedding",
            "norm",
            "output_head",
        ]
        resolved_modality_scope = modality_scope or (
            "full-multimodal" if _is_gemma4_multimodal_family(normalized_family) else "text-only"
        )
    else:
        resolved_policy = policy or "shared-decoder-role-aware"
        resolved_roles = protected_roles or [
            "embedding",
            "norm",
            "output_head",
        ]
        resolved_modality_scope = modality_scope or "text-only"

    resolved_layers = protected_layers or _boundary_layers(num_layers)
    return {
        "enabled": True,
        "schema": "hypura.turboquant.weight.v1",
        "codec": "tq4_1s",
        "model_family": model_family,
        "source_ftype": normalized_source_ftype,
        "policy": resolved_policy,
        "protected_roles": list(resolved_roles),
        "protected_layers": [int(layer) for layer in resolved_layers],
        "modality_scope": resolved_modality_scope,
        "tensor_plan": _default_tq4_1s_tensor_plan(),
    }


def validate_weight_plan(
    weight_plan: dict[str, Any],
    *,
    model_family: str,
    num_layers: int,
) -> None:
    expected_weight_plan = build_default_weight_plan(
        model_family=model_family,
        num_layers=num_layers,
        source_ftype=str(weight_plan.get("source_ftype", "q8_0")),
        policy=str(weight_plan.get("policy")) if weight_plan.get("policy") is not None else None,
        protected_roles=list(weight_plan.get("protected_roles", [])),
        protected_layers=[int(v) for v in weight_plan.get("protected_layers", [])],
        modality_scope=str(weight_plan.get("modality_scope")) if weight_plan.get("modality_scope") is not None else None,
    )
    if str(weight_plan.get("schema")) != "hypura.turboquant.weight.v1":
        raise ValueError("weight_plan.schema must be 'hypura.turboquant.weight.v1'")
    if str(weight_plan.get("codec")) != "tq4_1s":
        raise ValueError("weight_plan.codec must be 'tq4_1s'")
    tensor_plan = weight_plan.get("tensor_plan")
    if not isinstance(tensor_plan, dict) or not tensor_plan:
        raise ValueError("weight_plan.tensor_plan must be a non-empty object")
    for tensor_name, tensor_codec in tensor_plan.items():
        if not isinstance(tensor_name, str) or not tensor_name.strip():
            raise ValueError("weight_plan.tensor_plan keys must be non-empty strings")
        normalized_codec = str(tensor_codec).strip().lower()
        if normalized_codec not in TRIALITY_WEIGHT_ALLOWED_TENSOR_CODECS:
            raise ValueError(
                "Unsupported weight_plan.tensor_plan codec "
                f"{tensor_codec!r}; expected one of {', '.join(TRIALITY_WEIGHT_ALLOWED_TENSOR_CODECS)}"
            )
    if "tq4_1s" not in {str(codec).strip().lower() for codec in tensor_plan.values()}:
        raise ValueError("weight_plan.tensor_plan must include at least one tq4_1s target")
    default_tensor_plan = expected_weight_plan["tensor_plan"]
    unknown_tensor_names = set(tensor_plan) - set(default_tensor_plan)
    if unknown_tensor_names:
        raise ValueError(
            "weight_plan.tensor_plan contains unsupported tensor selectors: "
            + ", ".join(sorted(str(name) for name in unknown_tensor_names))
        )


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
    if mode in {TRIALITY_PROXY_PARETO_MODE, TRIALITY_PROXY_PARETO_LEGACY_ALIAS}:
        return TrialityModeSpec(
            mode=TRIALITY_PROXY_PARETO_MODE,
            runtime_mode=TRIALITY_RUNTIME_MODE,
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
    model_family: str = "generic",
    weight_source_ftype: str = "q8_0",
    weight_policy: str | None = None,
    weight_protected_roles: list[str] | None = None,
    weight_protected_layers: list[int] | None = None,
    modality_scope: str | None = None,
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
        "codec": "tq4_1s",
        "mode": spec.mode,
        "model_family": model_family,
        "runtime_mode": normalize_triality_runtime_mode(spec.runtime_mode),
        "head_dim": int(head_dim),
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "rotation_policy": spec.rotation_policy,
        "rotation_block_size": TRIALITY_ROTATION_BLOCK_SIZE,
        "rotation_seed": int(resolved_rotation_seed),
        "triality_view": normalize_triality_view(spec.triality_view),
        "triality_mix": float(spec.triality_mix),
        "view_bundle_complete": spec.triality_view == "none",
        "orthogonality_error": float(offline_metrics.get("orthogonality_error", 0.0) if offline_metrics else 0.0),
        "determinant_error_max": float(offline_metrics.get("determinant_error_max", 0.0) if offline_metrics else 0.0),
        "paper_fidelity": spec.paper_fidelity,
        "k_bits": float(spec.k_bits),
        "v_bits": float(spec.v_bits),
        "offline_metrics": offline_metrics or {},
        "weight_plan": build_default_weight_plan(
            model_family=model_family,
            num_layers=num_layers,
            source_ftype=weight_source_ftype,
            policy=weight_policy,
            protected_roles=weight_protected_roles,
            protected_layers=weight_protected_layers,
            modality_scope=modality_scope,
        ),
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
            "frontier_label": TRIALITY_PROXY_PARETO_MODE,
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
    weight_plan: dict[str, Any] | None = None,
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
        "hypura.turboquant.codec": "tq4_1s",
        "hypura.turboquant.rotation_policy": rotation_policy or spec.rotation_policy,
        "hypura.turboquant.rotation_block_size": TRIALITY_ROTATION_BLOCK_SIZE,
        "hypura.turboquant.rotation_seed": int(
            spec.rotation_seed if rotation_seed is None else rotation_seed
        ),
        "hypura.turboquant.triality_view": normalize_triality_view(triality_view or spec.triality_view),
        "hypura.turboquant.triality_mix": float(
            spec.triality_mix if triality_mix is None else triality_mix
        ),
        "hypura.turboquant.view_bundle_complete": spec.triality_view == "none"
        or normalize_triality_view(triality_view or spec.triality_view) == "vector",
        "hypura.turboquant.orthogonality_error": 0.0,
        "hypura.turboquant.determinant_error_max": 0.0,
        "hypura.turboquant.paper_fidelity": spec.paper_fidelity
        if paper_fidelity is None
        else bool(paper_fidelity),
        "hypura.turboquant.k_bits": float(spec.k_bits if k_bits is None else k_bits),
        "hypura.turboquant.v_bits": float(spec.v_bits if v_bits is None else v_bits),
        "hypura.turboquant.payload_format": TRIALITY_GGUF_PAYLOAD_FORMAT,
        "hypura.turboquant.payload_bytes": len(payload_json.encode("utf-8")),
        "hypura.turboquant.payload_json": payload_json,
        "hypura.turboquant.runtime_mode": normalize_triality_runtime_mode(runtime_mode or spec.runtime_mode),
    }
    if source_profile:
        metadata["hypura.turboquant.source_profile"] = source_profile

    if weight_plan is not None:
        weight_payload_json = payload_json_dumps(weight_plan)
        metadata.update(
            {
                "hypura.turboquant.weight.enabled": bool(weight_plan.get("enabled", True)),
                "hypura.turboquant.weight.codec": str(weight_plan["codec"]),
                "hypura.turboquant.weight.source_ftype": str(weight_plan["source_ftype"]),
                "hypura.turboquant.weight.policy": str(weight_plan["policy"]),
                "hypura.turboquant.weight.protected_roles": json.dumps(
                    list(weight_plan.get("protected_roles", [])),
                    separators=(",", ":"),
                ),
                "hypura.turboquant.weight.protected_layers": json.dumps(
                    [int(v) for v in weight_plan.get("protected_layers", [])],
                    separators=(",", ":"),
                ),
                "hypura.turboquant.weight.modality_scope": str(
                    weight_plan.get("modality_scope", "text-only")
                ),
                "hypura.turboquant.weight.payload_format": TRIALITY_GGUF_PAYLOAD_FORMAT,
                "hypura.turboquant.weight.payload_bytes": len(
                    weight_payload_json.encode("utf-8")
                ),
                "hypura.turboquant.weight.payload_json": weight_payload_json,
            }
        )

    validate_triality_metadata(metadata)
    return metadata


def validate_triality_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_kind") != "triality_gguf_payload":
        raise ValueError("payload schema_kind must be 'triality_gguf_payload'")
    if int(payload.get("schema_version", 0)) != TRIALITY_GGUF_SCHEMA_VERSION:
        raise ValueError(
            f"payload schema_version must be {TRIALITY_GGUF_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    if str(payload.get("codec", "")).strip().lower() != "tq4_1s":
        raise ValueError("payload codec must be 'tq4_1s'")
    resolve_triality_mode_spec(str(payload.get("mode")))
    if int(payload.get("rotation_block_size", 0)) != TRIALITY_ROTATION_BLOCK_SIZE:
        raise ValueError(
            f"payload rotation_block_size must be {TRIALITY_ROTATION_BLOCK_SIZE}"
        )
    normalize_triality_view(str(payload.get("triality_view", "none")))
    runtime_mode = normalize_triality_runtime_mode(str(payload.get("runtime_mode", "")))
    if runtime_mode not in {"paper-key-only", TRIALITY_RUNTIME_MODE}:
        raise ValueError(f"Unsupported payload runtime_mode {payload.get('runtime_mode')!r}")
    orthogonality_error = float(payload.get("orthogonality_error", 0.0))
    determinant_error_max = float(payload.get("determinant_error_max", 0.0))
    if orthogonality_error < 0.0 or determinant_error_max < 0.0:
        raise ValueError("payload orthogonality and determinant errors must be non-negative")
    weight_plan = payload.get("weight_plan")
    if not isinstance(weight_plan, dict):
        raise ValueError("payload must include a weight_plan object")
    validate_weight_plan(
        weight_plan,
        model_family=str(weight_plan.get("model_family", payload.get("model_family", "generic"))),
        num_layers=int(payload.get("num_layers", 0)),
    )


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

    codec = str(metadata["hypura.turboquant.codec"]).strip().lower()
    if codec != "tq4_1s":
        raise ValueError(f"Unsupported hypura.turboquant.codec {codec!r}; expected 'tq4_1s'")

    rotation_block_size = int(metadata["hypura.turboquant.rotation_block_size"])
    if rotation_block_size != TRIALITY_ROTATION_BLOCK_SIZE:
        raise ValueError(
            f"Unsupported hypura.turboquant.rotation_block_size {rotation_block_size}; "
            f"expected {TRIALITY_ROTATION_BLOCK_SIZE}"
        )

    payload_format = str(metadata["hypura.turboquant.payload_format"])
    if payload_format != TRIALITY_GGUF_PAYLOAD_FORMAT:
        raise ValueError(
            f"Unsupported Triality payload_format {payload_format!r}; expected {TRIALITY_GGUF_PAYLOAD_FORMAT!r}"
        )

    normalize_triality_view(str(metadata["hypura.turboquant.triality_view"]))
    runtime_mode = normalize_triality_runtime_mode(str(metadata.get("hypura.turboquant.runtime_mode", spec.runtime_mode)))
    if runtime_mode not in {"paper-key-only", TRIALITY_RUNTIME_MODE}:
        raise ValueError(f"Unsupported hypura.turboquant.runtime_mode {runtime_mode!r}")

    orthogonality_error = float(metadata["hypura.turboquant.orthogonality_error"])
    determinant_error_max = float(metadata["hypura.turboquant.determinant_error_max"])
    if orthogonality_error < 0.0 or determinant_error_max < 0.0:
        raise ValueError("hypura.turboquant orthogonality/determinant errors must be non-negative")

    payload_json = metadata.get("hypura.turboquant.payload_json")
    payload_bytes = int(metadata["hypura.turboquant.payload_bytes"])
    if payload_json is not None:
        actual = len(str(payload_json).encode("utf-8"))
        if actual != payload_bytes:
            raise ValueError(
                "hypura.turboquant.payload_bytes does not match payload_json length: "
                f"{payload_bytes} != {actual}"
            )

    weight_enabled = bool(metadata["hypura.turboquant.weight.enabled"])
    weight_codec = str(metadata["hypura.turboquant.weight.codec"]).strip().lower()
    if weight_enabled and weight_codec != "tq4_1s":
        raise ValueError(
            f"Unsupported hypura.turboquant.weight.codec {weight_codec!r}; expected 'tq4_1s'"
        )
    weight_source_ftype = str(metadata["hypura.turboquant.weight.source_ftype"]).strip().lower()
    if weight_enabled and weight_source_ftype not in TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES:
        raise ValueError(
            "Unsupported hypura.turboquant.weight.source_ftype "
            f"{weight_source_ftype!r}; expected one of {', '.join(TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES)}"
        )

    for key in (
        "hypura.turboquant.weight.protected_roles",
        "hypura.turboquant.weight.protected_layers",
    ):
        try:
            parsed = json.loads(str(metadata[key]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be JSON-encoded") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{key} must decode to a list")

    weight_payload_json = metadata.get("hypura.turboquant.weight.payload_json")
    weight_payload_bytes = int(metadata["hypura.turboquant.weight.payload_bytes"])
    if weight_payload_json is not None:
        actual = len(str(weight_payload_json).encode("utf-8"))
        if actual != weight_payload_bytes:
            raise ValueError(
                "hypura.turboquant.weight.payload_bytes does not match payload_json length: "
                f"{weight_payload_bytes} != {actual}"
            )
        try:
            parsed_weight_payload = json.loads(str(weight_payload_json))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "hypura.turboquant.weight.payload_json must be valid JSON"
            ) from exc
        if not isinstance(parsed_weight_payload, dict):
            raise ValueError("hypura.turboquant.weight.payload_json must decode to an object")
        if parsed_weight_payload.get("codec") != weight_codec:
            raise ValueError(
                "hypura.turboquant.weight.payload_json codec does not match hypura.turboquant.weight.codec"
            )
        validate_weight_plan(
            parsed_weight_payload,
            model_family=str(parsed_weight_payload.get("model_family", "generic")),
            num_layers=max(
                [int(v) for v in parsed_weight_payload.get("protected_layers", [])] + [0]
            )
            + 1,
        )

    paper_fidelity = bool(metadata["hypura.turboquant.paper_fidelity"])
    if paper_fidelity != spec.paper_fidelity:
        raise ValueError(
            f"Mode {mode!r} expects paper_fidelity={spec.paper_fidelity}, got {paper_fidelity}"
        )


__all__ = [
    "TRIALITY_ALLOWED_MODES",
    "TRIALITY_FIXTURE_MANIFEST_VERSION",
    "TRIALITY_GGUF_NAMESPACE",
    "TRIALITY_GGUF_PAYLOAD_FORMAT",
    "TRIALITY_GGUF_SCHEMA_VERSION",
    "TRIALITY_PROXY_PARETO_LEGACY_ALIAS",
    "TRIALITY_PROXY_PARETO_MODE",
    "TRIALITY_ROTATION_BLOCK_SIZE",
    "TRIALITY_RUNTIME_MODE",
    "TRIALITY_RUNTIME_MODE_ALIASES",
    "TRIALITY_REQUIRED_KEYS",
    "TrialityModeSpec",
    "build_sample_env",
    "build_triality_fixture_manifest",
    "build_triality_metadata",
    "build_triality_payload",
    "build_default_weight_plan",
    "expected_modalities",
    "normalize_model_family",
    "normalize_triality_runtime_mode",
    "normalize_triality_view",
    "payload_json_dumps",
    "resolve_triality_mode_spec",
    "validate_triality_metadata",
    "validate_triality_payload",
    "TRIALITY_WEIGHT_ALLOWED_SOURCE_FTYPES",
]
