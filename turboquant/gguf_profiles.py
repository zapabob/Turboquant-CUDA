"""Embed TurboQuant runtime profiles into a single GGUF container."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, TypeAlias

import numpy as np

from turboquant.paper_baseline import PaperMixedBitPolicy
from turboquant.research_extension.k_triality import (
    PRODUCTION_K_TURBOQUANT_MODE,
    PRODUCTION_K_TURBOQUANT_VIEW,
    load_triality_proxy_rotations,
)
from turboquant.schema import (
    TURBOQUANT_GGUF_FLOAT_KEYS,
    TURBOQUANT_GGUF_STRING_KEYS,
    TURBOQUANT_GGUF_U32_KEYS,
    build_paper_turboquant_config,
    build_turboquant_gguf_contract,
    validate_turboquant_gguf_contract,
)
from turboquant.triality_contract import (
    TRIALITY_PROXY_PARETO_LEGACY_ALIAS,
    TRIALITY_PROXY_PARETO_MODE,
    TRIALITY_RUNTIME_MODE,
    TRIALITY_RUNTIME_MODE_ALIASES,
    build_triality_metadata,
    build_triality_payload,
    normalize_triality_runtime_mode,
    normalize_triality_view,
    payload_json_dumps,
    validate_triality_metadata,
)


GGUF_TURBOQUANT_SCHEMA_VERSION = 1
GGUF_TURBOQUANT_NAMESPACE = "turboquant"
GGUF_TURBOQUANT_EXACT_PROFILE = "exact"
GGUF_HYPURA_COMPAT_AUTO = "auto"
GGUF_HYPURA_COMPAT_OFF = "off"
HYPURA_TURBOQUANT_NAMESPACE = "hypura.turboquant"
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")
_TENSOR_NAME_PREFIX = f"{GGUF_TURBOQUANT_NAMESPACE}.profile."

GGUFScalar: TypeAlias = str | int | float | bool
GGUFArrayValue: TypeAlias = list[str] | list[int] | list[float] | list[bool]
GGUFMetadataValue: TypeAlias = GGUFScalar | GGUFArrayValue


@dataclass(slots=True)
class TurboQuantEmbeddedTensor:
    """Additional tensor payload stored inside the output GGUF.

    ``data`` is expected to be a CPU-side ndarray. For learned triality
    rotations this is a shape comment of ``[head_dim, head_dim]`` in `float32`.
    """

    name: str
    data: np.ndarray


@dataclass(slots=True)
class TurboQuantGGUFProfile:
    """Serializable TurboQuant runtime profile for a GGUF bundle."""

    name: str
    kind: str
    runtime_mode: str
    description: str
    manifest: dict[str, Any]
    metadata: dict[str, GGUFMetadataValue] = field(default_factory=dict)
    tensors: list[TurboQuantEmbeddedTensor] = field(default_factory=list)


@dataclass(slots=True)
class TurboQuantGGUFManifest:
    """Top-level embedded TurboQuant manifest read back from a GGUF file."""

    schema_version: int
    default_profile: str
    packaged_at_utc: str
    base_architecture: str
    profiles: dict[str, TurboQuantGGUFProfile]


@dataclass(slots=True)
class HypuraTurboQuantBridgeConfig:
    """Hypura/Kobold-compatible TurboQuant bridge metadata embedded in GGUF."""

    source_profile: str
    mode: str
    rotation_policy: str | None = None
    triality_view: str | None = None
    triality_mix: float | None = None
    rotation_seed: int = 0
    artifact_path: str | None = None


def import_vendor_gguf() -> ModuleType:
    """Import the vendored `gguf-py` package shipped with the zapabob llama.cpp checkout."""

    module_path = Path(__file__).resolve()
    candidate_dirs: list[Path] = []
    for env_key in ("LLAMA_CPP_DIR", "HYPURA_LLAMA_CPP_DIR"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidate_dirs.append(Path(env_value) / "gguf-py")
    candidate_dirs.extend(
        [
            module_path.parents[1] / "zapabob" / "llama.cpp" / "gguf-py",
            module_path.parents[1] / "vendor" / "llama.cpp" / "gguf-py",
            module_path.parents[2] / "llama.cpp" / "gguf-py",
            module_path.parents[3] / "zapabob" / "llama.cpp" / "gguf-py",
            module_path.parents[3] / "vendor" / "llama.cpp" / "gguf-py",
        ]
    )

    gguf_python_dir = next((path for path in candidate_dirs if path.exists()), None)
    if gguf_python_dir is None:
        searched = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(f"Vendored gguf-py directory is missing; searched: {searched}")

    gguf_python_dir_str = str(gguf_python_dir)
    if gguf_python_dir_str not in sys.path:
        sys.path.insert(0, gguf_python_dir_str)
    return importlib.import_module("gguf")


def infer_gguf_attention_head_dim(path: Path) -> int:
    """Infer the attention head dimension from an existing GGUF file."""

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(path)
    arch = _require_source_architecture(reader)
    key_length_field = reader.get_field(f"{arch}.attention.key_length")
    if key_length_field is not None:
        head_dim = int(key_length_field.contents())
        if head_dim <= 0:
            raise ValueError(f"Invalid attention.key_length in {path}: {head_dim}")
        return head_dim

    embedding_field = reader.get_field(f"{arch}.embedding_length")
    head_count_field = reader.get_field(f"{arch}.attention.head_count")
    if embedding_field is None or head_count_field is None:
        raise ValueError(
            "Unable to infer head dimension from GGUF metadata: "
            f"missing {arch}.attention.key_length and fallback embedding/head count fields"
        )
    embedding_length = int(embedding_field.contents())
    head_count = int(head_count_field.contents())
    if head_count <= 0:
        raise ValueError(f"Invalid attention.head_count in {path}: {head_count}")
    if embedding_length % head_count != 0:
        raise ValueError(
            "embedding_length must be divisible by attention.head_count to infer head_dim; "
            f"got embedding_length={embedding_length}, head_count={head_count}"
        )
    return embedding_length // head_count


def infer_gguf_block_count(path: Path) -> int:
    """Infer the decoder block count from an existing GGUF file."""

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(path)
    arch = _require_source_architecture(reader)
    block_count_field = reader.get_field(f"{arch}.block_count")
    if block_count_field is None:
        raise ValueError(f"Missing {arch}.block_count in GGUF metadata: {path}")
    block_count = int(block_count_field.contents())
    if block_count <= 0:
        raise ValueError(f"Invalid block_count in {path}: {block_count}")
    return block_count


def build_paper_gguf_profile(
    *,
    bits_total: float,
    head_dim: int,
    rotation_seed: int = 0,
    qjl_seed: int = 1,
    name: str = "paper",
) -> TurboQuantGGUFProfile:
    """Build a paper-faithful random-Haar key-only profile manifest."""

    _validate_profile_name(name)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if bits_total <= 0:
        raise ValueError(f"bits_total must be positive, got {bits_total}")

    mixed_policy = PaperMixedBitPolicy.for_total_bits(total_bits=bits_total, dim=head_dim)
    manifest = {
        "schema_kind": "gguf_embedded_profile",
        "schema_version": GGUF_TURBOQUANT_SCHEMA_VERSION,
        "profile_name": name,
        "profile_kind": "paper_faithful",
        "runtime_mode": "key_only_random",
        "bits_total": float(bits_total),
        "head_dim": int(head_dim),
        "rotation_policy": "random_haar",
        "rotation_seed": int(rotation_seed),
        "qjl_bits": 1,
        "qjl_seed": int(qjl_seed),
        "paper_config": build_paper_turboquant_config(
            bit_grid=[float(bits_total)],
            dim=head_dim,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
        ),
    }
    return TurboQuantGGUFProfile(
        name=name,
        kind="paper_faithful",
        runtime_mode="key_only_random",
        description="Paper-faithful TurboQuant Stage 1+2 baseline with random Haar rotation on the K side.",
        manifest=manifest,
        metadata={
            "bits_total": float(bits_total),
            "head_dim": int(head_dim),
            "rotation_policy": "random_haar",
            "rotation_seed": int(rotation_seed),
            "qjl_bits": 1,
            "qjl_seed": int(qjl_seed),
            "mixed_low_bits": int(mixed_policy.low_bits),
            "mixed_high_bits": int(mixed_policy.high_bits),
            "mixed_high_count": int(mixed_policy.high_count),
        },
    )


def build_so8_triality_vector_gguf_profile(
    *,
    rotation_dir: Path,
    bits_total: float,
    expected_head_dim: int | None = None,
    expected_block_count: int | None = None,
    name: str = "so8_triality_vector",
) -> TurboQuantGGUFProfile:
    """Build the production triality-vector profile from learned rotation artifacts."""

    _validate_profile_name(name)
    if bits_total <= 0:
        raise ValueError(f"bits_total must be positive, got {bits_total}")

    artifacts = load_triality_proxy_rotations(rotation_dir)
    selected = []
    for (layer_idx, bit_value, view), artifact in sorted(artifacts.items()):
        if view != PRODUCTION_K_TURBOQUANT_VIEW:
            continue
        if abs(bit_value - float(bits_total)) > 1e-6:
            continue
        selected.append((layer_idx, artifact))
    if not selected:
        raise FileNotFoundError(
            "No triality-vector rotation artifacts match the requested bit setting "
            f"{bits_total:g} under {rotation_dir}"
        )

    layer_indices = [layer_idx for layer_idx, _artifact in selected]
    if expected_block_count is not None and layer_indices:
        max_layer = max(layer_indices)
        if max_layer >= expected_block_count:
            raise ValueError(
                "Embedded triality rotations exceed GGUF block count: "
                f"max layer {max_layer}, block_count {expected_block_count}"
            )
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError(
            "Strict GGUF `tq_*` export requires contiguous layer coverage starting at 0; "
            f"got layer indices {layer_indices}"
        )

    common_metadata = selected[0][1].metadata
    head_dim = int(selected[0][1].rotation.shape[-1])
    if expected_head_dim is not None and head_dim != expected_head_dim:
        raise ValueError(
            "Triality rotation head_dim does not match GGUF head_dim: "
            f"{head_dim} != {expected_head_dim}"
        )
    for layer_idx, artifact in selected:
        rotation = artifact.rotation
        # rotation shape: [head_dim, head_dim]
        if tuple(rotation.shape) != (head_dim, head_dim):
            raise ValueError(
                "Each embedded triality rotation must be square [head_dim, head_dim]; "
                f"layer {layer_idx} has shape {tuple(rotation.shape)}"
            )

    tensors: list[TurboQuantEmbeddedTensor] = []
    tensor_names: list[str] = []
    for layer_idx, artifact in selected:
        tensor_name = _triality_rotation_tensor_name(profile_name=name, layer_idx=layer_idx, bits_total=bits_total)
        tensor_names.append(tensor_name)
        tensors.append(
            TurboQuantEmbeddedTensor(
                name=tensor_name,
                data=artifact.rotation.detach().cpu().to(dtype=artifact.rotation.dtype).numpy().astype(np.float32, copy=False),
            )
        )
    strict_gguf_contract = build_turboquant_gguf_contract([artifact.metadata for _layer_idx, artifact in selected])

    manifest = {
        "schema_kind": "gguf_embedded_profile",
        "schema_version": GGUF_TURBOQUANT_SCHEMA_VERSION,
        "profile_name": name,
        "profile_kind": "triality_proxy_vector",
        "runtime_mode": PRODUCTION_K_TURBOQUANT_MODE,
        "bits_total": float(bits_total),
        "head_dim": int(head_dim),
        "triality_mode": common_metadata["tq_triality_mode"],
        "triality_view": PRODUCTION_K_TURBOQUANT_VIEW,
        "rotation_policy": common_metadata["tq_rotation_policy"],
        "rotation_block_size": int(common_metadata.get("rotation_block_size", 8)),
        "qjl_bits": int(common_metadata["tq_qjl_bits"]),
        "qjl_dim": int(common_metadata["tq_qjl_dim"]),
        "rotation_seed": int(common_metadata["tq_rotation_seed"]),
        "qjl_seed": int(common_metadata["tq_qjl_seed"]),
        "orthogonality_error": float(common_metadata.get("orthogonality_error", 0.0)),
        "determinant_error_max": float(common_metadata.get("determinant_error_max", 0.0)),
        "view_bundle_complete": bool(common_metadata.get("view_bundle_complete", True)),
        "layer_indices": layer_indices,
        "artifact_tensor_names": tensor_names,
        "strict_gguf_contract": strict_gguf_contract,
    }
    return TurboQuantGGUFProfile(
        name=name,
        kind="triality_proxy_vector",
        runtime_mode=PRODUCTION_K_TURBOQUANT_MODE,
        description="Production K-side Triality proxy with the vector view and learned block-SO(8) rotations.",
        manifest=manifest,
        metadata={
            "bits_total": float(bits_total),
            "head_dim": int(head_dim),
            "layer_indices": layer_indices,
            "rotation_policy": str(common_metadata["tq_rotation_policy"]),
            "rotation_block_size": int(common_metadata.get("rotation_block_size", 8)),
            "rotation_seed": int(common_metadata["tq_rotation_seed"]),
            "qjl_bits": int(common_metadata["tq_qjl_bits"]),
            "qjl_dim": int(common_metadata["tq_qjl_dim"]),
            "qjl_seed": int(common_metadata["tq_qjl_seed"]),
            "triality_mode": str(common_metadata["tq_triality_mode"]),
            "triality_view": PRODUCTION_K_TURBOQUANT_VIEW,
            "orthogonality_error": float(common_metadata.get("orthogonality_error", 0.0)),
            "determinant_error_max": float(common_metadata.get("determinant_error_max", 0.0)),
            "view_bundle_complete": bool(common_metadata.get("view_bundle_complete", True)),
            "artifact_tensor_names": tensor_names,
            **strict_gguf_contract,
        },
        tensors=tensors,
    )


def package_turboquant_gguf(
    *,
    source_path: Path,
    output_path: Path,
    profiles: list[TurboQuantGGUFProfile],
    default_profile: str = GGUF_TURBOQUANT_EXACT_PROFILE,
    hypura_compatibility_profile: str = GGUF_HYPURA_COMPAT_AUTO,
) -> TurboQuantGGUFManifest:
    """Copy an existing GGUF and embed TurboQuant manifests plus artifact tensors."""

    if source_path.resolve() == output_path.resolve():
        raise ValueError("source_path and output_path must differ")
    if not source_path.exists():
        raise FileNotFoundError(f"Source GGUF does not exist: {source_path}")
    if not profiles:
        raise ValueError("At least one embedded TurboQuant profile is required")

    _validate_default_profile(default_profile=default_profile, profiles=profiles)
    _validate_unique_profile_names(profiles)
    strict_gguf_profile = resolve_strict_gguf_contract_profile(
        profiles=profiles,
        requested_profile=hypura_compatibility_profile,
    )
    hypura_bridge = resolve_hypura_compatibility_bridge(
        profiles=profiles,
        requested_profile=hypura_compatibility_profile,
    )

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(source_path)
    _ensure_no_existing_turboquant_namespace(reader)
    arch = _require_source_architecture(reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output_path, arch=arch, use_temp_file=False, endianess=reader.endianess)
    _copy_source_kv_metadata(reader=reader, writer=writer, gguf=gguf)
    packaged_at_utc = datetime.now(timezone.utc).isoformat()
    _write_top_level_manifest(
        writer=writer,
        default_profile=default_profile,
        packaged_at_utc=packaged_at_utc,
        base_architecture=arch,
        profile_names=[profile.name for profile in profiles],
        gguf=gguf,
    )
    if strict_gguf_profile is not None:
        _write_strict_gguf_contract(writer=writer, profile=strict_gguf_profile, gguf=gguf)
    if hypura_bridge is not None:
        bridge_profile = next(
            profile for profile in profiles if profile.name == hypura_bridge.source_profile
        )
        _write_hypura_bridge_metadata(
            writer=writer,
            bridge=hypura_bridge,
            profile=bridge_profile,
            gguf=gguf,
        )
    for profile in profiles:
        _write_profile_manifest(writer=writer, profile=profile, gguf=gguf)

    for tensor in reader.tensors:
        tensor_array = np.array(tensor.data, copy=False)
        tensor_shape = tuple(int(value) for value in tensor_array.shape)
        writer.add_tensor_info(
            name=tensor.name,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_array.dtype,
            tensor_nbytes=tensor_array.nbytes,
            raw_dtype=tensor.tensor_type,
        )
    for profile in profiles:
        for tensor in profile.tensors:
            writer.add_tensor(name=tensor.name, tensor=tensor.data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        tensor_array = np.array(tensor.data, copy=False)
        writer.write_tensor_data(tensor_array, tensor_endianess=reader.endianess)
    for profile in profiles:
        for tensor in profile.tensors:
            writer.write_tensor_data(tensor.data)
    writer.close()
    return read_turboquant_gguf_manifest(output_path)


def read_turboquant_gguf_manifest(path: Path) -> TurboQuantGGUFManifest:
    """Read the embedded TurboQuant manifest from a GGUF file."""

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(path)
    schema_field = reader.get_field(f"{GGUF_TURBOQUANT_NAMESPACE}.schema_version")
    if schema_field is None:
        raise ValueError(f"GGUF file does not contain embedded {GGUF_TURBOQUANT_NAMESPACE} metadata: {path}")
    default_profile = _read_required_string(reader, f"{GGUF_TURBOQUANT_NAMESPACE}.default_profile")
    packaged_at_utc = _read_required_string(reader, f"{GGUF_TURBOQUANT_NAMESPACE}.packaged_at_utc")
    base_architecture = _read_required_string(reader, f"{GGUF_TURBOQUANT_NAMESPACE}.base_architecture")
    profile_names = _read_required_string_array(reader, f"{GGUF_TURBOQUANT_NAMESPACE}.profiles")

    profiles: dict[str, TurboQuantGGUFProfile] = {}
    for profile_name in profile_names:
        prefix = f"{GGUF_TURBOQUANT_NAMESPACE}.profile.{profile_name}."
        kind = _read_required_string(reader, f"{prefix}kind")
        runtime_mode = _read_required_string(reader, f"{prefix}runtime_mode")
        description = _read_required_string(reader, f"{prefix}description")
        manifest_json = _read_required_string(reader, f"{prefix}manifest_json")
        manifest = json.loads(manifest_json)
        metadata: dict[str, GGUFMetadataValue] = {}
        for key, field in reader.fields.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix in {"kind", "runtime_mode", "description", "manifest_json"}:
                continue
            metadata[suffix] = field.contents()
        tensor_names = metadata.get("artifact_tensor_names", [])
        if isinstance(tensor_names, list):
            artifact_tensor_names = tuple(str(item) for item in tensor_names)
        else:
            artifact_tensor_names = tuple()
        profiles[profile_name] = TurboQuantGGUFProfile(
            name=profile_name,
            kind=kind,
            runtime_mode=runtime_mode,
            description=description,
            manifest=manifest,
            metadata=metadata,
            tensors=[
                TurboQuantEmbeddedTensor(name=name, data=np.array(_tensor_by_name(reader, name).data, copy=False))
                for name in artifact_tensor_names
            ],
        )
    return TurboQuantGGUFManifest(
        schema_version=int(schema_field.contents()),
        default_profile=default_profile,
        packaged_at_utc=packaged_at_utc,
        base_architecture=base_architecture,
        profiles=profiles,
    )


def resolve_hypura_compatibility_bridge(
    *,
    profiles: list[TurboQuantGGUFProfile],
    requested_profile: str = GGUF_HYPURA_COMPAT_AUTO,
) -> HypuraTurboQuantBridgeConfig | None:
    """Resolve an optional Hypura/Kobold-compatible bridge profile from embedded profiles."""

    if requested_profile == GGUF_HYPURA_COMPAT_OFF:
        return None

    profiles_by_name = {profile.name: profile for profile in profiles}
    if requested_profile == GGUF_HYPURA_COMPAT_AUTO:
        profile = profiles_by_name.get("so8_triality_vector")
        return None if profile is None else build_hypura_bridge_config(profile)

    if requested_profile not in profiles_by_name:
        available = ", ".join(sorted(profiles_by_name))
        raise ValueError(
            "Requested Hypura compatibility profile is not embedded in the GGUF: "
            f"{requested_profile!r}. Available profiles: {available}"
        )
    return build_hypura_bridge_config(profiles_by_name[requested_profile])


def resolve_strict_gguf_contract_profile(
    *,
    profiles: list[TurboQuantGGUFProfile],
    requested_profile: str = GGUF_HYPURA_COMPAT_AUTO,
) -> TurboQuantGGUFProfile | None:
    """Resolve which embedded profile should own the canonical top-level `tq_*` GGUF contract."""

    research_profiles = [profile for profile in profiles if profile.kind != "paper_faithful"]
    profiles_by_name = {profile.name: profile for profile in profiles}

    if requested_profile == GGUF_HYPURA_COMPAT_OFF:
        return research_profiles[0] if len(research_profiles) == 1 else None

    if requested_profile == GGUF_HYPURA_COMPAT_AUTO:
        profile = profiles_by_name.get("so8_triality_vector")
        return profile if profile is not None else (research_profiles[0] if len(research_profiles) == 1 else None)

    if requested_profile not in profiles_by_name:
        available = ", ".join(sorted(profiles_by_name))
        raise ValueError(
            "Requested strict GGUF contract profile is not embedded in the GGUF: "
            f"{requested_profile!r}. Available profiles: {available}"
        )
    profile = profiles_by_name[requested_profile]
    return None if profile.kind == "paper_faithful" else profile


def build_hypura_bridge_config(profile: TurboQuantGGUFProfile) -> HypuraTurboQuantBridgeConfig:
    """Map an embedded TurboQuant profile to the legacy Hypura GGUF compatibility bridge."""

    if profile.kind == "triality_proxy_vector":
        rotation_seed = int(profile.metadata.get("rotation_seed", 0))
        return HypuraTurboQuantBridgeConfig(
            source_profile=profile.name,
            mode=TRIALITY_RUNTIME_MODE,
            rotation_policy="triality_vector",
            triality_view=PRODUCTION_K_TURBOQUANT_VIEW,
            rotation_seed=rotation_seed,
        )
    if profile.kind == "paper_faithful":
        raise ValueError(
            "Hypura GGUF compatibility currently supports only embedded research/triality profiles. "
            "Paper-faithful profiles still require a parsed paper sidecar in Hypura."
        )
    raise ValueError(
        "Unsupported embedded profile for Hypura GGUF compatibility bridge: "
        f"{profile.name} ({profile.kind})"
    )


def read_hypura_gguf_bridge_config(path: Path) -> HypuraTurboQuantBridgeConfig | None:
    """Read the embedded Hypura/Kobold bridge metadata from a GGUF file, if present."""

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(path)
    enabled = _read_optional_bool(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.enabled")
    if not enabled:
        return None
    public_mode = _read_required_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.mode")
    runtime_mode = _read_optional_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.runtime_mode")
    raw_triality_view = _read_optional_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.triality_view")
    if runtime_mode is None:
        runtime_mode = {
            "paper-faithful": "paper-key-only",
            TRIALITY_PROXY_PARETO_MODE: TRIALITY_RUNTIME_MODE,
            TRIALITY_PROXY_PARETO_LEGACY_ALIAS: TRIALITY_RUNTIME_MODE,
        }.get(public_mode, public_mode)
    return HypuraTurboQuantBridgeConfig(
        source_profile=_read_optional_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.source_profile") or "unknown",
        mode=normalize_triality_runtime_mode(runtime_mode),
        rotation_policy=_read_optional_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.rotation_policy"),
        triality_view=normalize_triality_view(raw_triality_view) if raw_triality_view is not None else None,
        triality_mix=_read_optional_float(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.triality_mix"),
        rotation_seed=_read_optional_int(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.rotation_seed") or 0,
        artifact_path=_read_optional_string(reader, f"{HYPURA_TURBOQUANT_NAMESPACE}.artifact"),
    )


def build_hypura_serve_command(
    *,
    gguf_path: Path,
    host: str = "127.0.0.1",
    port: int = 5001,
    context: int = 8192,
    turboquant_mode: str = "gguf-auto",
    release: bool = False,
) -> list[str]:
    """Build a `cargo run -p hypura -- serve ...` command for Kobold-compatible serving."""

    if port <= 0:
        raise ValueError(f"port must be positive, got {port}")
    if context <= 0:
        raise ValueError(f"context must be positive, got {context}")

    supported_modes = {"exact", "paper-key-only", "paper-full-kv", "research-kv-split", TRIALITY_RUNTIME_MODE}
    if turboquant_mode == "gguf-auto":
        bridge = read_hypura_gguf_bridge_config(gguf_path)
        if bridge is None:
            raise ValueError(
                "GGUF does not contain Hypura bridge metadata. Repackage with "
                "--hypura-compatible-profile auto or pass an explicit --turboquant-mode."
            )
        resolved_mode = bridge.mode
    elif turboquant_mode in supported_modes or turboquant_mode in TRIALITY_RUNTIME_MODE_ALIASES:
        resolved_mode = normalize_triality_runtime_mode(turboquant_mode)
    else:
        raise ValueError(
            "Unsupported Hypura turboquant_mode. Expected one of "
            "'gguf-auto', 'exact', 'paper-key-only', 'paper-full-kv', or 'research-kv-split'."
        )

    command = ["cargo", "run"]
    if release:
        command.append("--release")
    command.extend(
        [
            "-p",
            "hypura",
            "--",
            "serve",
            str(gguf_path),
            "--host",
            host,
            "--port",
            str(port),
            "--context",
            str(context),
            "--turboquant-mode",
            resolved_mode,
        ]
    )
    return command


def _copy_source_kv_metadata(*, reader: Any, writer: Any, gguf: ModuleType) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue
        writer.add_key_value(
            key,
            field.contents(),
            field.types[0],
            field.types[-1] if field.types and field.types[0] == gguf.GGUFValueType.ARRAY else None,
        )


def _ensure_no_existing_turboquant_namespace(reader: Any) -> None:
    for key in reader.fields.keys():
        if key.startswith(f"{GGUF_TURBOQUANT_NAMESPACE}."):
            raise ValueError(
                "Source GGUF already contains embedded turboquant metadata. "
                "Refusing to package over an existing turboquant namespace."
            )
        if key.startswith(f"{HYPURA_TURBOQUANT_NAMESPACE}."):
            raise ValueError(
                "Source GGUF already contains embedded Hypura TurboQuant metadata. "
                "Refusing to package over an existing hypura.turboquant namespace."
            )
    for tensor in reader.tensors:
        if tensor.name.startswith(_TENSOR_NAME_PREFIX):
            raise ValueError(
                "Source GGUF already contains embedded turboquant tensors. "
                "Refusing to package over an existing turboquant namespace."
            )


def _require_source_architecture(reader: Any) -> str:
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError("GGUF metadata is missing general.architecture")
    arch = str(arch_field.contents())
    if not arch:
        raise ValueError("general.architecture must not be empty")
    return arch


def _write_top_level_manifest(
    *,
    writer: Any,
    default_profile: str,
    packaged_at_utc: str,
    base_architecture: str,
    profile_names: list[str],
    gguf: ModuleType,
) -> None:
    writer.add_key_value(f"{GGUF_TURBOQUANT_NAMESPACE}.schema_version", GGUF_TURBOQUANT_SCHEMA_VERSION, gguf.GGUFValueType.UINT32)
    writer.add_key_value(f"{GGUF_TURBOQUANT_NAMESPACE}.default_profile", default_profile, gguf.GGUFValueType.STRING)
    writer.add_key_value(f"{GGUF_TURBOQUANT_NAMESPACE}.packaged_at_utc", packaged_at_utc, gguf.GGUFValueType.STRING)
    writer.add_key_value(f"{GGUF_TURBOQUANT_NAMESPACE}.base_architecture", base_architecture, gguf.GGUFValueType.STRING)
    writer.add_key_value(
        f"{GGUF_TURBOQUANT_NAMESPACE}.profiles",
        profile_names,
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.STRING,
    )


def _write_strict_gguf_contract(*, writer: Any, profile: TurboQuantGGUFProfile, gguf: ModuleType) -> None:
    contract = profile.manifest.get("strict_gguf_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Embedded profile {profile.name!r} is missing manifest.strict_gguf_contract and cannot export canonical tq_* keys"
        )
    validate_turboquant_gguf_contract(contract)

    writer.add_key_value("tq_schema_version", int(contract["tq_schema_version"]), gguf.GGUFValueType.UINT32)
    for key in TURBOQUANT_GGUF_FLOAT_KEYS:
        writer.add_key_value(key, contract[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
    for key in TURBOQUANT_GGUF_U32_KEYS:
        writer.add_key_value(key, contract[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    for key in TURBOQUANT_GGUF_STRING_KEYS:
        writer.add_key_value(key, contract[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING)


def _write_hypura_bridge_metadata(
    *,
    writer: Any,
    bridge: HypuraTurboQuantBridgeConfig,
    profile: TurboQuantGGUFProfile,
    gguf: ModuleType,
) -> None:
    if profile.kind == "paper_faithful":
        public_mode = "paper-faithful"
    else:
        public_mode = TRIALITY_PROXY_PARETO_MODE

    payload = build_triality_payload(
        mode=public_mode,
        head_dim=int(profile.manifest.get("head_dim", profile.metadata.get("head_dim", 0))),
        num_layers=len(profile.manifest.get("layer_indices", [])) or 1,
        num_kv_heads=1,
        rotation_seed=int(profile.metadata.get("rotation_seed", bridge.rotation_seed)),
        source_manifest=profile.manifest,
    )
    payload_json = payload_json_dumps(payload)
    metadata = build_triality_metadata(
        mode=public_mode,
        payload_json=payload_json,
        weight_plan=payload.get("weight_plan"),
        rotation_policy=bridge.rotation_policy or str(profile.metadata.get("rotation_policy", "")),
        rotation_seed=bridge.rotation_seed,
        triality_view=bridge.triality_view or str(profile.metadata.get("triality_view", "none")),
        triality_mix=bridge.triality_mix if bridge.triality_mix is not None else float(profile.metadata.get("triality_mix", 0.0)),
        k_bits=float(profile.metadata.get("bits_total", profile.manifest.get("bits_total", 0.0))),
        v_bits=float(profile.metadata.get("v_bits", 8.0 if public_mode == TRIALITY_PROXY_PARETO_MODE else 16.0)),
        runtime_mode=bridge.mode,
        source_profile=bridge.source_profile,
    )
    metadata["hypura.turboquant.orthogonality_error"] = float(profile.metadata.get("orthogonality_error", 0.0))
    metadata["hypura.turboquant.determinant_error_max"] = float(profile.metadata.get("determinant_error_max", 0.0))
    metadata["hypura.turboquant.view_bundle_complete"] = bool(profile.metadata.get("view_bundle_complete", True))
    validate_triality_metadata(metadata)

    for key, value in metadata.items():
        if isinstance(value, bool):
            writer.add_key_value(key, value, gguf.GGUFValueType.BOOL)
        elif isinstance(value, int):
            if key.endswith("payload_bytes"):
                writer.add_key_value(key, value, gguf.GGUFValueType.UINT64)
            else:
                writer.add_key_value(key, value, gguf.GGUFValueType.UINT32)
        elif isinstance(value, float):
            writer.add_key_value(key, value, gguf.GGUFValueType.FLOAT32)
        else:
            writer.add_key_value(key, str(value), gguf.GGUFValueType.STRING)
    if bridge.artifact_path is not None:
        writer.add_key_value(
            f"{HYPURA_TURBOQUANT_NAMESPACE}.artifact",
            bridge.artifact_path,
            gguf.GGUFValueType.STRING,
        )


def _write_profile_manifest(*, writer: Any, profile: TurboQuantGGUFProfile, gguf: ModuleType) -> None:
    prefix = f"{GGUF_TURBOQUANT_NAMESPACE}.profile.{profile.name}."
    writer.add_key_value(f"{prefix}kind", profile.kind, gguf.GGUFValueType.STRING)
    writer.add_key_value(f"{prefix}runtime_mode", profile.runtime_mode, gguf.GGUFValueType.STRING)
    writer.add_key_value(f"{prefix}description", profile.description, gguf.GGUFValueType.STRING)
    writer.add_key_value(
        f"{prefix}manifest_json",
        json.dumps(profile.manifest, sort_keys=True, separators=(",", ":")),
        gguf.GGUFValueType.STRING,
    )
    for metadata_key, metadata_value in profile.metadata.items():
        _write_profile_metadata_value(
            writer=writer,
            key=f"{prefix}{metadata_key}",
            value=metadata_value,
            gguf=gguf,
        )


def _write_profile_metadata_value(*, writer: Any, key: str, value: GGUFMetadataValue, gguf: ModuleType) -> None:
    final_key = key.split(".")[-1]
    if not _METADATA_KEY_RE.fullmatch(final_key):
        raise ValueError(f"Unsupported GGUF profile metadata key suffix: {final_key!r}")
    if isinstance(value, bool):
        writer.add_key_value(key, value, gguf.GGUFValueType.BOOL)
        return
    if isinstance(value, int):
        writer.add_key_value(key, value, gguf.GGUFValueType.INT32)
        return
    if isinstance(value, float):
        writer.add_key_value(key, value, gguf.GGUFValueType.FLOAT32)
        return
    if isinstance(value, str):
        writer.add_key_value(key, value, gguf.GGUFValueType.STRING)
        return
    if isinstance(value, list):
        if not value:
            return
        first = value[0]
        if isinstance(first, bool):
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.BOOL)
            return
        if isinstance(first, int):
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
            return
        if isinstance(first, float):
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
            return
        if isinstance(first, str):
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING)
            return
    raise TypeError(f"Unsupported GGUF metadata value for {key}: {type(value)!r}")


def _validate_default_profile(*, default_profile: str, profiles: list[TurboQuantGGUFProfile]) -> None:
    profile_names = {profile.name for profile in profiles}
    if default_profile != GGUF_TURBOQUANT_EXACT_PROFILE and default_profile not in profile_names:
        raise ValueError(
            "default_profile must be 'exact' or the name of an embedded profile; "
            f"got {default_profile!r}"
        )


def _validate_unique_profile_names(profiles: list[TurboQuantGGUFProfile]) -> None:
    seen: set[str] = set()
    for profile in profiles:
        _validate_profile_name(profile.name)
        if profile.name in seen:
            raise ValueError(f"Duplicate embedded TurboQuant profile name: {profile.name}")
        seen.add(profile.name)


def _validate_profile_name(name: str) -> None:
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(f"Profile names must match {_PROFILE_NAME_RE.pattern!r}; got {name!r}")
    if name == GGUF_TURBOQUANT_EXACT_PROFILE:
        raise ValueError("'exact' is reserved as the no-op runtime profile name")


def _triality_rotation_tensor_name(*, profile_name: str, layer_idx: int, bits_total: float) -> str:
    bit_token = str(bits_total).replace(".", "p")
    tensor_name = f"tq.p.{profile_name}.l{layer_idx:02d}.b{bit_token}.rot"
    if len(tensor_name) <= 63:
        return tensor_name

    profile_token = hashlib.sha1(profile_name.encode("utf-8")).hexdigest()[:10]
    tensor_name = f"tq.p.{profile_token}.l{layer_idx:02d}.b{bit_token}.rot"
    if len(tensor_name) <= 63:
        return tensor_name
    raise ValueError(
        "Embedded TurboQuant tensor name exceeds GGUF length limits even after hashing: "
        f"{tensor_name!r}"
    )


def _read_required_string(reader: Any, key: str) -> str:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"Missing GGUF metadata field: {key}")
    return str(field.contents())


def _read_required_string_array(reader: Any, key: str) -> list[str]:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"Missing GGUF metadata field: {key}")
    return [str(item) for item in field.contents()]


def _read_optional_bool(reader: Any, key: str) -> bool | None:
    field = reader.get_field(key)
    return None if field is None else bool(field.contents())


def _read_optional_float(reader: Any, key: str) -> float | None:
    field = reader.get_field(key)
    return None if field is None else float(field.contents())


def _read_optional_int(reader: Any, key: str) -> int | None:
    field = reader.get_field(key)
    return None if field is None else int(field.contents())


def _read_optional_string(reader: Any, key: str) -> str | None:
    field = reader.get_field(key)
    return None if field is None else str(field.contents())


def _tensor_by_name(reader: Any, tensor_name: str) -> Any:
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return tensor
    raise ValueError(f"Embedded TurboQuant tensor not found in GGUF: {tensor_name}")


__all__ = [
    "build_hypura_bridge_config",
    "build_hypura_serve_command",
    "GGUF_TURBOQUANT_EXACT_PROFILE",
    "GGUF_HYPURA_COMPAT_AUTO",
    "GGUF_HYPURA_COMPAT_OFF",
    "GGUF_TURBOQUANT_NAMESPACE",
    "GGUF_TURBOQUANT_SCHEMA_VERSION",
    "HYPURA_TURBOQUANT_NAMESPACE",
    "HypuraTurboQuantBridgeConfig",
    "TurboQuantEmbeddedTensor",
    "TurboQuantGGUFManifest",
    "TurboQuantGGUFProfile",
    "build_paper_gguf_profile",
    "build_so8_triality_vector_gguf_profile",
    "import_vendor_gguf",
    "infer_gguf_attention_head_dim",
    "infer_gguf_block_count",
    "package_turboquant_gguf",
    "read_hypura_gguf_bridge_config",
    "read_turboquant_gguf_manifest",
    "resolve_strict_gguf_contract_profile",
    "resolve_hypura_compatibility_bridge",
]
