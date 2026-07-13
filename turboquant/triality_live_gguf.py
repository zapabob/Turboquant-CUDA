from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from turboquant.gguf_profiles import import_vendor_gguf
from turboquant.schema import (
    TURBOQUANT_GGUF_FLOAT_KEYS,
    TURBOQUANT_GGUF_STRING_KEYS,
    TURBOQUANT_GGUF_U32_KEYS,
    validate_turboquant_gguf_contract,
)
from turboquant.triality_contract import (
    TRIALITY_GGUF_SCHEMA_V2,
    TRIALITY_PROXY_PARETO_MODE,
    build_triality_metadata,
    build_triality_payload,
    payload_json_dumps,
    validate_triality_metadata,
    validate_triality_payload,
)
from turboquant.triality_schema_v2 import (
    TRIALITY_IDENTITY_DEV_ROTATION_POLICY,
    build_triality_v2_tensors,
    tensor_sha256,
)


TRIALITY_LIVE_MANIFEST_VERSION = 1
TRIALITY_LIVE_DEFAULT_TIMESTAMP = "1970-01-01T00:00:00+00:00"
TRIALITY_LIVE_WEIGHT_POLICY = "preserve-source-weights"
_PRODUCER = "turboquant.triality_live_gguf"
_TQ_SCHEMA_VERSION = 1
_TQ_TOTAL_BITS = 3.5
_TQ_QJL_BITS = 1
_TQ_STAGE1_BITS = _TQ_TOTAL_BITS - _TQ_QJL_BITS
_TQ_QJL_SEED = 71
_TRIALITY_PREFIX = "hypura.turboquant."
_GGUF_MAX_TENSOR_NAME_BYTES = 64


@dataclass(frozen=True, slots=True)
class TrialityLiveGGUFSummary:
    """Verified identity-view GGUF materialization result."""

    source_path: Path
    output_path: Path
    manifest_path: Path | None
    source_sha256: str
    output_sha256: str
    architecture: str
    file_type: int
    alignment: int
    endianness: str
    head_dim: int
    num_layers: int
    num_kv_heads: int
    source_tensor_count: int
    added_tensor_count: int
    profile_id: str
    generated_at_utc: str


def manifest_path_for(output_path: Path) -> Path:
    """Return the deterministic sidecar path for an output GGUF."""

    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_dev), int(stat.st_ino)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> bool:
    try:
        if _file_identity(path) != identity:
            return False
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _acquire_publish_lock(
    lock_path: Path, *, token: str
) -> tuple[int, tuple[int, int]]:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Triality live publish is already reserved: {lock_path}"
        ) from exc
    stat = os.fstat(fd)
    identity = int(stat.st_dev), int(stat.st_ino)
    try:
        os.write(fd, token.encode("ascii"))
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        _unlink_if_owned(lock_path, identity)
        raise
    return fd, identity


def _require_publish_destinations_absent(output_path: Path, sidecar_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output GGUF already exists: {output_path}")
    if sidecar_path.exists():
        raise FileExistsError(f"Output manifest already exists: {sidecar_path}")


def _link_no_replace(source: Path, destination: Path) -> tuple[int, int]:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Publish destination already exists: {destination}"
        ) from exc
    return _file_identity(destination)


def materialize_triality_live_gguf(
    *,
    source_path: Path,
    output_path: Path,
    profile_id: str = "liveq4",
    generated_at_utc: str = TRIALITY_LIVE_DEFAULT_TIMESTAMP,
    development_identity_views: bool = False,
    disable_weight_conversion: bool = False,
) -> TrialityLiveGGUFSummary:
    """Copy a llama Q4_0 GGUF and add a complete identity Triality v2 bundle.

    Both safety switches must be explicitly enabled by callers. The output and
    its sidecar are published under one lock with same-directory atomic,
    no-replace links. Existing destinations are never overwritten.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    sidecar_path = manifest_path_for(output_path)
    _require_opt_ins(
        development_identity_views=development_identity_views,
        disable_weight_conversion=disable_weight_conversion,
    )
    _validate_timestamp(generated_at_utc)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source GGUF does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source_path and output_path must differ")
    if output_path.exists():
        raise FileExistsError(f"Output GGUF already exists: {output_path}")
    if sidecar_path.exists():
        raise FileExistsError(f"Output manifest already exists: {sidecar_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source_path)
    source_size = source_path.stat().st_size
    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(source_path)
    architecture = _require_string(reader, "general.architecture")
    if architecture != "llama":
        raise ValueError(
            f"Triality live materialization requires architecture 'llama', got {architecture!r}"
        )
    file_type = _require_int(reader, "general.file_type")
    if file_type != int(gguf.LlamaFileType.MOSTLY_Q4_0):
        raise ValueError(
            "Triality live materialization requires general.file_type Q4_0 "
            f"({int(gguf.LlamaFileType.MOSTLY_Q4_0)}), got {file_type}"
        )
    num_layers = _require_positive_int(reader, "llama.block_count")
    head_dim = _infer_head_dim(reader, architecture)
    if head_dim % 8 != 0:
        raise ValueError(
            f"llama attention head dimension must be a multiple of 8, got {head_dim}"
        )
    num_kv_heads = _infer_kv_heads(reader, architecture)
    _ensure_clean_namespace(reader)
    source_tq = _read_existing_tq_contract(
        reader=reader,
        num_layers=num_layers,
        gguf=gguf,
    )
    source_manifest = {
        "architecture": architecture,
        "development_identity_views": True,
        "file_type": file_type,
        "generated_at_utc": generated_at_utc,
        "generator": _PRODUCER,
        "sha256": source_sha256,
        "size_bytes": source_size,
        "source_filename": source_path.name,
        "weight_conversion_disabled": True,
    }
    payload = build_triality_payload(
        mode=TRIALITY_PROXY_PARETO_MODE,
        head_dim=head_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        model_family="llama",
        weight_source_ftype="q4_0",
        weight_policy=TRIALITY_LIVE_WEIGHT_POLICY,
        weight_protected_roles=[],
        weight_protected_layers=[],
        modality_scope="text-only",
        rotation_seed=0,
        source_manifest=source_manifest,
        schema_version=TRIALITY_GGUF_SCHEMA_V2,
        profile_id=profile_id,
        enable_ncka=False,
        enable_urt=False,
        rotation_policy=TRIALITY_IDENTITY_DEV_ROTATION_POLICY,
        weight_enabled=False,
    )
    payload_json = payload_json_dumps(payload)
    public_metadata = build_triality_metadata(
        mode=TRIALITY_PROXY_PARETO_MODE,
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
        rotation_policy=TRIALITY_IDENTITY_DEV_ROTATION_POLICY,
        rotation_seed=0,
        runtime_mode=str(payload["runtime_mode"]),
        cache_type_k=str(payload["cache_type_k"]),
        cache_type_v=str(payload["cache_type_v"]),
        schema_version=TRIALITY_GGUF_SCHEMA_V2,
    )
    strict_tq = _build_strict_tq_contract(num_layers=num_layers, head_dim=head_dim)
    strict_tq_to_add = strict_tq if source_tq is None else {}
    tensors = build_triality_v2_tensors(payload)
    _validate_tensor_names(tensors)

    token = uuid4().hex
    temp_output = output_path.parent / f".{output_path.name}.{token}.tmp"
    temp_manifest = output_path.parent / f".{sidecar_path.name}.{token}.tmp"
    lock_path = output_path.parent / f".{output_path.name}.triality-live.lock"
    lock_fd, lock_identity = _acquire_publish_lock(lock_path, token=token)
    output_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    published = False
    try:
        _require_publish_destinations_absent(output_path, sidecar_path)
        _write_augmented_gguf(
            source_reader=reader,
            output_path=temp_output,
            strict_tq=strict_tq_to_add,
            public_metadata=public_metadata,
            tensors=tensors,
            gguf=gguf,
        )
        if (
            _sha256_file(source_path) != source_sha256
            or source_path.stat().st_size != source_size
        ):
            raise RuntimeError("Source GGUF changed during materialization")
        verified = _verify_triality_live_gguf(
            source_path=source_path,
            model_path=temp_output,
            manifest_path=None,
            require_manifest=False,
        )
        sidecar = _build_sidecar(
            summary=verified,
            source_reader=reader,
            model_path=temp_output,
            output_filename=output_path.name,
        )
        _write_json_file(temp_manifest, sidecar)
        gc.collect()
        output_identity = _link_no_replace(temp_output, output_path)
        sidecar_identity = _link_no_replace(temp_manifest, sidecar_path)
        summary = verify_triality_live_gguf(
            source_path=source_path,
            model_path=output_path,
            manifest_path=sidecar_path,
        )
        published = True
        return summary
    except BaseException:
        if sidecar_identity is not None:
            _unlink_if_owned(sidecar_path, sidecar_identity)
        if output_identity is not None:
            _unlink_if_owned(output_path, output_identity)
        raise
    finally:
        for temporary in (temp_output, temp_manifest):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            os.close(lock_fd)
        finally:
            _unlink_if_owned(lock_path, lock_identity)
        if not published:
            gc.collect()


def verify_triality_live_gguf(
    *,
    source_path: Path,
    model_path: Path,
    manifest_path: Path | None = None,
) -> TrialityLiveGGUFSummary:
    """Verify source preservation, strict types, identity tensors, and sidecar."""

    return _verify_triality_live_gguf(
        source_path=Path(source_path),
        model_path=Path(model_path),
        manifest_path=manifest_path_for(Path(model_path))
        if manifest_path is None
        else Path(manifest_path),
        require_manifest=True,
    )


def _verify_triality_live_gguf(
    *,
    source_path: Path,
    model_path: Path,
    manifest_path: Path | None,
    require_manifest: bool,
) -> TrialityLiveGGUFSummary:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source GGUF does not exist: {source_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Materialized GGUF does not exist: {model_path}")
    if source_path.resolve() == model_path.resolve():
        raise ValueError("source_path and model_path must differ")
    if require_manifest and (manifest_path is None or not manifest_path.is_file()):
        raise FileNotFoundError(
            f"Materialized GGUF manifest does not exist: {manifest_path}"
        )

    gguf = import_vendor_gguf()
    source = gguf.GGUFReader(source_path)
    model = gguf.GGUFReader(model_path)
    _verify_source_contract(source=source, model=model, gguf=gguf)

    architecture = _require_string(model, "general.architecture")
    file_type = _require_int(model, "general.file_type")
    num_layers = _require_positive_int(model, "llama.block_count")
    head_dim = _infer_head_dim(model, architecture)
    num_kv_heads = _infer_kv_heads(model, architecture)
    payload_json = _require_string(model, "hypura.turboquant.payload_json")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("hypura.turboquant.payload_json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("hypura.turboquant.payload_json must decode to an object")
    validate_triality_payload(payload)
    _verify_live_payload(
        payload,
        head_dim=head_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        source_sha256=_sha256_file(source_path),
        source_size=source_path.stat().st_size,
    )
    expected_public = build_triality_metadata(
        mode=TRIALITY_PROXY_PARETO_MODE,
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
        rotation_policy=TRIALITY_IDENTITY_DEV_ROTATION_POLICY,
        rotation_seed=0,
        runtime_mode=str(payload["runtime_mode"]),
        cache_type_k=str(payload["cache_type_k"]),
        cache_type_v=str(payload["cache_type_v"]),
        schema_version=TRIALITY_GGUF_SCHEMA_V2,
    )
    validate_triality_metadata(expected_public)
    source_tq = _read_existing_tq_contract(
        reader=source,
        num_layers=num_layers,
        gguf=gguf,
    )
    expected_tq = (
        _build_strict_tq_contract(num_layers=num_layers, head_dim=head_dim)
        if source_tq is None
        else source_tq
    )
    model_tq = _read_existing_tq_contract(
        reader=model,
        num_layers=num_layers,
        gguf=gguf,
    )
    if model_tq != expected_tq:
        raise ValueError(
            "Materialized GGUF tq_* contract differs from the canonical source contract"
        )
    _verify_metadata_bundle(
        source=source,
        model=model,
        strict_tq=expected_tq if source_tq is None else {},
        public_metadata=expected_public,
        gguf=gguf,
    )
    expected_tensors = build_triality_v2_tensors(payload)
    _validate_tensor_names(expected_tensors)
    _verify_added_tensors(
        source=source,
        model=model,
        expected_tensors=expected_tensors,
        gguf=gguf,
    )

    source_sha256 = _sha256_file(source_path)
    output_sha256 = _sha256_file(model_path)
    source_manifest = payload["source_manifest"]
    generated_at_utc = str(source_manifest["generated_at_utc"])
    summary = TrialityLiveGGUFSummary(
        source_path=source_path,
        output_path=model_path,
        manifest_path=manifest_path if require_manifest else None,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        architecture=architecture,
        file_type=file_type,
        alignment=int(model.alignment),
        endianness=model.endianess.name.lower(),
        head_dim=head_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        source_tensor_count=len(source.tensors),
        added_tensor_count=len(expected_tensors),
        profile_id=str(payload["profile_id"]),
        generated_at_utc=generated_at_utc,
    )
    if require_manifest:
        assert manifest_path is not None
        _verify_sidecar(
            manifest_path=manifest_path,
            summary=summary,
            source=source,
            model=model,
        )
    return summary


def _require_opt_ins(
    *, development_identity_views: bool, disable_weight_conversion: bool
) -> None:
    if not development_identity_views or not disable_weight_conversion:
        raise PermissionError(
            "materialization requires both development_identity_views=True and "
            "disable_weight_conversion=True"
        )


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")


def _build_strict_tq_contract(*, num_layers: int, head_dim: int) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "tq_schema_version": _TQ_SCHEMA_VERSION,
        "tq_total_bits": [_TQ_TOTAL_BITS] * num_layers,
        "tq_runtime_bits_per_channel": [_TQ_TOTAL_BITS] * num_layers,
        "tq_stage1_effective_bits": [_TQ_STAGE1_BITS] * num_layers,
        "tq_qjl_bits": [_TQ_QJL_BITS] * num_layers,
        "tq_qjl_dim": [head_dim] * num_layers,
        "tq_rotation_policy": [TRIALITY_IDENTITY_DEV_ROTATION_POLICY] * num_layers,
        "tq_rotation_seed": [0] * num_layers,
        "tq_qjl_seed": [_TQ_QJL_SEED] * num_layers,
        "tq_triality_mode": ["key_only_block_so8_triality_vector"] * num_layers,
        "tq_triality_view": ["vector"] * num_layers,
        "tq_stage1_allocation_scheme": ["magnitude-topk"] * num_layers,
        "tq_stage1_bitwidth_payload_dtype": ["uint8"] * num_layers,
        "tq_norm_dtype": ["float32"] * num_layers,
        "tq_sign_pack_format": ["int8_unpacked_binary"] * num_layers,
    }
    validate_turboquant_gguf_contract(contract, expected_len=num_layers)
    return contract


def _write_augmented_gguf(
    *,
    source_reader: Any,
    output_path: Path,
    strict_tq: dict[str, Any],
    public_metadata: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
    gguf: Any,
) -> None:
    writer = gguf.GGUFWriter(
        output_path,
        arch="llama",
        use_temp_file=False,
        endianess=source_reader.endianess,
    )
    _enable_empty_array_packing(writer=writer, gguf=gguf)
    writer.data_alignment = int(source_reader.alignment)
    _copy_source_metadata(reader=source_reader, writer=writer, gguf=gguf)
    _write_typed_metadata(writer=writer, metadata=strict_tq, gguf=gguf)
    _write_typed_metadata(writer=writer, metadata=public_metadata, gguf=gguf)

    added_arrays: list[tuple[str, np.ndarray[Any, Any]]] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        logical_shape = tuple(int(value) for value in tensor["shape"])
        storage_shape = tuple(reversed(logical_shape))
        array = np.asarray(tensor["data"], dtype=np.float32).reshape(storage_shape)
        added_arrays.append((name, array))

    for tensor in source_reader.tensors:
        tensor_array = np.asarray(tensor.data)
        writer.add_tensor_info(
            name=tensor.name,
            tensor_shape=tuple(int(value) for value in tensor_array.shape),
            tensor_dtype=tensor_array.dtype,
            tensor_nbytes=int(tensor.n_bytes),
            raw_dtype=tensor.tensor_type,
        )
    for name, array in added_arrays:
        writer.add_tensor_info(
            name=name,
            tensor_shape=array.shape,
            tensor_dtype=array.dtype,
            tensor_nbytes=array.nbytes,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in source_reader.tensors:
        writer.write_tensor_data(
            np.asarray(tensor.data), tensor_endianess=source_reader.endianess
        )
    for _name, array in added_arrays:
        writer.write_tensor_data(array)
    writer.close()


def _enable_empty_array_packing(*, writer: Any, gguf: Any) -> None:
    original_pack = writer._pack_val

    def pack_value(
        val: Any,
        vtype: Any,
        add_vtype: bool,
        sub_type: Any | None = None,
    ) -> bytes:
        if (
            vtype == gguf.GGUFValueType.ARRAY
            and isinstance(val, (list, tuple))
            and not val
            and sub_type is not None
        ):
            packed = bytearray()
            if add_vtype:
                packed += writer._pack("I", vtype)
            packed += writer._pack("I", sub_type)
            packed += writer._pack("Q", 0)
            return bytes(packed)
        return original_pack(val, vtype, add_vtype, sub_type)

    writer._pack_val = pack_value


def _copy_source_metadata(*, reader: Any, writer: Any, gguf: Any) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(
            key,
            field.contents(),
            field.types[0],
            _field_array_subtype(field=field, gguf=gguf),
        )


def _write_typed_metadata(*, writer: Any, metadata: dict[str, Any], gguf: Any) -> None:
    for key, value in metadata.items():
        main_type, subtype = _metadata_type(key=key, value=value, gguf=gguf)
        writer.add_key_value(key, value, main_type, subtype)


def _metadata_type(*, key: str, value: Any, gguf: Any) -> tuple[Any, Any | None]:
    if key == "tq_schema_version":
        return gguf.GGUFValueType.UINT32, None
    if key in TURBOQUANT_GGUF_FLOAT_KEYS:
        return gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32
    if key in TURBOQUANT_GGUF_U32_KEYS:
        return gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32
    if key in TURBOQUANT_GGUF_STRING_KEYS:
        return gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING
    if key in {
        "hypura.turboquant.triality.views",
        "hypura.turboquant.ncka.coordinate_names",
        "hypura.turboquant.urt.supported_representations",
    }:
        return gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING
    if key in {
        "hypura.turboquant.triality.weights",
        "hypura.turboquant.triality.bias",
        "hypura.turboquant.triality.scale",
        "hypura.turboquant.triality.temperature",
    }:
        return gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32
    if isinstance(value, bool):
        return gguf.GGUFValueType.BOOL, None
    if isinstance(value, int):
        if key.endswith("payload_bytes"):
            return gguf.GGUFValueType.UINT64, None
        return gguf.GGUFValueType.UINT32, None
    if isinstance(value, float):
        return gguf.GGUFValueType.FLOAT32, None
    if isinstance(value, str):
        return gguf.GGUFValueType.STRING, None
    raise TypeError(f"No canonical GGUF metadata type for {key}: {type(value)!r}")


def _ensure_clean_namespace(reader: Any) -> None:
    for key in reader.fields:
        if key.startswith(_TRIALITY_PREFIX) or key.startswith("turboquant."):
            raise ValueError(
                f"Source GGUF already contains TurboQuant namespace metadata: {key}"
            )
    for tensor in reader.tensors:
        if tensor.name.startswith("turboquant."):
            raise ValueError(
                f"Source GGUF already contains TurboQuant namespace tensor: {tensor.name}"
            )


def _read_existing_tq_contract(
    *, reader: Any, num_layers: int, gguf: Any
) -> dict[str, Any] | None:
    expected_keys = {
        "tq_schema_version",
        *TURBOQUANT_GGUF_FLOAT_KEYS,
        *TURBOQUANT_GGUF_U32_KEYS,
        *TURBOQUANT_GGUF_STRING_KEYS,
    }
    actual_keys = {key for key in reader.fields if key.startswith("tq_")}
    if not actual_keys:
        return None
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Source GGUF tq_* contract is partial or non-canonical: "
            f"missing={missing}, unexpected={unexpected}"
        )
    contract: dict[str, Any] = {}
    for key in sorted(expected_keys):
        field = reader.get_field(key)
        assert field is not None
        value = _normalise_value(field.contents())
        expected_type, expected_subtype = _metadata_type(
            key=key, value=value, gguf=gguf
        )
        expected_type_tuple = (
            (expected_type, expected_subtype)
            if expected_subtype is not None
            else (expected_type,)
        )
        if _field_type_tuple(field, gguf=gguf) != expected_type_tuple:
            raise ValueError(f"Source GGUF tq_* metadata has non-canonical type: {key}")
        contract[key] = value
    if int(contract["tq_schema_version"]) != 1:
        raise ValueError("Source GGUF tq_schema_version must remain 1")
    validate_turboquant_gguf_contract(contract, expected_len=num_layers)
    return contract


def _verify_source_contract(*, source: Any, model: Any, gguf: Any) -> None:
    if source.endianess != model.endianess:
        raise ValueError("Materialized GGUF endianness differs from source")
    if int(source.alignment) != int(model.alignment):
        raise ValueError("Materialized GGUF alignment differs from source")
    if _require_string(source, "general.architecture") != "llama":
        raise ValueError("Source GGUF architecture must be llama")
    source_file_type = _require_int(source, "general.file_type")
    if source_file_type != int(gguf.LlamaFileType.MOSTLY_Q4_0):
        raise ValueError("Source GGUF file type must be Q4_0")
    if _require_int(model, "general.file_type") != source_file_type:
        raise ValueError("Materialized GGUF file type differs from source")

    source_fields = {
        key: field
        for key, field in source.fields.items()
        if not key.startswith("GGUF.")
    }
    for key, source_field in source_fields.items():
        model_field = model.get_field(key)
        if model_field is None:
            raise ValueError(f"Materialized GGUF is missing source metadata: {key}")
        if _field_type_tuple(model_field, gguf=gguf) != _field_type_tuple(
            source_field, gguf=gguf
        ):
            raise ValueError(f"Materialized GGUF metadata type changed: {key}")
        if _normalise_value(model_field.contents()) != _normalise_value(
            source_field.contents()
        ):
            raise ValueError(f"Materialized GGUF metadata value changed: {key}")

    if len(model.tensors) < len(source.tensors):
        raise ValueError("Materialized GGUF is missing source tensors")
    for index, source_tensor in enumerate(source.tensors):
        model_tensor = model.tensors[index]
        if model_tensor.name != source_tensor.name:
            raise ValueError("Materialized GGUF source tensor order changed")
        if model_tensor.tensor_type != source_tensor.tensor_type:
            raise ValueError(
                f"Materialized GGUF tensor type changed: {source_tensor.name}"
            )
        if tuple(model_tensor.shape.tolist()) != tuple(source_tensor.shape.tolist()):
            raise ValueError(
                f"Materialized GGUF tensor shape changed: {source_tensor.name}"
            )
        if int(model_tensor.n_elements) != int(source_tensor.n_elements):
            raise ValueError(
                f"Materialized GGUF tensor element count changed: {source_tensor.name}"
            )
        if int(model_tensor.n_bytes) != int(source_tensor.n_bytes):
            raise ValueError(
                f"Materialized GGUF tensor byte count changed: {source_tensor.name}"
            )
        if np.asarray(model_tensor.data).dtype != np.asarray(source_tensor.data).dtype:
            raise ValueError(
                f"Materialized GGUF tensor storage dtype changed: {source_tensor.name}"
            )
        if _raw_tensor_sha256(model_tensor) != _raw_tensor_sha256(source_tensor):
            raise ValueError(
                f"Materialized GGUF tensor bytes changed: {source_tensor.name}"
            )


def _verify_live_payload(
    payload: dict[str, Any],
    *,
    head_dim: int,
    num_layers: int,
    num_kv_heads: int,
    source_sha256: str,
    source_size: int,
) -> None:
    if int(payload.get("schema_version", 0)) != TRIALITY_GGUF_SCHEMA_V2:
        raise ValueError("Live GGUF payload must use Triality schema v2")
    if payload.get("rotation_policy") != TRIALITY_IDENTITY_DEV_ROTATION_POLICY:
        raise ValueError("Live GGUF payload must use identity_dev rotations")
    if int(payload.get("rotation_seed", -1)) != 0:
        raise ValueError("Live GGUF identity rotations require rotation_seed=0")
    if (
        int(payload.get("head_dim", 0)) != head_dim
        or int(payload.get("num_layers", 0)) != num_layers
        or int(payload.get("num_kv_heads", 0)) != num_kv_heads
    ):
        raise ValueError("Live GGUF payload dimensions do not match source metadata")
    if bool(payload["ncka"]["enabled"]) or bool(payload["urt"]["enabled"]):
        raise ValueError("Live GGUF requires canonical disabled NC-KA and URT")
    weight_plan = payload.get("weight_plan")
    if not isinstance(weight_plan, dict) or weight_plan != {
        "codec": "tq4_1s",
        "enabled": False,
        "model_family": "llama",
        "modality_scope": "text-only",
        "policy": TRIALITY_LIVE_WEIGHT_POLICY,
        "protected_layers": [],
        "protected_roles": [],
        "schema": "hypura.turboquant.weight.v1",
        "source_ftype": "q4_0",
        "tensor_plan": {},
    }:
        raise ValueError("Live GGUF requires the canonical disabled Q4_0 weight plan")
    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("Live GGUF payload requires source_manifest")
    if source_manifest.get("sha256") != source_sha256:
        raise ValueError("Live GGUF source_manifest SHA256 mismatch")
    if int(source_manifest.get("size_bytes", -1)) != source_size:
        raise ValueError("Live GGUF source_manifest size mismatch")
    if source_manifest.get("generator") != _PRODUCER:
        raise ValueError("Live GGUF source_manifest generator mismatch")
    if (
        source_manifest.get("architecture") != "llama"
        or int(source_manifest.get("file_type", -1)) != 2
    ):
        raise ValueError("Live GGUF source_manifest model contract mismatch")
    if source_manifest.get("development_identity_views") is not True:
        raise ValueError("Live GGUF source_manifest lacks identity-view opt-in")
    if source_manifest.get("weight_conversion_disabled") is not True:
        raise ValueError("Live GGUF source_manifest lacks weight-conversion opt-out")
    _validate_timestamp(str(source_manifest.get("generated_at_utc", "")))


def _verify_metadata_bundle(
    *,
    source: Any,
    model: Any,
    strict_tq: dict[str, Any],
    public_metadata: dict[str, Any],
    gguf: Any,
) -> None:
    expected = {**strict_tq, **public_metadata}
    source_keys = {key for key in source.fields if not key.startswith("GGUF.")}
    model_keys = {key for key in model.fields if not key.startswith("GGUF.")}
    extra_keys = model_keys - source_keys
    if extra_keys != set(expected):
        missing = sorted(set(expected) - extra_keys)
        unexpected = sorted(extra_keys - set(expected))
        raise ValueError(
            f"Materialized GGUF metadata key set mismatch: missing={missing}, unexpected={unexpected}"
        )
    for key, expected_value in expected.items():
        field = model.get_field(key)
        if field is None:
            raise ValueError(f"Materialized GGUF is missing metadata: {key}")
        main_type, subtype = _metadata_type(key=key, value=expected_value, gguf=gguf)
        expected_types = (main_type, subtype) if subtype is not None else (main_type,)
        actual_types = _field_type_tuple(field, gguf=gguf)
        if actual_types != expected_types:
            raise ValueError(
                f"Materialized GGUF metadata has wrong type for {key}: "
                f"{actual_types} != {expected_types}"
            )
        actual = _normalise_value(field.contents())
        encoded_expected = _encoded_metadata_value(
            expected_value, main_type=main_type, subtype=subtype, gguf=gguf
        )
        if actual != encoded_expected:
            raise ValueError(
                f"Materialized GGUF metadata value mismatch for {key}: "
                f"{actual!r} != {encoded_expected!r}"
            )


def _verify_added_tensors(
    *, source: Any, model: Any, expected_tensors: dict[str, dict[str, Any]], gguf: Any
) -> None:
    added = model.tensors[len(source.tensors) :]
    expected_names = sorted(expected_tensors)
    if [tensor.name for tensor in added] != expected_names:
        raise ValueError("Materialized GGUF Triality tensor key set or order mismatch")
    for tensor in added:
        expected = expected_tensors[tensor.name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"Triality tensor must be F32: {tensor.name}")
        logical_shape = tuple(int(value) for value in expected["shape"])
        if tuple(int(value) for value in tensor.shape.tolist()) != logical_shape:
            raise ValueError(f"Triality tensor shape mismatch: {tensor.name}")
        values = np.asarray(tensor.data, dtype=np.float32).reshape(-1).tolist()
        if tensor_sha256(values) != str(payload_hash := _manifest_hash(expected)):
            raise ValueError(
                f"Triality tensor data hash mismatch: {tensor.name} "
                f"({_raw_tensor_sha256(tensor)}; expected logical {payload_hash})"
            )


def _manifest_hash(tensor: dict[str, Any]) -> str:
    return tensor_sha256(float(value) for value in tensor["data"])


def _validate_tensor_names(tensors: dict[str, dict[str, Any]]) -> None:
    for name in tensors:
        encoded_length = len(name.encode("utf-8"))
        if encoded_length >= _GGUF_MAX_TENSOR_NAME_BYTES:
            raise ValueError(
                f"Triality tensor name must be shorter than "
                f"{_GGUF_MAX_TENSOR_NAME_BYTES} bytes: {name!r} ({encoded_length})"
            )


def _encoded_metadata_value(
    value: Any, *, main_type: Any, subtype: Any | None, gguf: Any
) -> Any:
    if main_type == gguf.GGUFValueType.FLOAT32:
        return float(np.float32(value))
    if main_type == gguf.GGUFValueType.ARRAY:
        assert isinstance(value, list)
        if subtype == gguf.GGUFValueType.FLOAT32:
            return [float(np.float32(item)) for item in value]
        if subtype == gguf.GGUFValueType.UINT32:
            return [int(item) for item in value]
        if subtype == gguf.GGUFValueType.STRING:
            return [str(item) for item in value]
    return _normalise_value(value)


def _build_sidecar(
    *,
    summary: TrialityLiveGGUFSummary,
    source_reader: Any,
    model_path: Path,
    output_filename: str,
) -> dict[str, Any]:
    model_reader = import_vendor_gguf().GGUFReader(model_path)
    return {
        "manifest_version": TRIALITY_LIVE_MANIFEST_VERSION,
        "producer": _PRODUCER,
        "generated_at_utc": summary.generated_at_utc,
        "hash_order": [
            "source.file",
            "source.tensors_in_gguf_order",
            "output.file",
            "output.added_tensors_in_gguf_order",
        ],
        "source": {
            "filename": summary.source_path.name,
            "sha256": summary.source_sha256,
            "size_bytes": summary.source_path.stat().st_size,
            "architecture": summary.architecture,
            "file_type": summary.file_type,
            "alignment": int(source_reader.alignment),
            "endianness": source_reader.endianess.name.lower(),
            "tensors": [_tensor_record(tensor) for tensor in source_reader.tensors],
        },
        "output": {
            "filename": output_filename,
            "sha256": summary.output_sha256,
            "size_bytes": model_path.stat().st_size,
            "architecture": summary.architecture,
            "file_type": summary.file_type,
            "alignment": int(model_reader.alignment),
            "endianness": model_reader.endianess.name.lower(),
            "added_tensors": [
                _tensor_record(tensor)
                for tensor in model_reader.tensors[summary.source_tensor_count :]
            ],
        },
        "contract": {
            "profile_id": summary.profile_id,
            "tq_schema_version": 1,
            "schema_version": TRIALITY_GGUF_SCHEMA_V2,
            "rotation_policy": TRIALITY_IDENTITY_DEV_ROTATION_POLICY,
            "development_identity_views": True,
            "weight_conversion_enabled": False,
            "weight_policy": TRIALITY_LIVE_WEIGHT_POLICY,
            "ncka_enabled": False,
            "urt_enabled": False,
            "triality_override_allowed": False,
        },
    }


def _verify_sidecar(
    *, manifest_path: Path, summary: TrialityLiveGGUFSummary, source: Any, model: Any
) -> None:
    try:
        sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid Triality live manifest JSON: {manifest_path}"
        ) from exc
    expected = _build_sidecar(
        summary=summary,
        source_reader=source,
        model_path=summary.output_path,
        output_filename=summary.output_path.name,
    )
    if sidecar != expected:
        raise ValueError(
            "Triality live manifest does not match the verified GGUF artifacts"
        )


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "ggml_type": tensor.tensor_type.name,
        "shape": [int(value) for value in tensor.shape.tolist()],
        "n_elements": int(tensor.n_elements),
        "n_bytes": int(tensor.n_bytes),
        "storage_dtype": np.asarray(tensor.data).dtype.str,
        "sha256": _raw_tensor_sha256(tensor),
    }


def _field_type_tuple(field: Any, *, gguf: Any) -> tuple[Any, ...]:
    if not field.types:
        return ()
    main_type = field.types[0]
    subtype = _field_array_subtype(field=field, gguf=gguf)
    return (main_type, subtype) if subtype is not None else (main_type,)


def _field_array_subtype(*, field: Any, gguf: Any) -> Any | None:
    if not field.types or field.types[0] != gguf.GGUFValueType.ARRAY:
        return None
    if len(field.types) > 1:
        return field.types[-1]
    if len(field.parts) < 4:
        raise ValueError(f"GGUF array field lacks subtype storage: {field.name}")
    return gguf.GGUFValueType(int(field.parts[3][0]))


def _require_string(reader: Any, key: str) -> str:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"GGUF metadata is missing required string: {key}")
    value = field.contents()
    if not isinstance(value, str):
        raise ValueError(f"GGUF metadata {key} must be a string")
    return value


def _require_int(reader: Any, key: str) -> int:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"GGUF metadata is missing required integer: {key}")
    value = field.contents()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"GGUF metadata {key} must be an integer")
    return int(value)


def _require_positive_int(reader: Any, key: str) -> int:
    value = _require_int(reader, key)
    if value <= 0:
        raise ValueError(f"GGUF metadata {key} must be positive, got {value}")
    return value


def _infer_head_dim(reader: Any, architecture: str) -> int:
    key_field = reader.get_field(f"{architecture}.attention.key_length")
    if key_field is not None:
        value = int(key_field.contents())
        if value <= 0:
            raise ValueError("llama.attention.key_length must be positive")
        return value
    embedding = _require_positive_int(reader, f"{architecture}.embedding_length")
    heads = _require_positive_int(reader, f"{architecture}.attention.head_count")
    if embedding % heads != 0:
        raise ValueError("llama embedding length must be divisible by head count")
    return embedding // heads


def _infer_kv_heads(reader: Any, architecture: str) -> int:
    field = reader.get_field(f"{architecture}.attention.head_count_kv")
    if field is not None:
        value = int(field.contents())
        if value <= 0:
            raise ValueError("llama.attention.head_count_kv must be positive")
        return value
    return _require_positive_int(reader, f"{architecture}.attention.head_count")


def _raw_tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(np.asarray(tensor.data).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalise_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    return value


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def summary_dict(summary: TrialityLiveGGUFSummary) -> dict[str, Any]:
    """Return a JSON-compatible, stable summary mapping for CLI output."""

    payload = asdict(summary)
    for key in ("source_path", "output_path", "manifest_path"):
        value = payload[key]
        payload[key] = None if value is None else str(value)
    return payload


__all__ = [
    "TRIALITY_LIVE_DEFAULT_TIMESTAMP",
    "TRIALITY_LIVE_MANIFEST_VERSION",
    "TrialityLiveGGUFSummary",
    "manifest_path_for",
    "materialize_triality_live_gguf",
    "summary_dict",
    "verify_triality_live_gguf",
]
