from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.materialize_triality_live_gguf import main as materialize_main
import turboquant.triality_live_gguf as live_gguf
from turboquant.gguf_profiles import import_vendor_gguf
from turboquant.triality_contract import (
    build_default_weight_plan,
    build_triality_payload,
    validate_weight_plan,
)
from turboquant.triality_live_gguf import (
    materialize_triality_live_gguf,
    verify_triality_live_gguf,
)
from turboquant.triality_schema_v2 import (
    TRIALITY_CONSENSUS_VIEWS,
    rotation_tensor_name,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_toy_q4(
    path: Path,
    *,
    architecture: str = "llama",
    file_type: int | None = None,
    existing_namespace: bool = False,
    existing_tq_schema: int | None = None,
    big_endian: bool = False,
) -> None:
    gguf = import_vendor_gguf()
    endianess = gguf.GGUFEndian.BIG if big_endian else gguf.GGUFEndian.LITTLE
    writer = gguf.GGUFWriter(
        path,
        arch=architecture,
        use_temp_file=False,
        endianess=endianess,
    )
    writer.add_name("toy-live-q4")
    writer.add_file_type(
        int(gguf.LlamaFileType.MOSTLY_Q4_0) if file_type is None else file_type
    )
    writer.add_custom_alignment(64)
    writer.add_uint32(f"{architecture}.block_count", 2)
    writer.add_uint32(f"{architecture}.embedding_length", 32)
    writer.add_uint32(f"{architecture}.attention.head_count", 2)
    writer.add_uint32(f"{architecture}.attention.head_count_kv", 2)
    writer.add_uint32(f"{architecture}.attention.key_length", 16)
    writer.add_key_value("toy.uint16", 7, gguf.GGUFValueType.UINT16)
    writer.add_key_value(
        "toy.uint16_array",
        [1, 2, 65535],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT16,
    )
    if existing_namespace:
        writer.add_uint32("tq_schema_version", 1)
    if existing_tq_schema is not None:
        writer.add_uint32("tq_schema_version", existing_tq_schema)
        for key, values in {
            "tq_total_bits": [4.0, 4.0],
            "tq_runtime_bits_per_channel": [4.0, 4.0],
            "tq_stage1_effective_bits": [3.0, 3.0],
        }.items():
            writer.add_key_value(
                key,
                values,
                gguf.GGUFValueType.ARRAY,
                gguf.GGUFValueType.FLOAT32,
            )
        for key, values in {
            "tq_qjl_bits": [1, 1],
            "tq_qjl_dim": [16, 16],
            "tq_rotation_seed": [9, 9],
            "tq_qjl_seed": [11, 11],
        }.items():
            writer.add_key_value(
                key,
                values,
                gguf.GGUFValueType.ARRAY,
                gguf.GGUFValueType.UINT32,
            )
        for key, values in {
            "tq_rotation_policy": ["random_haar", "random_haar"],
            "tq_triality_mode": ["paper-key-only", "paper-key-only"],
            "tq_triality_view": ["none", "none"],
            "tq_stage1_allocation_scheme": ["uniform", "uniform"],
            "tq_stage1_bitwidth_payload_dtype": ["uint8", "uint8"],
            "tq_norm_dtype": ["float32", "float32"],
            "tq_sign_pack_format": [
                "int8_unpacked_binary",
                "int8_unpacked_binary",
            ],
        }.items():
            writer.add_key_value(
                key,
                values,
                gguf.GGUFValueType.ARRAY,
                gguf.GGUFValueType.STRING,
            )
    source_float = np.linspace(-1.0, 1.0, num=64, dtype=np.float32).reshape(2, 32)
    q4 = gguf.quantize(source_float, gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor(
        "blk.0.attn_q.weight",
        q4,
        raw_dtype=gguf.GGMLQuantizationType.Q4_0,
    )
    writer.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _materialize(source: Path, output: Path):
    return materialize_triality_live_gguf(
        source_path=source,
        output_path=output,
        generated_at_utc="2026-07-13T00:00:00+00:00",
        development_identity_views=True,
        disable_weight_conversion=True,
    )


@pytest.mark.parametrize("big_endian", [False, True])
def test_materialize_preserves_q4_source_and_adds_complete_identity_bundle(
    tmp_path: Path, big_endian: bool
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    _write_toy_q4(source, big_endian=big_endian)
    source_hash = _sha256(source)
    gguf = import_vendor_gguf()
    source_reader = gguf.GGUFReader(source)
    source_tensor_records = [
        (
            tensor.name,
            tensor.tensor_type,
            tuple(tensor.shape.tolist()),
            int(tensor.n_bytes),
            np.asarray(tensor.data).dtype,
            hashlib.sha256(np.asarray(tensor.data).tobytes()).hexdigest(),
        )
        for tensor in source_reader.tensors
    ]

    summary = _materialize(source, output)
    verified = verify_triality_live_gguf(source_path=source, model_path=output)
    assert verified == summary
    assert _sha256(source) == source_hash
    assert summary.file_type == int(gguf.LlamaFileType.MOSTLY_Q4_0)
    assert summary.alignment == 64
    assert summary.endianness == ("big" if big_endian else "little")
    assert summary.added_tensor_count == 3 * 2 + 4

    model = gguf.GGUFReader(output)
    assert model.get_field("general.file_type").contents() == 2
    assert model.get_field("general.alignment").contents() == 64
    assert tuple(model.get_field("toy.uint16").types) == (gguf.GGUFValueType.UINT16,)
    assert tuple(model.get_field("toy.uint16_array").types) == (
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT16,
    )
    model_source_records = [
        (
            tensor.name,
            tensor.tensor_type,
            tuple(tensor.shape.tolist()),
            int(tensor.n_bytes),
            np.asarray(tensor.data).dtype,
            hashlib.sha256(np.asarray(tensor.data).tobytes()).hexdigest(),
        )
        for tensor in model.tensors[: len(source_reader.tensors)]
    ]
    assert model_source_records == source_tensor_records

    for layer in range(2):
        for view in TRIALITY_CONSENSUS_VIEWS:
            tensor = next(
                tensor
                for tensor in model.tensors
                if tensor.name == rotation_tensor_name(layer, view, "liveq4")
            )
            assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
            np.testing.assert_array_equal(
                np.asarray(tensor.data, dtype=np.float32),
                np.eye(16, dtype=np.float32),
            )

    override = model.get_field("hypura.turboquant.triality.override_allowed")
    assert override.contents() is False
    assert tuple(override.types) == (gguf.GGUFValueType.BOOL,)
    assert model.get_field("hypura.turboquant.triality_override_allowed") is None
    ncka_coordinates = model.get_field("hypura.turboquant.ncka.coordinate_names")
    assert ncka_coordinates.contents() == []
    assert (
        gguf.GGUFValueType(int(ncka_coordinates.parts[3][0]))
        == gguf.GGUFValueType.STRING
    )
    payload = json.loads(model.get_field("hypura.turboquant.payload_json").contents())
    assert model.get_field("tq_schema_version").contents() == 1
    assert tuple(model.get_field("tq_schema_version").types) == (
        gguf.GGUFValueType.UINT32,
    )
    assert model.get_field("hypura.turboquant.schema_version").contents() == 2
    assert model.get_field("tq_schema_version").contents() != 2
    assert payload["ncka"]["enabled"] is False
    assert payload["urt"]["enabled"] is False
    assert payload["weight_plan"]["enabled"] is False
    assert payload["weight_plan"]["source_ftype"] == "q4_0"
    assert payload["weight_plan"]["tensor_plan"] == {}


def test_fixed_timestamp_produces_byte_identical_ggufs(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    _write_toy_q4(source)
    _materialize(source, first)
    _materialize(source, second)
    assert first.read_bytes() == second.read_bytes()


def test_materializer_rejects_triality_tensor_names_at_gguf_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.gguf"
    _write_toy_q4(source)
    with pytest.raises(ValueError, match="must be shorter than 64 bytes"):
        materialize_triality_live_gguf(
            source_path=source,
            output_path=tmp_path / "output.gguf",
            profile_id="stories260k",
            development_identity_views=True,
            disable_weight_conversion=True,
        )


@pytest.mark.parametrize(
    ("identity_opt_in", "weight_opt_out"),
    [(False, False), (True, False), (False, True)],
)
def test_materializer_requires_both_explicit_opt_ins(
    tmp_path: Path, identity_opt_in: bool, weight_opt_out: bool
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    _write_toy_q4(source)
    with pytest.raises(PermissionError, match="requires both"):
        materialize_triality_live_gguf(
            source_path=source,
            output_path=output,
            development_identity_views=identity_opt_in,
            disable_weight_conversion=weight_opt_out,
        )
    assert not output.exists()


def test_materializer_cli_requires_both_explicit_flags(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    _write_toy_q4(source)
    with pytest.raises(PermissionError, match="requires both"):
        materialize_main(["--input-gguf", str(source), "--output-gguf", str(output)])


def test_materializer_rejects_same_path_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    _write_toy_q4(source)
    with pytest.raises(ValueError, match="must differ"):
        _materialize(source, source)
    output.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        _materialize(source, output)
    assert output.read_bytes() == b"do-not-overwrite"


@pytest.mark.parametrize("race_destination", ["output", "sidecar"])
def test_materializer_publish_race_never_overwrites_or_crosses_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race_destination: str
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    sidecar = tmp_path / "output.gguf.manifest.json"
    _write_toy_q4(source)
    raced_path = output if race_destination == "output" else sidecar
    sentinel = f"external-{race_destination}".encode()
    original_link = live_gguf.os.link

    def link_with_race(source_path, destination_path, *args, **kwargs):
        destination = Path(destination_path)
        if destination == raced_path and not destination.exists():
            destination.write_bytes(sentinel)
        return original_link(source_path, destination_path, *args, **kwargs)

    monkeypatch.setattr(live_gguf.os, "link", link_with_race)
    with pytest.raises(FileExistsError, match="destination already exists"):
        _materialize(source, output)

    assert raced_path.read_bytes() == sentinel
    counterpart = sidecar if race_destination == "output" else output
    assert not counterpart.exists()
    assert not (tmp_path / ".output.gguf.triality-live.lock").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_materializer_rollback_removes_only_invocation_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    sidecar = tmp_path / "output.gguf.manifest.json"
    _write_toy_q4(source)
    original_link_no_replace = live_gguf._link_no_replace

    def publish_with_foreign_replacement(
        source_path: Path, destination_path: Path
    ) -> tuple[int, int]:
        if destination_path == output:
            invocation_identity = original_link_no_replace(
                source_path, destination_path
            )
            destination_path.unlink()
            destination_path.write_bytes(b"external-output")
            return invocation_identity
        destination_path.write_bytes(b"external-sidecar")
        return original_link_no_replace(source_path, destination_path)

    monkeypatch.setattr(live_gguf, "_link_no_replace", publish_with_foreign_replacement)
    with pytest.raises(FileExistsError, match="destination already exists"):
        _materialize(source, output)

    assert output.read_bytes() == b"external-output"
    assert sidecar.read_bytes() == b"external-sidecar"
    assert not (tmp_path / ".output.gguf.triality-live.lock").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_materializer_rejects_non_llama_non_q4_and_existing_namespace(
    tmp_path: Path,
) -> None:
    gguf = import_vendor_gguf()
    wrong_arch = tmp_path / "wrong-arch.gguf"
    wrong_type = tmp_path / "wrong-type.gguf"
    existing = tmp_path / "existing.gguf"
    _write_toy_q4(wrong_arch, architecture="qwen35")
    _write_toy_q4(wrong_type, file_type=int(gguf.LlamaFileType.MOSTLY_Q8_0))
    _write_toy_q4(existing, existing_namespace=True)
    with pytest.raises(ValueError, match="architecture 'llama'"):
        _materialize(wrong_arch, tmp_path / "wrong-arch-out.gguf")
    with pytest.raises(ValueError, match="requires general.file_type Q4_0"):
        _materialize(wrong_type, tmp_path / "wrong-type-out.gguf")
    with pytest.raises(ValueError, match="partial or non-canonical"):
        _materialize(existing, tmp_path / "existing-out.gguf")


def test_materializer_preserves_existing_canonical_tq_v1_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tq-v1.gguf"
    output = tmp_path / "output-tq-v1.gguf"
    _write_toy_q4(source, existing_tq_schema=1)
    gguf = import_vendor_gguf()
    source_reader = gguf.GGUFReader(source)
    source_tq = {
        key: (
            tuple(field.types),
            field.contents(),
        )
        for key, field in source_reader.fields.items()
        if key.startswith("tq_")
    }
    _materialize(source, output)
    model = gguf.GGUFReader(output)
    output_tq = {
        key: (
            tuple(field.types),
            field.contents(),
        )
        for key, field in model.fields.items()
        if key.startswith("tq_")
    }
    assert output_tq == source_tq
    assert model.get_field("tq_schema_version").contents() == 1
    assert model.get_field("hypura.turboquant.schema_version").contents() == 2


def test_materializer_rejects_tq_schema_v2_in_source_namespace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tq-v2.gguf"
    _write_toy_q4(source, existing_tq_schema=2)
    with pytest.raises(ValueError, match="must remain 1"):
        _materialize(source, tmp_path / "output.gguf")


def test_disabled_weight_plan_is_q4_only_and_canonical() -> None:
    plan = build_default_weight_plan(
        model_family="llama",
        num_layers=2,
        source_ftype="q4_0",
        enabled=False,
    )
    validate_weight_plan(plan, model_family="llama", num_layers=2)
    assert plan["policy"] == "preserve-source-weights"
    assert plan["tensor_plan"] == {}
    with pytest.raises(ValueError, match="only valid for q4_0"):
        build_default_weight_plan(
            model_family="llama",
            num_layers=2,
            source_ftype="q8_0",
            enabled=False,
        )
    with pytest.raises(ValueError, match="require disabled"):
        build_default_weight_plan(
            model_family="llama",
            num_layers=2,
            source_ftype="q4_0",
            enabled=True,
        )


def test_identity_policy_builds_all_three_views_as_identity() -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
        model_family="llama",
        weight_source_ftype="q4_0",
        weight_enabled=False,
        rotation_policy="identity_dev",
        schema_version=2,
        source_manifest={"development_identity_views": True},
    )
    assert payload["rotation_policy"] == "identity_dev"
    assert len(payload["tensor_manifest"]) == 7


@pytest.mark.parametrize("tensor_kind", ["source", "rotation"])
def test_verifier_rejects_corrupted_tensor_bytes(
    tmp_path: Path, tensor_kind: str
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    corrupt = tmp_path / f"corrupt-{tensor_kind}.gguf"
    _write_toy_q4(source)
    summary = _materialize(source, output)
    shutil.copyfile(output, corrupt)
    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(corrupt)
    if tensor_kind == "source":
        target = reader.tensors[0]
    else:
        target = next(
            tensor
            for tensor in reader.tensors
            if tensor.name == rotation_tensor_name(0, "vector", "liveq4")
        )
    offset = int(target.data_offset)
    del reader
    with corrupt.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([original[0] ^ 0x01]))
    expected_error = "source_manifest SHA256 mismatch|tensor bytes changed"
    if tensor_kind == "rotation":
        expected_error = "tensor data hash mismatch"
    with pytest.raises(ValueError, match=expected_error):
        verify_triality_live_gguf(
            source_path=source,
            model_path=corrupt,
            manifest_path=summary.manifest_path,
        )
