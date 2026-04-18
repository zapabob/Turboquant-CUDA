from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.gguf_profiles import (
    build_hypura_serve_command,
    GGUF_HYPURA_COMPAT_AUTO,
    GGUF_TURBOQUANT_EXACT_PROFILE,
    build_paper_gguf_profile,
    build_so8_triality_vector_gguf_profile,
    import_vendor_gguf,
    infer_gguf_attention_head_dim,
    infer_gguf_block_count,
    package_turboquant_gguf,
    read_hypura_gguf_bridge_config,
    read_turboquant_gguf_manifest,
)
from turboquant.research_extension.k_triality import TrialityRotationArtifact, save_triality_proxy_rotations
from turboquant.schema import build_turboquant_artifact_metadata


def _write_toy_gguf(path: Path, *, include_turboquant_metadata: bool = False) -> np.ndarray:
    gguf = import_vendor_gguf()
    writer = gguf.GGUFWriter(path, arch="qwen35", use_temp_file=False)
    writer.add_name("toy-qwen")
    writer.add_uint32("general.file_type", 7)
    writer.add_uint32("qwen35.block_count", 2)
    writer.add_uint32("qwen35.embedding_length", 16)
    writer.add_uint32("qwen35.attention.head_count", 2)
    writer.add_uint32("qwen35.attention.key_length", 8)
    if include_turboquant_metadata:
        writer.add_uint32("turboquant.schema_version", 999)
    tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
    writer.add_tensor("token_embd.weight", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return tensor


def _write_toy_q8_gguf(path: Path) -> np.ndarray:
    gguf = import_vendor_gguf()
    writer = gguf.GGUFWriter(path, arch="qwen35", use_temp_file=False)
    writer.add_name("toy-qwen-q8")
    writer.add_uint32("general.file_type", 7)
    writer.add_uint32("qwen35.block_count", 2)
    writer.add_uint32("qwen35.embedding_length", 32)
    writer.add_uint32("qwen35.attention.head_count", 1)
    writer.add_uint32("qwen35.attention.key_length", 32)
    byte_tensor = np.arange(32 * 34, dtype=np.uint8).reshape(32, 34)
    writer.add_tensor("token_embd.weight", byte_tensor, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return byte_tensor


def _write_triality_rotation_dir(path: Path, *, bits_total: float = 3.5) -> None:
    allocation = ChannelBitAllocation.preset(effective_bits=bits_total - 1.0, width=8)
    vector_metadata = build_turboquant_artifact_metadata(
        total_bits=bits_total,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=8,
        allocation=allocation,
    )
    plus_metadata = build_turboquant_artifact_metadata(
        total_bits=bits_total,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=19,
        qjl_seed=73,
        triality_mode="triality_proxy",
        triality_view="spinor_plus_proxy",
        width=8,
        allocation=allocation,
    )
    save_triality_proxy_rotations(
        [
            TrialityRotationArtifact(
                layer_idx=0,
                bits=bits_total,
                view="vector",
                rotation=torch.eye(8, dtype=torch.float32),
                rotation_seed=17,
                qjl_seed=71,
                metadata=vector_metadata,
            ),
            TrialityRotationArtifact(
                layer_idx=1,
                bits=bits_total,
                view="vector",
                rotation=(2.0 * torch.eye(8, dtype=torch.float32)),
                rotation_seed=17,
                qjl_seed=71,
                metadata=vector_metadata,
            ),
            TrialityRotationArtifact(
                layer_idx=0,
                bits=bits_total,
                view="spinor_plus_proxy",
                rotation=(3.0 * torch.eye(8, dtype=torch.float32)),
                rotation_seed=19,
                qjl_seed=73,
                metadata=plus_metadata,
            ),
        ],
        path,
    )


def test_infer_gguf_head_dim_and_block_count(tmp_path: Path) -> None:
    source = tmp_path / "toy.gguf"
    _write_toy_gguf(source)
    assert infer_gguf_attention_head_dim(source) == 8
    assert infer_gguf_block_count(source) == 2


def test_package_turboquant_gguf_embeds_profiles_and_preserves_original_tensor(tmp_path: Path) -> None:
    source = tmp_path / "toy.gguf"
    original_tensor = _write_toy_gguf(source)
    rotation_dir = tmp_path / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    _write_triality_rotation_dir(rotation_dir)

    output = tmp_path / "toy.turboquant.gguf"
    manifest = package_turboquant_gguf(
        source_path=source,
        output_path=output,
        profiles=[
            build_paper_gguf_profile(bits_total=3.5, head_dim=8),
            build_so8_triality_vector_gguf_profile(
                rotation_dir=rotation_dir,
                bits_total=3.5,
                expected_head_dim=8,
                expected_block_count=2,
            ),
        ],
        default_profile=GGUF_TURBOQUANT_EXACT_PROFILE,
    )

    assert manifest.default_profile == "exact"
    assert set(manifest.profiles.keys()) == {"paper", "so8_triality_vector"}
    assert manifest.profiles["paper"].metadata["bits_total"] == 3.5
    assert manifest.profiles["so8_triality_vector"].metadata["layer_indices"] == [0, 1]
    tensor_names = manifest.profiles["so8_triality_vector"].metadata["artifact_tensor_names"]
    assert tensor_names == [
        "tq.p.so8_triality_vector.l00.b3p5.rot",
        "tq.p.so8_triality_vector.l01.b3p5.rot",
    ]
    assert all(len(name) <= 63 for name in tensor_names)

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(output)
    restored = next(tensor for tensor in reader.tensors if tensor.name == "token_embd.weight")
    np.testing.assert_array_equal(np.array(restored.data, copy=False), original_tensor)
    embedded_rotation = next(
        tensor for tensor in reader.tensors if tensor.name == "tq.p.so8_triality_vector.l01.b3p5.rot"
    )
    np.testing.assert_array_equal(np.array(embedded_rotation.data, copy=False), 2.0 * np.eye(8, dtype=np.float32))


def test_read_turboquant_manifest_round_trip_returns_embedded_rotation_tensors(tmp_path: Path) -> None:
    source = tmp_path / "toy.gguf"
    _write_toy_gguf(source)
    rotation_dir = tmp_path / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    _write_triality_rotation_dir(rotation_dir, bits_total=2.5)
    output = tmp_path / "toy.roundtrip.gguf"

    package_turboquant_gguf(
        source_path=source,
        output_path=output,
        profiles=[
            build_paper_gguf_profile(bits_total=2.5, head_dim=8),
            build_so8_triality_vector_gguf_profile(
                rotation_dir=rotation_dir,
                bits_total=2.5,
                expected_head_dim=8,
                expected_block_count=2,
            ),
        ],
    )
    manifest = read_turboquant_gguf_manifest(output)
    profile = manifest.profiles["so8_triality_vector"]
    assert profile.runtime_mode == "key_only_block_so8_triality_vector"
    assert profile.manifest["bits_total"] == 2.5
    assert len(profile.tensors) == 2
    np.testing.assert_array_equal(profile.tensors[0].data, np.eye(8, dtype=np.float32))


def test_package_embeds_hypura_bridge_metadata_for_triality_profile(tmp_path: Path) -> None:
    source = tmp_path / "toy.gguf"
    _write_toy_gguf(source)
    rotation_dir = tmp_path / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    _write_triality_rotation_dir(rotation_dir, bits_total=3.5)
    output = tmp_path / "toy.hypura.gguf"

    package_turboquant_gguf(
        source_path=source,
        output_path=output,
        profiles=[
            build_paper_gguf_profile(bits_total=3.5, head_dim=8),
            build_so8_triality_vector_gguf_profile(
                rotation_dir=rotation_dir,
                bits_total=3.5,
                expected_head_dim=8,
                expected_block_count=2,
            ),
        ],
        hypura_compatibility_profile=GGUF_HYPURA_COMPAT_AUTO,
    )

    manifest = read_turboquant_gguf_manifest(output)
    strict_contract = manifest.profiles["so8_triality_vector"].manifest["strict_gguf_contract"]
    assert strict_contract["tq_total_bits"] == [3.5, 3.5]
    assert strict_contract["tq_triality_mode"] == ["triality_proxy", "triality_proxy"]
    assert strict_contract["tq_triality_view"] == ["vector", "vector"]

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(output)
    assert int(reader.get_field("tq_schema_version").contents()) == 1
    assert list(reader.get_field("tq_qjl_bits").contents()) == [1, 1]
    assert list(reader.get_field("tq_triality_mode").contents()) == ["triality_proxy", "triality_proxy"]
    assert list(reader.get_field("tq_triality_view").contents()) == ["vector", "vector"]
    assert str(reader.get_field("hypura.turboquant.mode").contents()) == "triality-proxy-so8-pareto"

    bridge = read_hypura_gguf_bridge_config(output)
    assert bridge is not None
    assert bridge.source_profile == "so8_triality_vector"
    assert bridge.mode == "research-kv-split"
    assert bridge.rotation_policy == "triality_vector"
    assert bridge.triality_view == "vector"
    assert bridge.rotation_seed == 17

    command = build_hypura_serve_command(
        gguf_path=output,
        host="127.0.0.1",
        port=5001,
        context=4096,
        turboquant_mode="gguf-auto",
        release=True,
    )
    assert command == [
        "cargo",
        "run",
        "--release",
        "-p",
        "hypura",
        "--",
        "serve",
        str(output),
        "--host",
        "127.0.0.1",
        "--port",
        "5001",
        "--context",
        "4096",
        "--turboquant-mode",
        "research-kv-split",
    ]


def test_package_preserves_quantized_q8_tensor_bytes_and_type(tmp_path: Path) -> None:
    source = tmp_path / "toy-q8.gguf"
    original_q8_bytes = _write_toy_q8_gguf(source)
    output = tmp_path / "toy-q8.turboquant.gguf"

    package_turboquant_gguf(
        source_path=source,
        output_path=output,
        profiles=[build_paper_gguf_profile(bits_total=3.5, head_dim=32)],
    )

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(output)
    restored = next(tensor for tensor in reader.tensors if tensor.name == "token_embd.weight")
    assert restored.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert tuple(restored.shape.tolist()) == (32, 32)
    np.testing.assert_array_equal(np.array(restored.data, copy=False), original_q8_bytes)


def test_hypura_bridge_rejects_embedded_paper_profile(tmp_path: Path) -> None:
    source = tmp_path / "toy.gguf"
    _write_toy_gguf(source)
    output = tmp_path / "toy-paper-only.gguf"

    with pytest.raises(ValueError, match="Paper-faithful profiles still require a parsed paper sidecar"):
        package_turboquant_gguf(
            source_path=source,
            output_path=output,
            profiles=[build_paper_gguf_profile(bits_total=3.5, head_dim=8)],
            hypura_compatibility_profile="paper",
        )


def test_package_rejects_existing_turboquant_namespace(tmp_path: Path) -> None:
    source = tmp_path / "already-tagged.gguf"
    _write_toy_gguf(source, include_turboquant_metadata=True)
    output = tmp_path / "repacked.gguf"
    with pytest.raises(ValueError, match="already contains embedded turboquant metadata"):
        package_turboquant_gguf(
            source_path=source,
            output_path=output,
            profiles=[build_paper_gguf_profile(bits_total=3.5, head_dim=8)],
        )
