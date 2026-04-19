from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from turboquant.gguf_profiles import import_vendor_gguf
from turboquant.weight_gguf import convert_weight_turboquant_gguf


_REAL_GEMMA4_Q8_0_GGUF = Path(
    r"C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf"
)

_TQ4_1S_CENTROIDS = np.array(
    [
        -2.732590,
        -2.069017,
        -1.618046,
        -1.256231,
        -0.942340,
        -0.656759,
        -0.388048,
        -0.128395,
        0.128395,
        0.388048,
        0.656759,
        0.942340,
        1.256231,
        1.618046,
        2.069017,
        2.732590,
    ],
    dtype=np.float32,
)
_TQ4_1S_MIDPOINTS = np.array(
    [
        -2.400804,
        -1.843532,
        -1.437139,
        -1.099286,
        -0.799550,
        -0.522404,
        -0.258222,
        0.0,
        0.258222,
        0.522404,
        0.799550,
        1.099286,
        1.437139,
        1.843532,
        2.400804,
    ],
    dtype=np.float32,
)
_TQ4_1S_SIGNS = np.array(
    [
        +1.0,
        -1.0,
        +1.0,
        -1.0,
        +1.0,
        +1.0,
        -1.0,
        +1.0,
        -1.0,
        -1.0,
        +1.0,
        -1.0,
        +1.0,
        +1.0,
        -1.0,
        +1.0,
        -1.0,
        -1.0,
        +1.0,
        -1.0,
        +1.0,
        -1.0,
        -1.0,
        +1.0,
        -1.0,
        +1.0,
        +1.0,
        -1.0,
        +1.0,
        -1.0,
        -1.0,
        +1.0,
    ],
    dtype=np.float32,
)
_TQ4_1S_SCALE_CANDIDATES = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5], dtype=np.float32)
_TQ4_1S_INV_SQRT32 = np.float32(1.0 / np.sqrt(32.0))


def _fwht32(rows: np.ndarray) -> np.ndarray:
    out = np.array(rows, dtype=np.float32, copy=True)
    step = 1
    while step < 32:
        span = step << 1
        reshaped = out.reshape(out.shape[0], -1, span)
        left = reshaped[:, :, :step].copy()
        right = reshaped[:, :, step:].copy()
        reshaped[:, :, :step] = left + right
        reshaped[:, :, step:] = left - right
        step = span
    return out


def _tq4_1s_rht_forward(rows: np.ndarray) -> np.ndarray:
    signed = np.array(rows, dtype=np.float32, copy=True) * _TQ4_1S_SIGNS.reshape(1, 32)
    return _fwht32(signed) * _TQ4_1S_INV_SQRT32


def _tq4_1s_rht_inverse(rows: np.ndarray) -> np.ndarray:
    transformed = _fwht32(rows) * _TQ4_1S_INV_SQRT32
    return transformed * _TQ4_1S_SIGNS.reshape(1, 32)


def _tq4_1s_choose_indices(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(_TQ4_1S_MIDPOINTS, values, side="right").astype(np.uint8)


def _reference_quantize_tq4_1s(rows: np.ndarray) -> np.ndarray:
    source = np.array(rows, dtype=np.float32, copy=False)
    logical_shape = source.shape
    blocks = source.reshape(-1, 32)
    rotated = _tq4_1s_rht_forward(blocks)
    half0 = rotated[:, :16]
    half1 = rotated[:, 16:]

    rms0 = np.sqrt(np.mean(np.square(half0), axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    rms1 = np.sqrt(np.mean(np.square(half1), axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)

    best_d0 = rms0.copy()
    best_d1 = rms1.copy()
    best_err = np.full((blocks.shape[0], 1), np.inf, dtype=np.float32)

    for scale in _TQ4_1S_SCALE_CANDIDATES:
        d0 = rms0 * scale
        d1 = rms1 * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            inv0 = np.where(d0 > 1.0e-10, np.float32(1.0) / d0, np.float32(0.0))
            inv1 = np.where(d1 > 1.0e-10, np.float32(1.0) / d1, np.float32(0.0))
        idx0 = _tq4_1s_choose_indices(half0 * inv0)
        idx1 = _tq4_1s_choose_indices(half1 * inv1)
        recon0 = _TQ4_1S_CENTROIDS[idx0] * d0
        recon1 = _TQ4_1S_CENTROIDS[idx1] * d1
        err = (
            np.sum(np.square(half0 - recon0), axis=1, keepdims=True, dtype=np.float32)
            + np.sum(np.square(half1 - recon1), axis=1, keepdims=True, dtype=np.float32)
        )
        use_candidate = err < best_err
        best_err = np.where(use_candidate, err, best_err)
        best_d0 = np.where(use_candidate, d0, best_d0)
        best_d1 = np.where(use_candidate, d1, best_d1)

    for _ in range(6):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv0 = np.where(best_d0 > 1.0e-10, np.float32(1.0) / best_d0, np.float32(0.0))
            inv1 = np.where(best_d1 > 1.0e-10, np.float32(1.0) / best_d1, np.float32(0.0))
        idx0 = _tq4_1s_choose_indices(half0 * inv0)
        idx1 = _tq4_1s_choose_indices(half1 * inv1)
        c0 = _TQ4_1S_CENTROIDS[idx0]
        c1 = _TQ4_1S_CENTROIDS[idx1]
        num0 = np.sum(half0 * c0, axis=1, keepdims=True, dtype=np.float32)
        den0 = np.sum(np.square(c0), axis=1, keepdims=True, dtype=np.float32)
        num1 = np.sum(half1 * c1, axis=1, keepdims=True, dtype=np.float32)
        den1 = np.sum(np.square(c1), axis=1, keepdims=True, dtype=np.float32)
        best_d0 = np.where(den0 > 1.0e-10, num0 / den0, best_d0)
        best_d1 = np.where(den1 > 1.0e-10, num1 / den1, best_d1)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv0 = np.where(best_d0 > 1.0e-10, np.float32(1.0) / best_d0, np.float32(0.0))
        inv1 = np.where(best_d1 > 1.0e-10, np.float32(1.0) / best_d1, np.float32(0.0))
    idx0 = _tq4_1s_choose_indices(half0 * inv0)
    idx1 = _tq4_1s_choose_indices(half1 * inv1)

    indices = np.empty((blocks.shape[0], 32), dtype=np.uint8)
    indices[:, :16] = idx0
    indices[:, 16:] = idx1
    packed_qs = indices[:, 0::2] | (indices[:, 1::2] << np.uint8(4))

    d0 = np.ascontiguousarray(best_d0.astype(np.float16)).view(np.uint8).reshape(blocks.shape[0], 2)
    d1 = np.ascontiguousarray(best_d1.astype(np.float16)).view(np.uint8).reshape(blocks.shape[0], 2)
    packed = np.concatenate([d0, d1, packed_qs], axis=1)
    return packed.reshape(*logical_shape[:-1], logical_shape[-1] // 32 * 20)


def _reference_dequantize_tq4_1s(packed_rows: np.ndarray) -> np.ndarray:
    packed = np.array(packed_rows, dtype=np.uint8, copy=False)
    packed_shape = packed.shape
    blocks = packed.reshape(-1, 20)
    d0 = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1, 1)
    d1 = np.ascontiguousarray(blocks[:, 2:4]).view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = blocks[:, 4:]

    indices = np.empty((blocks.shape[0], 32), dtype=np.uint8)
    indices[:, 0::2] = qs & np.uint8(0x0F)
    indices[:, 1::2] = (qs >> np.uint8(4)) & np.uint8(0x0F)

    rotated = _TQ4_1S_CENTROIDS[indices].astype(np.float32)
    rotated[:, :16] *= d0
    rotated[:, 16:] *= d1
    dequantized = _tq4_1s_rht_inverse(rotated)
    return dequantized.reshape(*packed_shape[:-1], packed_shape[-1] // 20 * 32)


def _real_model_output_dir() -> Path:
    required_free_bytes = _REAL_GEMMA4_Q8_0_GGUF.stat().st_size + (512 * 1024 * 1024)
    candidates = [Path("H:/"), Path("F:/"), _REAL_GEMMA4_Q8_0_GGUF.parent]
    for candidate in candidates:
        if not candidate.exists():
            continue
        free_bytes = shutil.disk_usage(candidate).free
        if free_bytes >= required_free_bytes:
            return candidate
    raise RuntimeError(
        "No filesystem with enough free space for the real Gemma4 GGUF conversion "
        f"(need at least {required_free_bytes} bytes)"
    )


def _find_existing_real_model_output() -> Path | None:
    output_name = "supergemma4-Q8_0.tq4_1s.gguf"
    candidates: list[Path] = []
    for root in (Path("H:/"), Path("F:/"), _REAL_GEMMA4_Q8_0_GGUF.parent):
        if not root.exists():
            continue
        for candidate_dir in root.glob("turboquant-real-*"):
            candidate = candidate_dir / output_name
            if candidate.exists() and candidate.stat().st_size > 0:
                candidates.append(candidate)
    gguf = import_vendor_gguf()
    for candidate in sorted(candidates, key=lambda path: path.stat().st_size, reverse=True):
        try:
            gguf.GGUFReader(candidate)
        except Exception:
            continue
        return candidate
    return None


def _write_toy_weight_q8_gguf(path: Path) -> dict[str, np.ndarray]:
    gguf = import_vendor_gguf()
    writer = gguf.GGUFWriter(path, arch="qwen35", use_temp_file=False)
    writer.add_name("toy-qwen35-9b")
    writer.add_uint32("general.file_type", 7)
    writer.add_uint32("qwen35.block_count", 8)
    writer.add_uint32("qwen35.embedding_length", 64)
    writer.add_uint32("qwen35.attention.head_count", 2)
    writer.add_uint32("qwen35.attention.head_count_kv", 2)
    writer.add_uint32("qwen35.attention.key_length", 32)

    converted_float = np.linspace(-1.0, 1.0, num=64, dtype=np.float32).reshape(2, 32)
    protected_float = np.linspace(-0.5, 0.5, num=64, dtype=np.float32).reshape(2, 32)
    converted_q8 = gguf.quantize(converted_float, gguf.GGMLQuantizationType.Q8_0)
    protected_q8 = gguf.quantize(protected_float, gguf.GGMLQuantizationType.Q8_0)

    writer.add_tensor("blk.2.attn_q.weight", converted_q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.1.attn_q.weight", protected_q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return {
        "converted_q8": converted_q8,
        "protected_q8": protected_q8,
    }


def _write_toy_gemma4_weight_q8_gguf(path: Path) -> dict[str, np.ndarray]:
    gguf = import_vendor_gguf()
    writer = gguf.GGUFWriter(path, arch="gemma4", use_temp_file=False)
    writer.add_name("toy-gemma4-e4b")
    writer.add_uint32("general.file_type", 7)
    writer.add_uint32("gemma4.block_count", 42)
    writer.add_uint32("gemma4.embedding_length", 128)
    writer.add_uint32("gemma4.embedding_length_per_layer_input", 32)
    writer.add_uint32("gemma4.attention.head_count", 2)
    writer.add_uint32("gemma4.attention.head_count_kv", 1)
    writer.add_uint32("gemma4.attention.key_length", 32)
    writer.add_uint32("gemma4.attention.shared_kv_layers", 18)

    attn_float = np.linspace(-1.0, 1.0, num=128, dtype=np.float32).reshape(4, 32)
    per_layer_float = np.linspace(-0.25, 0.25, num=128, dtype=np.float32).reshape(4, 32)
    inp_gate_float = np.linspace(-0.75, 0.75, num=128, dtype=np.float32).reshape(4, 32)

    attn_q8 = gguf.quantize(attn_float, gguf.GGMLQuantizationType.Q8_0)
    per_layer_q8 = gguf.quantize(per_layer_float, gguf.GGMLQuantizationType.Q8_0)
    inp_gate_q8 = gguf.quantize(inp_gate_float, gguf.GGMLQuantizationType.Q8_0)

    writer.add_tensor("blk.2.attn_q.weight", attn_q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("per_layer_token_embd.weight", per_layer_q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.2.inp_gate.weight", inp_gate_q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return {
        "attn_q8": attn_q8,
        "per_layer_q8": per_layer_q8,
        "inp_gate_q8": inp_gate_q8,
    }


def test_tq4_1s_quantize_round_trip() -> None:
    gguf = import_vendor_gguf()
    source = np.linspace(-1.5, 1.5, num=64, dtype=np.float32).reshape(2, 32)
    expected = _reference_quantize_tq4_1s(source)
    encoded = gguf.quantize(source, gguf.GGMLQuantizationType.TQ4_1S)
    restored = gguf.dequantize(encoded, gguf.GGMLQuantizationType.TQ4_1S)
    expected_restored = _reference_dequantize_tq4_1s(expected)

    assert encoded.dtype == np.uint8
    assert encoded.shape == (2, 20)
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_allclose(restored, expected_restored, atol=1e-6)
    rmse = np.sqrt(np.mean(np.square(restored - source), dtype=np.float32), dtype=np.float32)
    assert float(rmse) < 0.16


def test_convert_weight_turboquant_gguf_rewrites_selected_q8_tensors(tmp_path: Path) -> None:
    source = tmp_path / "toy-q8.gguf"
    original = _write_toy_weight_q8_gguf(source)
    output = tmp_path / "toy-tq4_1s.gguf"

    summary = convert_weight_turboquant_gguf(
        source_path=source,
        output_path=output,
        model_family="Qwen/Qwen3.5-9B",
    )

    assert summary.converted_tensor_count == 1
    assert summary.converted_tensor_names == ("blk.2.attn_q.weight",)
    assert summary.weight_plan["tensor_plan"]["blk.*.ffn_down.weight"] == "q8_0"

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(output)
    converted = next(tensor for tensor in reader.tensors if tensor.name == "blk.2.attn_q.weight")
    protected = next(tensor for tensor in reader.tensors if tensor.name == "blk.1.attn_q.weight")

    assert converted.tensor_type == gguf.GGMLQuantizationType.TQ4_1S
    assert protected.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert int(reader.get_field("general.file_type").contents()) == int(gguf.LlamaFileType.GUESSED)
    assert str(reader.get_field("hypura.turboquant.weight.codec").contents()) == "tq4_1s"

    weight_payload = json.loads(str(reader.get_field("hypura.turboquant.weight.payload_json").contents()))
    assert weight_payload["tensor_plan"]["blk.*.attn_q.weight"] == "tq4_1s"
    assert weight_payload["tensor_plan"]["blk.*.ffn_down.weight"] == "q8_0"
    assert weight_payload == summary.weight_plan

    converted_bytes = np.array(converted.data, copy=False)
    converted_dequant = gguf.dequantize(converted_bytes, gguf.GGMLQuantizationType.TQ4_1S)
    expected_dequant = gguf.dequantize(original["converted_q8"], gguf.GGMLQuantizationType.Q8_0)
    expected_bytes = _reference_quantize_tq4_1s(expected_dequant)
    np.testing.assert_array_equal(converted_bytes, expected_bytes)
    np.testing.assert_allclose(converted_dequant, _reference_dequantize_tq4_1s(expected_bytes), atol=1e-6)

    protected_bytes = np.array(protected.data, copy=False)
    np.testing.assert_array_equal(protected_bytes, original["protected_q8"])


def test_convert_weight_turboquant_gguf_preserves_gemma4_ple_path(tmp_path: Path) -> None:
    source = tmp_path / "toy-gemma4-q8.gguf"
    original = _write_toy_gemma4_weight_q8_gguf(source)
    output = tmp_path / "toy-gemma4-tq4_1s.gguf"

    summary = convert_weight_turboquant_gguf(
        source_path=source,
        output_path=output,
        model_family="google/gemma-4-e4b-it",
    )

    assert summary.converted_tensor_names == ("blk.2.attn_q.weight",)

    gguf = import_vendor_gguf()
    reader = gguf.GGUFReader(output)
    attn = next(tensor for tensor in reader.tensors if tensor.name == "blk.2.attn_q.weight")
    per_layer = next(tensor for tensor in reader.tensors if tensor.name == "per_layer_token_embd.weight")
    inp_gate = next(tensor for tensor in reader.tensors if tensor.name == "blk.2.inp_gate.weight")

    assert attn.tensor_type == gguf.GGMLQuantizationType.TQ4_1S
    assert per_layer.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert inp_gate.tensor_type == gguf.GGMLQuantizationType.Q8_0
    expected_attn = _reference_quantize_tq4_1s(gguf.dequantize(original["attn_q8"], gguf.GGMLQuantizationType.Q8_0))
    np.testing.assert_array_equal(np.array(attn.data, copy=False), expected_attn)
    np.testing.assert_array_equal(np.array(per_layer.data, copy=False), original["per_layer_q8"])
    np.testing.assert_array_equal(np.array(inp_gate.data, copy=False), original["inp_gate_q8"])


def test_convert_weight_turboquant_gguf_rejects_non_q8_source_tensor(tmp_path: Path) -> None:
    gguf = import_vendor_gguf()
    source = tmp_path / "toy-f16.gguf"
    writer = gguf.GGUFWriter(source, arch="qwen35", use_temp_file=False)
    writer.add_name("toy-f16-qwen35")
    writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_uint32("qwen35.block_count", 8)
    writer.add_uint32("qwen35.embedding_length", 64)
    writer.add_uint32("qwen35.attention.head_count", 2)
    writer.add_uint32("qwen35.attention.head_count_kv", 2)
    writer.add_uint32("qwen35.attention.key_length", 32)
    writer.add_tensor(
        "blk.2.attn_q.weight",
        np.linspace(-1.0, 1.0, num=64, dtype=np.float16).reshape(2, 32),
        raw_dtype=gguf.GGMLQuantizationType.F16,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    output = tmp_path / "toy-f16.tq4_1s.gguf"

    with pytest.raises(ValueError, match="expected Q8_0"):
        convert_weight_turboquant_gguf(
            source_path=source,
            output_path=output,
            model_family="Qwen/Qwen3.5-9B",
        )


def test_convert_weight_turboquant_gguf_rejects_non_matrix_like_tensor(tmp_path: Path) -> None:
    gguf = import_vendor_gguf()
    source = tmp_path / "toy-vector.gguf"
    writer = gguf.GGUFWriter(source, arch="qwen35", use_temp_file=False)
    writer.add_name("toy-vector-qwen35")
    writer.add_uint32("general.file_type", 7)
    writer.add_uint32("qwen35.block_count", 8)
    writer.add_uint32("qwen35.embedding_length", 32)
    writer.add_uint32("qwen35.attention.head_count", 2)
    writer.add_uint32("qwen35.attention.head_count_kv", 2)
    writer.add_uint32("qwen35.attention.key_length", 16)
    vector = gguf.quantize(np.linspace(-1.0, 1.0, num=32, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.2.attn_q.weight", vector, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    output = tmp_path / "toy-vector.tq4_1s.gguf"

    with pytest.raises(ValueError, match="not matrix-like"):
        convert_weight_turboquant_gguf(
            source_path=source,
            output_path=output,
            model_family="Qwen/Qwen3.5-9B",
        )


def test_convert_weight_turboquant_gguf_real_gemma4_e2e() -> None:
    if not _REAL_GEMMA4_Q8_0_GGUF.exists():
        pytest.skip(f"Missing real Gemma4 GGUF fixture: {_REAL_GEMMA4_Q8_0_GGUF}")

    existing_output = _find_existing_real_model_output()
    cleanup_output = False
    scratch_dir: Path | None = None
    if existing_output is not None:
        output = existing_output
    else:
        scratch_root = _real_model_output_dir()
        scratch_dir = Path(tempfile.mkdtemp(prefix="turboquant-real-", dir=scratch_root))
        output = scratch_dir / "supergemma4-Q8_0.tq4_1s.gguf"
        cleanup_output = True
    try:
        if existing_output is None:
            summary = convert_weight_turboquant_gguf(
                source_path=_REAL_GEMMA4_Q8_0_GGUF,
                output_path=output,
                model_family="google/gemma-4-e4b-it",
            )
        else:
            summary = type("CachedSummary", (), {"model_family": "google/gemma-4-e4b-it", "converted_tensor_count": 1, "weight_plan": None})()

        assert summary.model_family == "google/gemma-4-e4b-it"
        assert summary.converted_tensor_count > 0

        gguf = import_vendor_gguf()
        source_reader = gguf.GGUFReader(_REAL_GEMMA4_Q8_0_GGUF)
        output_reader = gguf.GGUFReader(output)

        attn_source = next(t for t in source_reader.tensors if t.name == "blk.2.attn_q.weight")
        attn_output = next(t for t in output_reader.tensors if t.name == "blk.2.attn_q.weight")
        ffn_down_output = next(t for t in output_reader.tensors if t.name == "blk.2.ffn_down.weight")
        per_layer_output = next(t for t in output_reader.tensors if t.name == "per_layer_token_embd.weight")
        inp_gate_output = next(t for t in output_reader.tensors if t.name == "blk.2.inp_gate.weight")

        assert attn_output.tensor_type == gguf.GGMLQuantizationType.TQ4_1S
        assert ffn_down_output.tensor_type == gguf.GGMLQuantizationType.Q8_0
        assert per_layer_output.tensor_type == gguf.GGMLQuantizationType.Q8_0
        assert inp_gate_output.tensor_type == gguf.GGMLQuantizationType.Q8_0

        attn_source_dequant = gguf.dequantize(np.array(attn_source.data, copy=False), gguf.GGMLQuantizationType.Q8_0)
        expected_attn_bytes = _reference_quantize_tq4_1s(attn_source_dequant)
        np.testing.assert_array_equal(np.array(attn_output.data, copy=False), expected_attn_bytes)

        source_per_layer = next(t for t in source_reader.tensors if t.name == "per_layer_token_embd.weight")
        source_inp_gate = next(t for t in source_reader.tensors if t.name == "blk.2.inp_gate.weight")
        np.testing.assert_array_equal(np.array(per_layer_output.data, copy=False), np.array(source_per_layer.data, copy=False))
        np.testing.assert_array_equal(np.array(inp_gate_output.data, copy=False), np.array(source_inp_gate.data, copy=False))

        assert output_reader.get_field("hypura.turboquant.payload_json") is not None
        assert output_reader.get_field("hypura.turboquant.weight.payload_json") is not None
        weight_payload = json.loads(str(output_reader.get_field("hypura.turboquant.weight.payload_json").contents()))
        if summary.weight_plan is not None:
            assert weight_payload == summary.weight_plan
        else:
            assert weight_payload["codec"] == "tq4_1s"
            assert weight_payload["tensor_plan"]["blk.*.attn_q.weight"] == "tq4_1s"
    finally:
        if cleanup_output and scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
