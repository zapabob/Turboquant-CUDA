from __future__ import annotations

from pathlib import Path

import torch

from turboquant.analysis import load_captured_layers, load_captured_runs
from turboquant.capture import CaptureMetadata, LayerCaptureRecord, normalize_past_key_values, save_capture_metadata
from turboquant.runtime import LOCAL_CAPTURE_MODEL_PATH, get_model_preset, model_preset_to_id
from turboquant.schema import build_capture_quantization_config


def build_capture_dir(root: Path, name: str, *, include_value: bool = True) -> Path:
    capture_dir = root / name
    capture_dir.mkdir(parents=True, exist_ok=True)
    key = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    value = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    key_path = capture_dir / "layer_00_key.pt"
    value_path = capture_dir / "layer_00_value.pt"
    torch.save(key, key_path)
    if include_value:
        torch.save(value, value_path)
    metadata = CaptureMetadata(
        model_name="H:\\Qwen3.5-9B-official-hf",
        tokenizer_name="Qwen3.5-9B",
        prompt_hash="abc123",
        prompt_length=12,
        timestamp_utc="2026-03-27T00:00:00+00:00",
        device="cpu",
        dtype="torch.float32",
        layer_count=1,
        package_versions={"torch": "test"},
        git_commit_hash="deadbeef",
        layers=[
            LayerCaptureRecord(
                layer_index=0,
                key_shape=list(key.shape),
                value_shape=list(value.shape),
                key_file=key_path.name,
                value_file=value_path.name,
            )
        ],
        model_source="H:\\Qwen3.5-9B-official-hf",
        prompt_label=name,
        capture_id=name,
        model_preset="qwen35_9b_12gb",
        lane_name="rtx3060_desktop_12gb",
        seed=7,
        quantization_config=build_capture_quantization_config(
            weight_load="4bit",
            requested_dtype="float16",
            trust_remote_code=False,
            max_length=96,
        ),
    )
    save_capture_metadata(capture_dir / "capture_manifest.json", metadata)
    return capture_dir


def test_runtime_exposes_rtx3060_capture_presets() -> None:
    nine_b = get_model_preset("qwen35_9b_12gb")
    assert nine_b.lane_name == "rtx3060_desktop_12gb"
    assert nine_b.model_id == "Qwen/Qwen3.5-9B"
    assert model_preset_to_id("qwen35_9b_12gb") == LOCAL_CAPTURE_MODEL_PATH

    four_b = get_model_preset("qwen35_4b_6gb")
    assert four_b.lane_name == "rtx3060_notebook_6gb"
    assert four_b.model_id == "Qwen/Qwen3.5-4B"
    assert model_preset_to_id("qwen35_4b_6gb") == "Qwen/Qwen3.5-4B"
    assert model_preset_to_id("qwen35_4b_6gb") != LOCAL_CAPTURE_MODEL_PATH


def test_load_captured_layers_requires_manifest(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    try:
        load_captured_layers(missing_dir)
    except FileNotFoundError as exc:
        assert "capture manifest" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing manifest")


def test_load_captured_layers_requires_value_tensor(tmp_path: Path) -> None:
    capture_dir = build_capture_dir(tmp_path, "broken", include_value=False)
    try:
        load_captured_layers(capture_dir)
    except FileNotFoundError as exc:
        assert "value tensor" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing value tensor")


def test_load_captured_runs_reads_prompt_scoped_directories(tmp_path: Path) -> None:
    build_capture_dir(tmp_path, "prompt-a")
    build_capture_dir(tmp_path, "prompt-b")
    bundles = load_captured_runs(tmp_path)
    labels = {bundle.metadata.prompt_label for bundle in bundles}
    assert labels == {"prompt-a", "prompt-b"}
    assert all(bundle.keys.shape == bundle.values.shape for bundle in bundles)


def test_normalize_past_key_values_supports_qwen_style_cache_lists() -> None:
    class DummyCache:
        def __init__(self) -> None:
            self.key_cache = [torch.randn((1, 2, 4, 8)), None]
            self.value_cache = [torch.randn((1, 2, 4, 8)), None]

    normalized = normalize_past_key_values(DummyCache())
    assert len(normalized) == 1
    assert normalized[0][0].shape == normalized[0][1].shape


def test_load_captured_layers_preserves_lane_seed_and_quantization_metadata(tmp_path: Path) -> None:
    capture_dir = build_capture_dir(tmp_path, "prompt-a")
    metadata, layers = load_captured_layers(capture_dir)

    assert metadata is not None
    assert layers
    assert metadata.model_preset == "qwen35_9b_12gb"
    assert metadata.lane_name == "rtx3060_desktop_12gb"
    assert metadata.seed == 7
    assert metadata.quantization_config == build_capture_quantization_config(
        weight_load="4bit",
        requested_dtype="float16",
        trust_remote_code=False,
        max_length=96,
    )
