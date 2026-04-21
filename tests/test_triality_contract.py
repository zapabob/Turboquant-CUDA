from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from turboquant.triality_contract import (
    TRIALITY_ROTATION_BLOCK_SIZE,
    TRIALITY_RUNTIME_MODE,
    TRIALITY_RUNTIME_MODE_BEST_PER_LAYER,
    TRIALITY_RUNTIME_MODE_MINUS,
    TRIALITY_RUNTIME_MODE_PLUS,
    build_triality_metadata,
    build_triality_payload,
    expected_modalities,
    normalize_triality_runtime_mode,
    normalize_triality_view,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_triality_fixture.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "model_family",
    [
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
    ],
)
def test_qwen35_payload_uses_weight_v1_config_i_contract(model_family: str) -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=128,
        num_layers=32,
        num_kv_heads=8,
        model_family=model_family,
    )

    weight_plan = payload["weight_plan"]
    assert weight_plan["schema"] == "hypura.turboquant.weight.v1"
    assert weight_plan["codec"] == "tq4_1s"
    assert weight_plan["policy"] == "qwen35-config-i"
    assert weight_plan["modality_scope"] == "text-only"
    assert weight_plan["protected_roles"] == [
        "embedding",
        "norm",
        "output_head",
        "recurrent_state",
    ]
    assert weight_plan["tensor_plan"] == {
        "blk.*.attn_q.weight": "tq4_1s",
        "blk.*.attn_k.weight": "tq4_1s",
        "blk.*.attn_v.weight": "tq4_1s",
        "blk.*.attn_output.weight": "tq4_1s",
        "blk.*.ffn_gate.weight": "tq4_1s",
        "blk.*.ffn_up.weight": "tq4_1s",
        "blk.*.ffn_down.weight": "q4_k",
    }


@pytest.mark.parametrize(
    "model_family",
    [
        "google/gemma-4-e2b-it",
        "google/gemma-4-e4b-it",
        "google/gemma-4-26b-a4b-it",
    ],
)
def test_gemma4_payload_uses_multimodal_safe_weight_v1_contract(model_family: str) -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=256,
        num_layers=48,
        num_kv_heads=8,
        model_family=model_family,
    )

    weight_plan = payload["weight_plan"]
    assert weight_plan["schema"] == "hypura.turboquant.weight.v1"
    assert weight_plan["codec"] == "tq4_1s"
    assert weight_plan["policy"] == "gemma4-kv-first-multimodal-safe"
    assert weight_plan["modality_scope"] == "full-multimodal"
    assert weight_plan["protected_roles"] == [
        "vision_encoder",
        "audio_encoder",
        "projector",
        "per_layer_multimodal_embedding",
        "embedding",
        "norm",
        "output_head",
    ]
    assert weight_plan["tensor_plan"] == {
        "blk.*.attn_q.weight": "tq4_1s",
        "blk.*.attn_k.weight": "tq4_1s",
        "blk.*.attn_v.weight": "tq4_1s",
        "blk.*.attn_output.weight": "tq4_1s",
        "blk.*.ffn_gate.weight": "tq4_1s",
        "blk.*.ffn_up.weight": "tq4_1s",
        "blk.*.ffn_down.weight": "q4_k",
    }


def test_expected_modalities_marks_gemma4_multimodal_families() -> None:
    assert expected_modalities(model_family="google/gemma-4-e2b-it") == [
        "text",
        "image",
        "audio",
    ]
    assert expected_modalities(model_family="google/gemma-4-e4b-it") == [
        "text",
        "image",
        "audio",
    ]
    assert expected_modalities(model_family="google/gemma-4-26b-a4b-it") == [
        "text",
        "image",
        "audio",
    ]


def test_triality_metadata_includes_weight_codec_and_v1_payload() -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=128,
        num_layers=32,
        num_kv_heads=8,
        model_family="Qwen/Qwen3.5-9B",
    )
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
    )

    assert metadata["hypura.turboquant.codec"] == "tq4_1s"
    assert metadata["hypura.turboquant.rotation_block_size"] == TRIALITY_ROTATION_BLOCK_SIZE
    assert metadata["hypura.turboquant.runtime_mode"] == TRIALITY_RUNTIME_MODE
    assert metadata["hypura.turboquant.weight.codec"] == "tq4_1s"
    weight_payload = json.loads(metadata["hypura.turboquant.weight.payload_json"])
    assert weight_payload["schema"] == "hypura.turboquant.weight.v1"
    assert weight_payload["codec"] == "tq4_1s"
    assert "target_type" not in weight_payload


def test_triality_alias_normalization_accepts_the_tom_and_zapabob_spellings() -> None:
    assert normalize_triality_runtime_mode("triality-vector") == TRIALITY_RUNTIME_MODE
    assert normalize_triality_runtime_mode("research-kv-split") == TRIALITY_RUNTIME_MODE
    assert normalize_triality_runtime_mode("triality_vector") == TRIALITY_RUNTIME_MODE
    assert normalize_triality_runtime_mode("triality-plus") == TRIALITY_RUNTIME_MODE_PLUS
    assert normalize_triality_runtime_mode("spinor_minus_proxy") == TRIALITY_RUNTIME_MODE_MINUS
    assert normalize_triality_runtime_mode("best_per_layer") == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER
    assert normalize_triality_view("plus") == "spinor_plus_proxy"
    assert normalize_triality_view("minus") == "spinor_minus_proxy"


def test_triality_metadata_supports_spinor_plus_public_view() -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=128,
        num_layers=32,
        num_kv_heads=8,
        model_family="Qwen/Qwen3.5-9B",
    )
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
        triality_view="plus",
        runtime_mode="triality-plus",
    )

    assert metadata["hypura.turboquant.triality_view"] == "spinor_plus_proxy"
    assert metadata["hypura.turboquant.runtime_mode"] == TRIALITY_RUNTIME_MODE_PLUS
    assert metadata["hypura.turboquant.view_bundle_complete"] is True


def test_triality_metadata_supports_best_per_layer_runtime_when_bundle_is_complete() -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=128,
        num_layers=32,
        num_kv_heads=8,
        model_family="Qwen/Qwen3.5-9B",
    )
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
        triality_view="vector",
        runtime_mode="best_per_layer",
    )

    assert metadata["hypura.turboquant.runtime_mode"] == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER
    assert metadata["hypura.turboquant.view_bundle_complete"] is True


def test_export_triality_fixture_writes_mmproj_pair_for_gemma4(tmp_path: Path) -> None:
    module = _load_module(EXPORT_SCRIPT_PATH, "export_triality_fixture")
    output_dir = tmp_path / "fixtures"
    argv = [
        "export_triality_fixture.py",
        "--output-dir",
        str(output_dir),
        "--mode",
        "triality-proxy-so8-pareto",
        "--model-family",
        "google/gemma-4-e4b-it",
    ]

    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    bundle_dir = output_dir / "triality-proxy-so8-pareto"
    manifest = json.loads(
        (bundle_dir / "triality-fixture-manifest.json").read_text(encoding="utf-8")
    )
    payload = json.loads(
        (bundle_dir / "triality-payload.json").read_text(encoding="utf-8")
    )

    assert manifest["mmproj_required"] is True
    assert manifest["paths"]["mmproj_model"] == "mmproj-triality-fixture.gguf"
    assert (bundle_dir / "mmproj-triality-fixture.gguf").exists()
    assert payload["weight_plan"]["schema"] == "hypura.turboquant.weight.v1"
    assert payload["weight_plan"]["codec"] == "tq4_1s"


def test_export_triality_fixture_writes_text_only_manifest_for_qwen35(tmp_path: Path) -> None:
    module = _load_module(EXPORT_SCRIPT_PATH, "export_triality_fixture_qwen")
    output_dir = tmp_path / "fixtures"
    argv = [
        "export_triality_fixture.py",
        "--output-dir",
        str(output_dir),
        "--mode",
        "triality-proxy-so8-pareto",
        "--model-family",
        "Qwen/Qwen3.5-27B",
    ]

    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    bundle_dir = output_dir / "triality-proxy-so8-pareto"
    manifest = json.loads(
        (bundle_dir / "triality-fixture-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["mmproj_required"] is False
    assert manifest["paths"]["mmproj_model"] is None
    assert manifest["modalities"] == ["text"]
