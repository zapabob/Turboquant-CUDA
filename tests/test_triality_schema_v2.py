from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from turboquant.triality_contract import (
    TRIALITY_GGUF_PAYLOAD_FORMAT_V1,
    TRIALITY_GGUF_PAYLOAD_FORMAT_V2,
    build_triality_metadata,
    build_triality_payload,
    payload_json_dumps,
    validate_triality_metadata,
    validate_triality_payload,
)
from turboquant.triality_fixture_gguf import read_fixture_gguf, write_fixture_gguf
from turboquant.triality_schema_v2 import (
    TRIALITY_CONSENSUS_VIEWS,
    build_triality_v2_tensors,
    consensus_tensor_name,
    rotation_tensor_name,
    tensor_sha256,
    validate_rotation_tensor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_triality_fixture.py"
VERIFY_SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_triality_export.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2_payload(*, enable_ncka: bool = True, enable_urt: bool = True) -> dict[str, Any]:
    return build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=2,
        num_kv_heads=1,
        model_family="Qwen/Qwen3.5-test",
        schema_version=2,
        enable_ncka=enable_ncka,
        enable_urt=enable_urt,
    )


def _run_script(module, argv: list[str]) -> int:
    old = sys.argv
    try:
        sys.argv = argv
        return module.main()
    finally:
        sys.argv = old


def test_schema_v2_contract_contains_exact_three_view_bundle() -> None:
    payload = _v2_payload()
    validate_triality_payload(payload)
    tensors = build_triality_v2_tensors(payload)

    assert payload["profile_id"] == "v2"
    for layer in range(2):
        for view in TRIALITY_CONSENSUS_VIEWS:
            name = rotation_tensor_name(layer, view)
            assert tensors[name]["shape"] == [8, 8]
            validate_rotation_tensor(tensors[name], head_dim=8)
    assert tensors[consensus_tensor_name("weights")]["shape"] == [3, 2]
    assert tensors["turboquant.profile.v2.ncka.inner_knots"]["shape"] == [3, 24, 2, 3]

    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json_dumps(payload),
        weight_plan=payload["weight_plan"],
    )
    assert (
        metadata["hypura.turboquant.payload_format"] == TRIALITY_GGUF_PAYLOAD_FORMAT_V2
    )
    assert metadata["hypura.turboquant.triality.profile_id"] == "v2"
    assert metadata["hypura.turboquant.triality.views"] == list(
        TRIALITY_CONSENSUS_VIEWS
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["tensor_manifest"].pop(
                rotation_tensor_name(0, "vector")
            ),
            "key set",
        ),
        (
            lambda value: value["tensor_manifest"].__setitem__("unexpected", {}),
            "key set",
        ),
        (
            lambda value: value["consensus"]["rows"][0].__setitem__(
                "weights", [0.5, 0.5, 0.5]
            ),
            "sum to 1",
        ),
        (
            lambda value: value["consensus"]["rows"][0].__setitem__(
                "scale", [1.0, 0.0, 1.0]
            ),
            "positive",
        ),
        (
            lambda value: value.__setitem__("profile_id", "../escape"),
            "safe non-empty token",
        ),
        (
            lambda value: value["ncka"].__setitem__("controller_sha256", "0" * 64),
            "controller hash",
        ),
        (
            lambda value: value["ncka"].__setitem__("fallback_policy", "implicit"),
            "fallback_policy",
        ),
        (
            lambda value: value["urt"].__setitem__("operator_word_sha256", "0" * 64),
            "operator word hash",
        ),
        (
            lambda value: value.__setitem__("rotation_polciy", "block_so8_learned"),
            "unexpected rotation_polciy",
        ),
        (
            lambda value: value.__setitem__("num_layers", "2"),
            "num_layers must be an integer",
        ),
        (
            lambda value: value["consensus"]["rows"][0].__setitem__("layer", True),
            "layer must be an integer",
        ),
        (
            lambda value: value["consensus"].__setitem__("js_fallback_threshold", 1),
            "must be a finite floating-point value",
        ),
        (
            lambda value: value["ncka"].__setitem__("enabled", "true"),
            "ncka.enabled must be a boolean",
        ),
        (
            lambda value: value.__setitem__("rotation_policy", "identity_typo"),
            "rotation_policy must be one of",
        ),
    ],
)
def test_schema_v2_payload_rejects_malformed_contracts(mutate, match: str) -> None:
    payload = deepcopy(_v2_payload())
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_triality_payload(payload)


def test_identity_dev_requires_marker_disabled_weights_and_no_override() -> None:
    with pytest.raises(ValueError, match="development identity marker"):
        build_triality_payload(
            mode="triality-proxy-so8-pareto",
            head_dim=8,
            num_layers=1,
            num_kv_heads=1,
            model_family="llama",
            weight_source_ftype="q4_0",
            weight_enabled=False,
            rotation_policy="identity_dev",
            schema_version=2,
        )

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
    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json_dumps(payload),
        weight_plan=payload["weight_plan"],
        rotation_policy="identity_dev",
        schema_version=2,
    )
    assert metadata["hypura.turboquant.triality.override_allowed"] is False
    metadata["hypura.turboquant.triality.override_allowed"] = True
    with pytest.raises(ValueError, match="does not match payload"):
        validate_triality_metadata(metadata)


def test_identity_dev_is_rejected_by_all_schema_v1_paths() -> None:
    with pytest.raises(ValueError, match="requires Triality schema-v2"):
        build_triality_payload(
            mode="triality-proxy-so8-pareto",
            head_dim=8,
            num_layers=1,
            num_kv_heads=1,
            rotation_policy="identity_dev",
            schema_version=1,
        )

    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
        schema_version=1,
    )
    handcrafted_payload = deepcopy(payload)
    handcrafted_payload["rotation_policy"] = "identity_dev"
    with pytest.raises(ValueError, match="requires Triality schema-v2"):
        validate_triality_payload(handcrafted_payload)

    payload_json = payload_json_dumps(payload)
    with pytest.raises(ValueError, match="requires Triality schema-v2"):
        build_triality_metadata(
            mode="triality-proxy-so8-pareto",
            payload_json=payload_json,
            weight_plan=payload["weight_plan"],
            rotation_policy="identity_dev",
            schema_version=1,
        )

    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json,
        weight_plan=payload["weight_plan"],
        schema_version=1,
    )
    metadata["hypura.turboquant.rotation_policy"] = "identity_dev"
    with pytest.raises(ValueError, match="requires Triality schema-v2"):
        validate_triality_metadata(metadata)


def test_rotation_validation_rejects_off_block_coupling() -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=16,
        num_layers=1,
        num_kv_heads=1,
        schema_version=2,
    )
    tensor = deepcopy(
        build_triality_v2_tensors(payload)[rotation_tensor_name(0, "vector")]
    )
    tensor["data"][8] = 0.25
    with pytest.raises(ValueError, match="block diagonal"):
        validate_rotation_tensor(tensor, head_dim=16)


def test_fixture_writer_is_byte_deterministic_and_round_trips_real_gguf(
    tmp_path: Path,
) -> None:
    payload = _v2_payload(enable_ncka=False, enable_urt=False)
    tensors = build_triality_v2_tensors(payload)
    metadata = {
        "z.last": "value",
        "a.first": [1.0, 2.0, 3.0],
        "schema": 2,
    }
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    write_fixture_gguf(path=first, metadata=metadata, tensors=tensors)
    write_fixture_gguf(
        path=second,
        metadata=dict(reversed(list(metadata.items()))),
        tensors=dict(reversed(list(tensors.items()))),
    )

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[:4] == b"GGUF"
    round_trip = read_fixture_gguf(first)
    assert round_trip["metadata"] == metadata
    assert set(round_trip["tensors"]) == set(tensors)
    for name, tensor in tensors.items():
        assert round_trip["tensors"][name]["dtype"] == tensor["dtype"]
        assert round_trip["tensors"][name]["shape"] == tensor["shape"]
        assert tensor_sha256(round_trip["tensors"][name]["data"]) == tensor_sha256(
            tensor["data"]
        )


def test_export_and_verify_v2_bundle_is_reproducible(tmp_path: Path) -> None:
    exporter = _load_module(EXPORT_SCRIPT_PATH, "export_triality_fixture_v2")
    verifier = _load_module(VERIFY_SCRIPT_PATH, "verify_triality_fixture_v2")
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        assert (
            _run_script(
                exporter,
                [
                    "export_triality_fixture.py",
                    "--output-dir",
                    str(root),
                    "--mode",
                    "triality-proxy-so8-pareto",
                    "--schema-version",
                    "2",
                    "--head-dim",
                    "8",
                    "--num-layers",
                    "2",
                    "--num-kv-heads",
                    "1",
                    "--enable-ncka",
                    "--enable-urt",
                ],
            )
            == 0
        )

    bundles = [root / "triality-proxy-so8-pareto" for root in roots]
    relative_files = sorted(
        path.relative_to(bundles[0]) for path in bundles[0].iterdir()
    )
    assert relative_files
    assert relative_files == sorted(
        path.relative_to(bundles[1]) for path in bundles[1].iterdir()
    )
    for relative in relative_files:
        assert (bundles[0] / relative).read_bytes() == (
            bundles[1] / relative
        ).read_bytes()

    manifest_path = bundles[0] / "triality-fixture-manifest.json"
    assert (
        _run_script(
            verifier,
            ["verify_triality_export.py", "--manifest", str(manifest_path)],
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["triality_schema_version"] == 2
    assert set(manifest["hashes"]) == {
        "payload_sha256",
        "metadata_sha256",
        "offline_metrics_sha256",
        "text_model_sha256",
        "mmproj_model_sha256",
    }
    escaped_manifest = deepcopy(manifest)
    escaped_manifest["paths"]["payload"] = "../outside.json"
    manifest_path.write_text(json.dumps(escaped_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the fixture bundle"):
        _run_script(
            verifier,
            ["verify_triality_export.py", "--manifest", str(manifest_path)],
        )


def test_schema_v1_remains_default_and_verifiable(tmp_path: Path) -> None:
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
    )
    assert payload["schema_version"] == 1
    assert "consensus" not in payload
    metadata = build_triality_metadata(
        mode="triality-proxy-so8-pareto",
        payload_json=payload_json_dumps(payload),
        weight_plan=payload["weight_plan"],
    )
    assert (
        metadata["hypura.turboquant.payload_format"] == TRIALITY_GGUF_PAYLOAD_FORMAT_V1
    )

    exporter = _load_module(EXPORT_SCRIPT_PATH, "export_triality_fixture_v1")
    verifier = _load_module(VERIFY_SCRIPT_PATH, "verify_triality_fixture_v1")
    assert (
        _run_script(
            exporter,
            [
                "export_triality_fixture.py",
                "--output-dir",
                str(tmp_path),
                "--mode",
                "triality-proxy-so8-pareto",
                "--head-dim",
                "8",
                "--num-layers",
                "1",
                "--num-kv-heads",
                "1",
            ],
        )
        == 0
    )
    manifest_path = (
        tmp_path / "triality-proxy-so8-pareto" / "triality-fixture-manifest.json"
    )
    assert (
        _run_script(
            verifier,
            ["verify_triality_export.py", "--manifest", str(manifest_path)],
        )
        == 0
    )


def test_v1_export_rejects_v2_only_features(tmp_path: Path) -> None:
    exporter = _load_module(EXPORT_SCRIPT_PATH, "export_triality_fixture_invalid_v1")
    with pytest.raises(ValueError, match="require --schema-version 2"):
        _run_script(
            exporter,
            [
                "export_triality_fixture.py",
                "--output-dir",
                str(tmp_path),
                "--mode",
                "triality-proxy-so8-pareto",
                "--enable-ncka",
            ],
        )
