from __future__ import annotations

from copy import deepcopy
import importlib.util
from itertools import permutations
import json
import math
from pathlib import Path
import struct
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
    TRIALITY_NCKA_COORDINATE_NAMES,
    build_triality_v2_tensors,
    consensus_tensor_name,
    ncka_tensor_name,
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


def _f32_values(tensor: dict[str, Any]) -> list[float]:
    return [
        struct.unpack("<f", struct.pack("<f", float(value)))[0]
        for value in tensor["data"]
    ]


def _interp_linear(knots: list[float], values: list[float], x: float) -> float:
    if x <= knots[0]:
        return values[0]
    if x >= knots[-1]:
        return values[-1]
    for index in range(len(knots) - 1):
        if x <= knots[index + 1]:
            width = knots[index + 1] - knots[index]
            fraction = (x - knots[index]) / width
            return values[index] + fraction * (values[index + 1] - values[index])
    raise AssertionError("interpolation interval was not found")


def _evaluate_ncka_controller(
    tensors: dict[str, dict[str, Any]], coordinates: list[float], *, profile: str
) -> list[float]:
    coordinate_count = len(TRIALITY_NCKA_COORDINATE_NAMES)
    outer_count = 2
    knot_count = 3
    coordinate_min = _f32_values(tensors[ncka_tensor_name("coordinate_min", profile)])
    coordinate_max = _f32_values(tensors[ncka_tensor_name("coordinate_max", profile)])
    inner_knots = _f32_values(tensors[ncka_tensor_name("inner_knots", profile)])
    inner_values = _f32_values(tensors[ncka_tensor_name("inner_values", profile)])
    outer_knots = _f32_values(tensors[ncka_tensor_name("outer_knots", profile)])
    outer_values = _f32_values(tensors[ncka_tensor_name("outer_values", profile)])
    normalized = [
        min(1.0, max(0.0, (value - lower) / max(upper - lower, 1.0e-6)))
        for value, lower, upper in zip(
            coordinates, coordinate_min, coordinate_max, strict=True
        )
    ]

    logits: list[float] = []
    for branch in range(3):
        logit = 0.0
        for outer in range(outer_count):
            inner_sum = 0.0
            for coordinate, value in enumerate(normalized):
                offset = (
                    (branch * outer_count + outer) * coordinate_count + coordinate
                ) * knot_count
                inner_sum += _interp_linear(
                    inner_knots[offset : offset + knot_count],
                    inner_values[offset : offset + knot_count],
                    value,
                )
            offset = (branch * outer_count + outer) * knot_count
            logit += _interp_linear(
                outer_knots[offset : offset + knot_count],
                outer_values[offset : offset + knot_count],
                inner_sum,
            )
        logits.append(logit)

    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _permute_ncka_coordinates(
    coordinates: list[float], permutation: tuple[int, ...]
) -> list[float]:
    if len(permutation) != 3:
        raise ValueError("S3 permutation must contain exactly three branches")
    coordinate_names = list(TRIALITY_NCKA_COORDINATE_NAMES)
    index_by_name = {name: index for index, name in enumerate(coordinate_names)}
    edge_by_name = {
        "pairwise_js.vector_plus": frozenset({0, 1}),
        "pairwise_js.vector_minus": frozenset({0, 2}),
        "pairwise_js.plus_minus": frozenset({1, 2}),
    }
    name_by_edge = {edge: name for name, edge in edge_by_name.items()}
    branch_names = list(TRIALITY_CONSENSUS_VIEWS)
    branch_index = {name: index for index, name in enumerate(branch_names)}
    permuted = [0.0] * len(coordinates)

    for old_index, coordinate_name in enumerate(coordinate_names):
        new_name = coordinate_name
        if coordinate_name in edge_by_name:
            new_edge = frozenset(
                permutation[branch] for branch in edge_by_name[coordinate_name]
            )
            new_name = name_by_edge[new_edge]
        elif "." in coordinate_name:
            family, suffix = coordinate_name.rsplit(".", 1)
            if suffix in branch_index:
                new_name = f"{family}.{branch_names[permutation[branch_index[suffix]]]}"
        permuted[index_by_name[new_name]] = coordinates[old_index]
    return permuted


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


def test_generated_ncka_controller_is_equivariant_for_all_s3_permutations() -> None:
    profile = "s3check"
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
        schema_version=2,
        profile_id=profile,
        enable_ncka=True,
    )
    tensors = build_triality_v2_tensors(payload)
    inner_values = _f32_values(tensors[ncka_tensor_name("inner_values", profile)])
    outer_values = _f32_values(tensors[ncka_tensor_name("outer_values", profile)])
    coordinate_count = len(TRIALITY_NCKA_COORDINATE_NAMES)
    for branch in range(3):
        inner_outer1_offset = (branch * 2 + 1) * coordinate_count * 3
        assert inner_values[
            inner_outer1_offset : inner_outer1_offset + coordinate_count * 3
        ] == [0.0] * (coordinate_count * 3)
        outer_outer1_offset = (branch * 2 + 1) * 3
        assert outer_values[outer_outer1_offset : outer_outer1_offset + 3] == [
            0.0,
            0.0,
            0.0,
        ]
    coordinates = [((index * 37 + 11) % 101) / 100.0 for index in range(24)]
    baseline = _evaluate_ncka_controller(tensors, coordinates, profile=profile)

    assert all(math.isfinite(value) and value >= 0.0 for value in baseline)
    assert math.isclose(sum(baseline), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert max(baseline) - min(baseline) > 1.0e-4

    for permutation in permutations(range(3)):
        permuted_coordinates = _permute_ncka_coordinates(coordinates, permutation)
        actual = _evaluate_ncka_controller(
            tensors, permuted_coordinates, profile=profile
        )
        expected = [0.0, 0.0, 0.0]
        for old_branch, new_branch in enumerate(permutation):
            expected[new_branch] = baseline[old_branch]
        assert all(math.isfinite(value) and value >= 0.0 for value in actual)
        assert math.isclose(sum(actual), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert actual == pytest.approx(expected, rel=0.0, abs=2.0e-7)


def test_ncka_memory_ratio_modulates_weights_and_preserves_s3_equivariance() -> None:
    profile = "s3memory"
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
        schema_version=2,
        profile_id=profile,
        enable_ncka=True,
    )
    tensors = build_triality_v2_tensors(payload)
    memory_index = list(TRIALITY_NCKA_COORDINATE_NAMES).index("memory_ratio")
    baseline_coordinates = [((index * 37 + 11) % 101) / 100.0 for index in range(24)]
    weights_by_memory: dict[float, list[float]] = {}

    for memory_ratio in (0.0, 1.0):
        coordinates = list(baseline_coordinates)
        coordinates[memory_index] = memory_ratio
        baseline = _evaluate_ncka_controller(tensors, coordinates, profile=profile)
        weights_by_memory[memory_ratio] = baseline
        assert all(math.isfinite(value) and value >= 0.0 for value in baseline)
        assert math.isclose(sum(baseline), 1.0, rel_tol=0.0, abs_tol=1.0e-12)

        for permutation in permutations(range(3)):
            actual = _evaluate_ncka_controller(
                tensors,
                _permute_ncka_coordinates(coordinates, permutation),
                profile=profile,
            )
            expected = [0.0, 0.0, 0.0]
            for old_branch, new_branch in enumerate(permutation):
                expected[new_branch] = baseline[old_branch]
            assert actual == pytest.approx(expected, rel=0.0, abs=2.0e-7)

    assert (
        max(
            abs(low - high)
            for low, high in zip(
                weights_by_memory[0.0], weights_by_memory[1.0], strict=True
            )
        )
        > 5.0e-4
    )


def test_generated_ncka_controller_matches_fixed_oracle_and_gguf_round_trip(
    tmp_path: Path,
) -> None:
    profile = "s3oracle"
    payload = build_triality_payload(
        mode="triality-proxy-so8-pareto",
        head_dim=8,
        num_layers=1,
        num_kv_heads=1,
        schema_version=2,
        profile_id=profile,
        enable_ncka=True,
    )
    tensors = build_triality_v2_tensors(payload)
    coordinates = [
        0.11,
        0.48,
        0.85,
        0.21,
        0.58,
        0.95,
        0.31,
        0.68,
        0.04,
        0.41,
        0.78,
        0.14,
        0.51,
        0.88,
        0.24,
        0.61,
        0.98,
        0.34,
        0.71,
        0.07,
        0.44,
        0.81,
        0.17,
        0.54,
    ]
    expected_weights = [
        0.32970839831029586,
        0.34043480116596675,
        0.3298568005237375,
    ]
    expected_controller_sha256 = (
        "7a1bdd43cdc7e105076d8171b1e29436dbb5367b7616330feb4fa9581bbffab9"
    )

    assert payload["ncka"]["controller_sha256"] == expected_controller_sha256
    assert _evaluate_ncka_controller(
        tensors, coordinates, profile=profile
    ) == pytest.approx(expected_weights, rel=0.0, abs=1.0e-12)

    fixture_path = tmp_path / "ncka-oracle.gguf"
    write_fixture_gguf(
        path=fixture_path,
        metadata={"controller_sha256": expected_controller_sha256},
        tensors=tensors,
    )
    round_trip = read_fixture_gguf(fixture_path)
    assert round_trip["metadata"]["controller_sha256"] == expected_controller_sha256
    assert _evaluate_ncka_controller(
        round_trip["tensors"], coordinates, profile=profile
    ) == pytest.approx(expected_weights, rel=0.0, abs=1.0e-12)


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
