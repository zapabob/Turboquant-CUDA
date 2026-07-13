from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Iterable


TRIALITY_SCHEMA_V2 = 2
TRIALITY_PAYLOAD_FORMAT_V2 = "json-inline-v2"
TRIALITY_CONSENSUS_VIEWS = (
    "vector",
    "spinor_plus_proxy",
    "spinor_minus_proxy",
)
TRIALITY_CONSENSUS_BRANCH_COUNT = len(TRIALITY_CONSENSUS_VIEWS)
TRIALITY_NCKA_SCHEMA_VERSION = 1
TRIALITY_NCKA_CONTROLLER_TYPE = "finite_moment_ka_v1"
TRIALITY_URT_SCHEMA_VERSION = 1
TRIALITY_URT_ALGEBRA_ID = "octonion_triality_proxy_v1"
TRIALITY_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

TRIALITY_NCKA_COORDINATE_NAMES = (
    "branch_entropy.vector",
    "branch_entropy.spinor_plus_proxy",
    "branch_entropy.spinor_minus_proxy",
    "orthogonality_error.vector",
    "orthogonality_error.spinor_plus_proxy",
    "orthogonality_error.spinor_minus_proxy",
    "determinant_error.vector",
    "determinant_error.spinor_plus_proxy",
    "determinant_error.spinor_minus_proxy",
    "expected_quant_error.vector",
    "expected_quant_error.spinor_plus_proxy",
    "expected_quant_error.spinor_minus_proxy",
    "pairwise_js.vector_plus",
    "pairwise_js.vector_minus",
    "pairwise_js.plus_minus",
    "candidate_cross_score_mean.vector",
    "candidate_cross_score_mean.spinor_plus_proxy",
    "candidate_cross_score_mean.spinor_minus_proxy",
    "candidate_cross_score_variance.vector",
    "candidate_cross_score_variance.spinor_plus_proxy",
    "candidate_cross_score_variance.spinor_minus_proxy",
    "winner_margin",
    "latency_multiplier",
    "memory_ratio",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_f32(values: Iterable[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def tensor_sha256(values: Iterable[float]) -> str:
    return _sha256_f32(values)


def rotation_tensor_name(layer: int, view: str, profile: str = "v2") -> str:
    return f"turboquant.profile.{profile}.layer.{layer}.rotation.{view}"


def consensus_tensor_name(field: str, profile: str = "v2") -> str:
    return f"turboquant.profile.{profile}.consensus.{field}"


def ncka_tensor_name(field: str, profile: str = "v2") -> str:
    return f"turboquant.profile.{profile}.ncka.{field}"


def _rotation_matrix(head_dim: int, layer: int, branch: int) -> list[float]:
    size = head_dim
    matrix = [0.0] * (size * size)
    for index in range(size):
        matrix[index * size + index] = 1.0
    if branch == 0:
        return matrix

    sign = 1.0 if branch == 1 else -1.0
    for block in range(0, size, 8):
        left = block + ((2 * layer) % 8)
        right = block + (((2 * layer) + 1) % 8)
        matrix[left * size + left] = 0.0
        matrix[right * size + right] = 0.0
        matrix[left * size + right] = -sign
        matrix[right * size + left] = sign
    return matrix


def _tensor(shape: list[int], values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    expected = math.prod(shape)
    if len(data) != expected:
        raise ValueError(
            f"tensor data length {len(data)} does not match shape product {expected}"
        )
    if not all(math.isfinite(value) for value in data):
        raise ValueError("tensor data must contain only finite values")
    return {"dtype": "f32", "shape": shape, "data": data}


def build_triality_v2_tensors(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the deterministic F32 tensor bundle referenced by a schema-v2 payload.

    Shape vectors use GGUF/ggml dimension order.  Dimension zero is therefore
    the contiguous dimension (three branches for consensus rows and knot index
    for the NC-KA banks).
    """
    num_layers = int(payload["num_layers"])
    head_dim = int(payload["head_dim"])
    profile = str(payload["profile_id"])
    if num_layers <= 0:
        raise ValueError("schema-v2 num_layers must be positive")
    if head_dim <= 0 or head_dim % 8 != 0:
        raise ValueError("schema-v2 head_dim must be a positive multiple of 8")
    if TRIALITY_PROFILE_ID_PATTERN.fullmatch(profile) is None:
        raise ValueError("schema-v2 profile_id must be a safe non-empty token")
    consensus = payload["consensus"]
    tensors: dict[str, dict[str, Any]] = {}
    for layer in range(num_layers):
        for branch, view in enumerate(TRIALITY_CONSENSUS_VIEWS):
            tensors[rotation_tensor_name(layer, view, profile)] = _tensor(
                [head_dim, head_dim], _rotation_matrix(head_dim, layer, branch)
            )

    for field in ("weights", "bias", "scale", "temperature"):
        rows = [row[field] for row in consensus["rows"]]
        tensors[consensus_tensor_name(field, profile)] = _tensor(
            [TRIALITY_CONSENSUS_BRANCH_COUNT, num_layers],
            (value for row in rows for value in row),
        )

    ncka = payload["ncka"]
    if bool(ncka["enabled"]):
        coordinate_count = len(ncka["coordinate_names"])
        outer_count = int(ncka["outer_count"])
        knot_count = int(ncka["knot_count"])
        coordinate_min = [0.0] * coordinate_count
        coordinate_max = [1.0] * coordinate_count
        knot_axis = [index / (knot_count - 1) for index in range(knot_count)]
        inner_knots = knot_axis * (
            TRIALITY_CONSENSUS_BRANCH_COUNT * outer_count * coordinate_count
        )
        inner_values = [
            (branch + 1) * (outer + 1) * (coordinate + 1) * knot / 10_000.0
            for branch in range(TRIALITY_CONSENSUS_BRANCH_COUNT)
            for outer in range(outer_count)
            for coordinate in range(coordinate_count)
            for knot in knot_axis
        ]
        outer_knots = knot_axis * (TRIALITY_CONSENSUS_BRANCH_COUNT * outer_count)
        outer_values = [
            (branch + 1) * (outer + 1) * knot / 100.0
            for branch in range(TRIALITY_CONSENSUS_BRANCH_COUNT)
            for outer in range(outer_count)
            for knot in knot_axis
        ]
        tensors.update(
            {
                ncka_tensor_name("coordinate_min", profile): _tensor(
                    [coordinate_count], coordinate_min
                ),
                ncka_tensor_name("coordinate_max", profile): _tensor(
                    [coordinate_count], coordinate_max
                ),
                ncka_tensor_name("inner_knots", profile): _tensor(
                    [
                        knot_count,
                        coordinate_count,
                        outer_count,
                        TRIALITY_CONSENSUS_BRANCH_COUNT,
                    ],
                    inner_knots,
                ),
                ncka_tensor_name("inner_values", profile): _tensor(
                    [
                        knot_count,
                        coordinate_count,
                        outer_count,
                        TRIALITY_CONSENSUS_BRANCH_COUNT,
                    ],
                    inner_values,
                ),
                ncka_tensor_name("outer_knots", profile): _tensor(
                    [knot_count, outer_count, TRIALITY_CONSENSUS_BRANCH_COUNT],
                    outer_knots,
                ),
                ncka_tensor_name("outer_values", profile): _tensor(
                    [knot_count, outer_count, TRIALITY_CONSENSUS_BRANCH_COUNT],
                    outer_values,
                ),
                ncka_tensor_name("fallback_weights", profile): _tensor(
                    [TRIALITY_CONSENSUS_BRANCH_COUNT], ncka["fallback_weights"]
                ),
            }
        )
    return tensors


def _tensor_manifest(tensors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "sha256": _sha256_f32(tensor["data"]),
        }
        for name, tensor in sorted(tensors.items())
    }


def _determinant(matrix: list[list[float]]) -> float:
    work = [row[:] for row in matrix]
    determinant = 1.0
    for column in range(len(work)):
        pivot = max(range(column, len(work)), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1.0e-12:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot_value
            for item in range(column + 1, len(work)):
                work[row][item] -= factor * work[column][item]
    return determinant


def validate_rotation_tensor(tensor: dict[str, Any], *, head_dim: int) -> None:
    """Validate a finite, block-diagonal SO(8) rotation tensor."""
    if tensor.get("dtype") != "f32" or tensor.get("shape") != [head_dim, head_dim]:
        raise ValueError("rotation tensor must be F32 [head_dim, head_dim]")
    data = _finite_vector(
        tensor.get("data"), name="rotation tensor", length=head_dim * head_dim
    )
    for row in range(head_dim):
        for column in range(head_dim):
            if row // 8 != column // 8 and abs(data[row * head_dim + column]) > 1.0e-7:
                raise ValueError("rotation tensor must be block diagonal in 8x8 blocks")
    for block_start in range(0, head_dim, 8):
        block = [
            [
                data[(block_start + row) * head_dim + block_start + column]
                for column in range(8)
            ]
            for row in range(8)
        ]
        for left in range(8):
            for right in range(8):
                inner = sum(block[row][left] * block[row][right] for row in range(8))
                expected = 1.0 if left == right else 0.0
                if not math.isclose(inner, expected, rel_tol=0.0, abs_tol=1.0e-5):
                    raise ValueError("rotation tensor block is not orthogonal")
        if not math.isclose(_determinant(block), 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("rotation tensor block determinant must be +1")


def build_triality_v2_extension(
    *,
    head_dim: int,
    num_layers: int,
    profile_id: str = "v2",
    enable_ncka: bool = False,
    enable_urt: bool = False,
) -> dict[str, Any]:
    """Build the deterministic consensus, NC-KA, URT, and tensor-manifest extension."""
    if num_layers <= 0:
        raise ValueError("schema-v2 num_layers must be positive")
    if head_dim <= 0 or head_dim % 8 != 0:
        raise ValueError("schema-v2 head_dim must be a positive multiple of 8")
    if TRIALITY_PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
        raise ValueError("schema-v2 profile_id must be a safe non-empty token")
    rows = [
        {
            "layer": layer,
            "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "bias": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "temperature": [1.0, 1.0, 1.0],
        }
        for layer in range(num_layers)
    ]
    consensus = {
        "schema_version": 1,
        "execution": "attention_logit_consensus",
        "view_count": TRIALITY_CONSENSUS_BRANCH_COUNT,
        "views": list(TRIALITY_CONSENSUS_VIEWS),
        "rows": rows,
        "js_fallback_threshold": 0.20,
        "fallback_policy": "static",
        "fallback_weights": [1.0, 0.0, 0.0],
    }
    normalisation_manifest = {
        "coordinate_names": list(TRIALITY_NCKA_COORDINATE_NAMES),
        "range": [0.0, 1.0],
        "clamp": True,
    }
    ncka = {
        "enabled": enable_ncka,
        "required": False,
        "schema_version": TRIALITY_NCKA_SCHEMA_VERSION if enable_ncka else 0,
        "controller_type": TRIALITY_NCKA_CONTROLLER_TYPE if enable_ncka else "",
        "coordinate_names": list(TRIALITY_NCKA_COORDINATE_NAMES) if enable_ncka else [],
        "outer_count": 2 if enable_ncka else 0,
        "knot_count": 3 if enable_ncka else 0,
        "s3_equivariant": enable_ncka,
        "fallback_policy": "static",
        "fallback_weights": [1.0, 0.0, 0.0],
        "normalisation_sha256": _sha256_text(normalisation_manifest)
        if enable_ncka
        else "",
    }
    operator_word_manifest = {
        "algebra_id": TRIALITY_URT_ALGEBRA_ID,
        "generators": ["e1", "e2", "e3"],
        "words": ["e1", "e2", "e3", "e1*e2", "e2*e3", "e3*e1"],
        "multiplication": "left_associative",
    }
    moment_manifest = {
        "degree": 4,
        "moments": ["mean", "variance", "skewness", "kurtosis"],
    }
    urt = {
        "enabled": enable_urt,
        "schema_version": TRIALITY_URT_SCHEMA_VERSION if enable_urt else 0,
        "abstract_algebra_id": TRIALITY_URT_ALGEBRA_ID if enable_urt else "",
        "operator_word_manifest": operator_word_manifest if enable_urt else {},
        "operator_word_sha256": _sha256_text(operator_word_manifest)
        if enable_urt
        else "",
        "reference_representation": "python_quantised_reference" if enable_urt else "",
        "supported_representations": [
            "python_quantised_reference",
            "llama_cpu_gguf",
            "llama_cuda_gguf",
            "hypura_native",
            "hypura_kobold_worker",
        ]
        if enable_urt
        else [],
        "consistency_tolerance": 1.0e-5 if enable_urt else 0.0,
        "moment_degree": 4 if enable_urt else 0,
        "moment_manifest_sha256": _sha256_text(moment_manifest) if enable_urt else "",
    }
    extension: dict[str, Any] = {
        "profile_id": profile_id,
        "consensus": consensus,
        "ncka": ncka,
        "urt": urt,
    }
    tensors = build_triality_v2_tensors(
        {"head_dim": head_dim, "num_layers": num_layers, **extension}
    )
    extension["tensor_manifest"] = _tensor_manifest(tensors)
    ncka["controller_sha256"] = (
        _sha256_text(
            {
                name: manifest
                for name, manifest in extension["tensor_manifest"].items()
                if f".profile.{profile_id}.ncka." in name
            }
        )
        if enable_ncka
        else ""
    )
    return extension


def triality_v2_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a validated schema-v2 payload into its public GGUF metadata keys."""
    consensus = payload["consensus"]
    ncka = payload["ncka"]
    urt = payload["urt"]

    def flatten(field: str) -> list[float]:
        return [value for row in consensus["rows"] for value in row[field]]

    return {
        "hypura.turboquant.triality.profile_id": payload["profile_id"],
        "hypura.turboquant.triality.execution": consensus["execution"],
        "hypura.turboquant.triality.view_count": consensus["view_count"],
        "hypura.turboquant.triality.views": consensus["views"],
        "hypura.turboquant.triality.weights": flatten("weights"),
        "hypura.turboquant.triality.bias": flatten("bias"),
        "hypura.turboquant.triality.scale": flatten("scale"),
        "hypura.turboquant.triality.temperature": flatten("temperature"),
        "hypura.turboquant.triality.js_fallback_threshold": consensus[
            "js_fallback_threshold"
        ],
        "hypura.turboquant.ncka.enabled": ncka["enabled"],
        "hypura.turboquant.ncka.required": ncka["required"],
        "hypura.turboquant.ncka.schema_version": ncka["schema_version"],
        "hypura.turboquant.ncka.controller_type": ncka["controller_type"],
        "hypura.turboquant.ncka.coordinate_names": ncka["coordinate_names"],
        "hypura.turboquant.ncka.outer_count": ncka["outer_count"],
        "hypura.turboquant.ncka.knot_count": ncka["knot_count"],
        "hypura.turboquant.ncka.s3_equivariant": ncka["s3_equivariant"],
        "hypura.turboquant.ncka.controller_sha256": ncka["controller_sha256"],
        "hypura.turboquant.ncka.normalisation_sha256": ncka["normalisation_sha256"],
        "hypura.turboquant.urt.enabled": urt["enabled"],
        "hypura.turboquant.urt.schema_version": urt["schema_version"],
        "hypura.turboquant.urt.abstract_algebra_id": urt["abstract_algebra_id"],
        "hypura.turboquant.urt.operator_word_manifest": _canonical_json(
            urt["operator_word_manifest"]
        ),
        "hypura.turboquant.urt.operator_word_sha256": urt["operator_word_sha256"],
        "hypura.turboquant.urt.reference_representation": urt[
            "reference_representation"
        ],
        "hypura.turboquant.urt.supported_representations": urt[
            "supported_representations"
        ],
        "hypura.turboquant.urt.consistency_tolerance": urt["consistency_tolerance"],
        "hypura.turboquant.urt.moment_degree": urt["moment_degree"],
        "hypura.turboquant.urt.moment_manifest_sha256": urt["moment_manifest_sha256"],
    }


_METADATA_KEY_EXTENSION = build_triality_v2_extension(head_dim=8, num_layers=1)
TRIALITY_V2_METADATA_KEYS = tuple(triality_v2_metadata(_METADATA_KEY_EXTENSION))


def _finite_vector(
    value: object, *, name: str, length: int | None = None
) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{name} must contain numeric values")
    values = [float(item) for item in value]
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite values")
    return values


def _probability_row(value: object, *, name: str) -> list[float]:
    values = _finite_vector(value, name=name, length=TRIALITY_CONSENSUS_BRANCH_COUNT)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must not contain negative weights")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"{name} must sum to 1")
    return values


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"schema-v2 payload requires a {key} object")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"{name} keys are invalid: {'; '.join(details)}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_triality_v2_payload(payload: dict[str, Any]) -> None:
    """Fail closed on malformed schema-v2 payloads and tensor manifests."""
    consensus = _required_object(payload, "consensus")
    ncka = _required_object(payload, "ncka")
    urt = _required_object(payload, "urt")
    manifest = _required_object(payload, "tensor_manifest")

    profile_id = payload.get("profile_id")
    if (
        not isinstance(profile_id, str)
        or TRIALITY_PROFILE_ID_PATTERN.fullmatch(profile_id) is None
    ):
        raise ValueError("schema-v2 profile_id must be a safe non-empty token")

    _require_exact_keys(
        consensus,
        {
            "schema_version",
            "execution",
            "view_count",
            "views",
            "rows",
            "js_fallback_threshold",
            "fallback_policy",
            "fallback_weights",
        },
        name="consensus",
    )
    if int(consensus["schema_version"]) != 1:
        raise ValueError("unsupported consensus schema_version")

    if int(consensus.get("view_count", 0)) != TRIALITY_CONSENSUS_BRANCH_COUNT:
        raise ValueError("schema-v2 consensus view_count must be 3")
    if consensus.get("views") != list(TRIALITY_CONSENSUS_VIEWS):
        raise ValueError("schema-v2 consensus views must use canonical branch order")
    if consensus.get("execution") != "attention_logit_consensus":
        raise ValueError(
            "schema-v2 consensus execution must be 'attention_logit_consensus'"
        )
    threshold = float(consensus.get("js_fallback_threshold", 0.0))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "schema-v2 js_fallback_threshold must be finite and non-negative"
        )
    _probability_row(
        consensus.get("fallback_weights"), name="consensus.fallback_weights"
    )
    if consensus.get("fallback_policy") != "static":
        raise ValueError("schema-v2 consensus fallback_policy must be 'static'")

    num_layers = int(payload.get("num_layers", 0))
    head_dim = int(payload.get("head_dim", 0))
    if head_dim <= 0 or head_dim % 8 != 0:
        raise ValueError("schema-v2 head_dim must be a positive multiple of 8")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != num_layers:
        raise ValueError(
            "schema-v2 consensus rows must contain exactly one row per layer"
        )
    for layer, row in enumerate(rows):
        if not isinstance(row, dict) or int(row.get("layer", -1)) != layer:
            raise ValueError("schema-v2 consensus row layer indices must be contiguous")
        _require_exact_keys(
            row,
            {"layer", "weights", "bias", "scale", "temperature"},
            name=f"consensus.rows[{layer}]",
        )
        _probability_row(row.get("weights"), name=f"consensus.rows[{layer}].weights")
        _finite_vector(row.get("bias"), name=f"consensus.rows[{layer}].bias", length=3)
        scales = _finite_vector(
            row.get("scale"), name=f"consensus.rows[{layer}].scale", length=3
        )
        temperatures = _finite_vector(
            row.get("temperature"),
            name=f"consensus.rows[{layer}].temperature",
            length=3,
        )
        if any(value <= 0.0 for value in scales):
            raise ValueError("schema-v2 consensus scales must be positive")
        if any(value <= 0.0 for value in temperatures):
            raise ValueError("schema-v2 consensus temperatures must be positive")

    _require_exact_keys(
        ncka,
        {
            "enabled",
            "required",
            "schema_version",
            "controller_type",
            "coordinate_names",
            "outer_count",
            "knot_count",
            "s3_equivariant",
            "fallback_policy",
            "fallback_weights",
            "normalisation_sha256",
            "controller_sha256",
        },
        name="ncka",
    )
    if ncka.get("fallback_policy") != "static":
        raise ValueError("NC-KA fallback_policy must be 'static'")
    _probability_row(ncka.get("fallback_weights"), name="ncka.fallback_weights")
    if bool(ncka.get("enabled")):
        if int(ncka.get("schema_version", 0)) != TRIALITY_NCKA_SCHEMA_VERSION:
            raise ValueError("unsupported NC-KA schema_version")
        supported = ncka.get("controller_type") == TRIALITY_NCKA_CONTROLLER_TYPE
        if not supported and bool(ncka.get("required")):
            raise ValueError("required NC-KA controller type is unsupported")
        if not supported:
            if ncka.get("fallback_policy") != "static":
                raise ValueError(
                    "optional unsupported NC-KA requires explicit static fallback"
                )
        if ncka.get("coordinate_names") != list(TRIALITY_NCKA_COORDINATE_NAMES):
            raise ValueError("enabled NC-KA requires canonical coordinate_names")
        if not bool(ncka.get("s3_equivariant")):
            raise ValueError("finite-moment NC-KA must declare s3_equivariant")
        if int(ncka.get("outer_count", 0)) <= 0 or int(ncka.get("knot_count", 0)) < 2:
            raise ValueError(
                "enabled NC-KA requires positive outer_count and at least two knots"
            )
        for field in ("controller_sha256", "normalisation_sha256"):
            if not _is_sha256(ncka.get(field)):
                raise ValueError(f"enabled NC-KA requires a SHA256 {field}")
        expected_normalisation_sha256 = _sha256_text(
            {
                "coordinate_names": list(TRIALITY_NCKA_COORDINATE_NAMES),
                "range": [0.0, 1.0],
                "clamp": True,
            }
        )
        if ncka["normalisation_sha256"] != expected_normalisation_sha256:
            raise ValueError("NC-KA normalisation hash mismatch")
    else:
        if bool(ncka.get("required")):
            raise ValueError("disabled NC-KA cannot be required")
        if any(
            (
                int(ncka.get("schema_version", 0)) != 0,
                ncka.get("controller_type") != "",
                ncka.get("coordinate_names") != [],
                int(ncka.get("outer_count", 0)) != 0,
                int(ncka.get("knot_count", 0)) != 0,
                bool(ncka.get("s3_equivariant")),
                ncka.get("controller_sha256") != "",
                ncka.get("normalisation_sha256") != "",
            )
        ):
            raise ValueError("disabled NC-KA must use the canonical empty contract")

    _require_exact_keys(
        urt,
        {
            "enabled",
            "schema_version",
            "abstract_algebra_id",
            "operator_word_manifest",
            "operator_word_sha256",
            "reference_representation",
            "supported_representations",
            "consistency_tolerance",
            "moment_degree",
            "moment_manifest_sha256",
        },
        name="urt",
    )
    if bool(urt.get("enabled")):
        if int(urt.get("schema_version", 0)) != TRIALITY_URT_SCHEMA_VERSION:
            raise ValueError("unsupported URT schema_version")
        manifest_value = urt.get("operator_word_manifest")
        if _sha256_text(manifest_value) != urt.get("operator_word_sha256"):
            raise ValueError("URT operator word hash mismatch")
        if urt.get("abstract_algebra_id") != TRIALITY_URT_ALGEBRA_ID:
            raise ValueError("unsupported URT abstract_algebra_id")
        supported_representations = urt.get("supported_representations")
        if (
            not isinstance(supported_representations, list)
            or not supported_representations
        ):
            raise ValueError("URT supported_representations must be non-empty")
        if urt.get("reference_representation") not in supported_representations:
            raise ValueError("URT reference representation must be supported")
        tolerance = float(urt.get("consistency_tolerance", 0.0))
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("URT consistency_tolerance must be positive")
        if int(urt.get("moment_degree", 0)) != 4:
            raise ValueError("URT moment_degree must be 4")
        expected_moment_hash = _sha256_text(
            {"degree": 4, "moments": ["mean", "variance", "skewness", "kurtosis"]}
        )
        if urt.get("moment_manifest_sha256") != expected_moment_hash:
            raise ValueError("URT moment manifest hash mismatch")
    else:
        if any(
            (
                int(urt.get("schema_version", 0)) != 0,
                urt.get("abstract_algebra_id") != "",
                urt.get("operator_word_manifest") != {},
                urt.get("operator_word_sha256") != "",
                urt.get("reference_representation") != "",
                urt.get("supported_representations") != [],
                float(urt.get("consistency_tolerance", 0.0)) != 0.0,
                int(urt.get("moment_degree", 0)) != 0,
                urt.get("moment_manifest_sha256") != "",
            )
        ):
            raise ValueError("disabled URT must use the canonical empty contract")

    expected_tensors = build_triality_v2_tensors(payload)
    expected_manifest = _tensor_manifest(expected_tensors)
    if set(manifest) != set(expected_manifest):
        missing = sorted(set(expected_manifest) - set(manifest))
        extra = sorted(set(manifest) - set(expected_manifest))
        raise ValueError(
            "schema-v2 tensor manifest key set mismatch: "
            f"missing={missing}, unexpected={extra}"
        )
    for name, expected in expected_manifest.items():
        actual = manifest.get(name)
        if actual != expected:
            raise ValueError(f"schema-v2 tensor manifest mismatch for {name}")
    if bool(ncka["enabled"]):
        ncka_manifest = {
            name: item
            for name, item in expected_manifest.items()
            if f".profile.{profile_id}.ncka." in name
        }
        if ncka["controller_sha256"] != _sha256_text(ncka_manifest):
            raise ValueError("NC-KA controller hash mismatch")


def validate_triality_v2_metadata(
    metadata: dict[str, Any], payload: dict[str, Any]
) -> None:
    missing = [key for key in TRIALITY_V2_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"Missing Triality schema-v2 metadata keys: {', '.join(missing)}"
        )
    expected = triality_v2_metadata(payload)
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(
                f"Triality schema-v2 metadata does not match payload: {key}"
            )


__all__ = [
    "TRIALITY_CONSENSUS_VIEWS",
    "TRIALITY_NCKA_CONTROLLER_TYPE",
    "TRIALITY_NCKA_COORDINATE_NAMES",
    "TRIALITY_PAYLOAD_FORMAT_V2",
    "TRIALITY_SCHEMA_V2",
    "TRIALITY_V2_METADATA_KEYS",
    "build_triality_v2_extension",
    "build_triality_v2_tensors",
    "consensus_tensor_name",
    "ncka_tensor_name",
    "rotation_tensor_name",
    "tensor_sha256",
    "triality_v2_metadata",
    "validate_rotation_tensor",
    "validate_triality_v2_metadata",
    "validate_triality_v2_payload",
]
