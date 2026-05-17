from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.research_extension.triality_proxy import triality_proxy_adapter
from turboquant.rotation import so8_block_diagonal_rotation_metrics


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "research_extension"
    / "triality_full_train_prod_bf16"
    / "metrics"
    / "triality_rotation_manifest.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "_docs" / "assets" / "2026-05-17-triality-so8-audit"
REQUIRED_VIEWS = ("vector", "spinor_plus_proxy", "spinor_minus_proxy")
DEFAULT_ORTHOGONALITY_ATOL = 1e-2
DEFAULT_DETERMINANT_ATOL = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Triality learned SO(8) rotation artifacts by bit and view."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orthogonality-atol", type=float, default=DEFAULT_ORTHOGONALITY_ATOL)
    parser.add_argument("--determinant-atol", type=float, default=DEFAULT_DETERMINANT_ATOL)
    parser.add_argument("--z-threshold", type=float, default=6.0)
    return parser.parse_args()


def _load_rotation(path: Path) -> tuple[torch.Tensor, str]:
    """Load a rotation tensor and fail loudly on unsupported artifact shapes."""

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "rotation" not in obj:
            raise ValueError(f"Rotation dict at {path} is missing key 'rotation'")
        obj = obj["rotation"]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor rotation at {path}, got {type(obj).__name__}")
    if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:
        raise ValueError(f"Expected square 2D rotation at {path}, got {tuple(obj.shape)}")
    if obj.shape[0] % 8 != 0:
        raise ValueError(f"Rotation dimension must be divisible by 8 at {path}, got {obj.shape[0]}")
    storage_dtype = str(obj.dtype).replace("torch.", "")
    return obj.to(dtype=torch.float64), storage_dtype


def _block_rows(rotation: torch.Tensor, adapter: torch.Tensor) -> list[dict[str, float | int]]:
    """Return per-SO(8)-block metrics for a learned and view-composed rotation."""

    block_rows: list[dict[str, float | int]] = []
    eye = torch.eye(8, dtype=torch.float64)
    num_blocks = rotation.shape[0] // 8
    for block_index in range(num_blocks):
        start = block_index * 8
        block = rotation[start : start + 8, start : start + 8]
        effective = adapter @ block
        learned_det = float(torch.linalg.det(block).item())
        effective_det = float(torch.linalg.det(effective).item())
        block_rows.append(
            {
                "block": block_index,
                "learned_orthogonality_error": float((block.T @ block - eye).abs().max().item()),
                "learned_determinant": learned_det,
                "learned_determinant_error": abs(learned_det - 1.0),
                "effective_orthogonality_error": float((effective.T @ effective - eye).abs().max().item()),
                "effective_determinant": effective_det,
                "effective_determinant_error": abs(effective_det - 1.0),
            }
        )
    return block_rows


def audit_manifest(
    manifest_path: Path,
    *,
    orthogonality_atol: float = DEFAULT_ORTHOGONALITY_ATOL,
    determinant_atol: float = DEFAULT_DETERMINANT_ATOL,
    z_threshold: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit all Triality rotations listed in a manifest.

    The detailed frame is one row per layer/bit/view/block. The summary frame is
    one row per bit/view and is the README-facing view.
    """

    manifest = pd.read_csv(manifest_path)
    missing_views = sorted(set(REQUIRED_VIEWS) - set(manifest["view"].unique()))
    if missing_views:
        raise ValueError(f"Manifest is missing required Triality views: {missing_views}")

    rows: list[dict[str, float | int | str]] = []
    root = REPO_ROOT
    for manifest_row in manifest.to_dict(orient="records"):
        view = str(manifest_row["view"])
        adapter = triality_proxy_adapter(view, dtype=torch.float64)
        rotation_path = root / str(manifest_row["rotation_path"])
        rotation, storage_dtype = _load_rotation(rotation_path)
        learned_ortho, learned_det_err = so8_block_diagonal_rotation_metrics(rotation)
        effective_ortho, effective_det_err = so8_block_diagonal_rotation_metrics(
            torch.block_diag(
                *[
                    adapter @ rotation[i : i + 8, i : i + 8]
                    for i in range(0, rotation.shape[0], 8)
                ]
            )
        )
        for block_row in _block_rows(rotation, adapter):
            rows.append(
                {
                    "layer": int(manifest_row["layer"]),
                    "bits": float(manifest_row["bits"]),
                    "bit_setting": str(manifest_row["bit_setting"]),
                    "view": view,
                    "mode": str(manifest_row["mode"]),
                    "rotation_path": str(manifest_row["rotation_path"]),
                    "rotation_seed": int(manifest_row["rotation_seed"]),
                    "qjl_seed": int(manifest_row["qjl_seed"]),
                    "storage_dtype": storage_dtype,
                    "rotation_dim": int(rotation.shape[0]),
                    "learned_rotation_orthogonality_error": float(learned_ortho),
                    "learned_rotation_determinant_error_max": float(learned_det_err),
                    "effective_rotation_orthogonality_error": float(effective_ortho),
                    "effective_rotation_determinant_error_max": float(effective_det_err),
                    **block_row,
                }
            )

    detail = pd.DataFrame(rows)
    group_cols = ["bits", "bit_setting", "view", "mode"]
    summary = (
        detail.groupby(group_cols, dropna=False)
        .agg(
            layers=("layer", "nunique"),
            blocks=("block", "count"),
            storage_dtypes=("storage_dtype", lambda values: ",".join(sorted(set(values)))),
            max_learned_orthogonality_error=("learned_orthogonality_error", "max"),
            mean_learned_orthogonality_error=("learned_orthogonality_error", "mean"),
            sd_learned_orthogonality_error=("learned_orthogonality_error", "std"),
            max_effective_orthogonality_error=("effective_orthogonality_error", "max"),
            mean_effective_orthogonality_error=("effective_orthogonality_error", "mean"),
            sd_effective_orthogonality_error=("effective_orthogonality_error", "std"),
            min_effective_determinant=("effective_determinant", "min"),
            mean_effective_determinant=("effective_determinant", "mean"),
            max_effective_determinant=("effective_determinant", "max"),
            max_effective_determinant_error=("effective_determinant_error", "max"),
            mean_effective_determinant_error=("effective_determinant_error", "mean"),
            sd_effective_determinant_error=("effective_determinant_error", "std"),
        )
        .reset_index()
        .sort_values(["bits", "view"])
    )
    summary["status"] = "pass"
    summary.loc[
        (summary["max_effective_orthogonality_error"] > orthogonality_atol)
        | (summary["max_effective_determinant_error"] > determinant_atol),
        "status",
    ] = "fail"

    for metric in ("effective_orthogonality_error", "effective_determinant_error"):
        mean = float(detail[metric].mean())
        std = float(detail[metric].std(ddof=0))
        z_col = f"{metric}_z"
        detail[z_col] = 0.0 if std == 0.0 else (detail[metric] - mean) / std
    detail["is_outlier"] = (
        (detail["effective_orthogonality_error"] > orthogonality_atol)
        | (detail["effective_determinant_error"] > determinant_atol)
        | (detail["effective_orthogonality_error_z"].abs() > z_threshold)
        | (detail["effective_determinant_error_z"].abs() > z_threshold)
    )
    outliers = detail[detail["is_outlier"]]
    status = {
        "manifest": str(manifest_path),
        "orthogonality_atol": orthogonality_atol,
        "determinant_atol": determinant_atol,
        "z_threshold": z_threshold,
        "rows": int(len(detail)),
        "outliers": int(len(outliers)),
        "status": "pass" if len(outliers) == 0 and (summary["status"] == "pass").all() else "fail",
        "max_effective_orthogonality_error": float(detail["effective_orthogonality_error"].max()),
        "max_effective_determinant_error": float(detail["effective_determinant_error"].max()),
    }
    return detail, summary, status


def _format_float(value: float) -> str:
    if abs(value) == 0.0:
        return "0.000e+00"
    return f"{value:.3e}"


def write_markdown_summary(summary: pd.DataFrame, status: dict[str, Any], output_path: Path) -> None:
    """Write a compact README-friendly SO(8) audit table."""

    lines = [
        "# Triality SO(8) Rotation Audit",
        "",
        f"- Status: `{status['status']}`",
        f"- Rows audited: `{status['rows']}`",
        f"- Outliers: `{status['outliers']}`",
        f"- Orthogonality threshold: `{status['orthogonality_atol']}`",
        f"- Determinant threshold: `{status['determinant_atol']}`",
        "",
        "| Bits | View | Layers | Blocks | Dtype | max orth err | mean det | max det err | Status |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['bit_setting']} | "
            f"`{row['view']}` | "
            f"{int(row['layers'])} | "
            f"{int(row['blocks'])} | "
            f"`{row['storage_dtypes']}` | "
            f"{_format_float(float(row['max_effective_orthogonality_error']))} | "
            f"{float(row['mean_effective_determinant']):.12f} | "
            f"{_format_float(float(row['max_effective_determinant_error']))} | "
            f"`{row['status']}` |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write orthogonality and determinant drift plots grouped by bit/view."""

    output_dir.mkdir(parents=True, exist_ok=True)
    views = list(REQUIRED_VIEWS)
    colors = {
        "vector": "#1f77b4",
        "spinor_plus_proxy": "#2ca02c",
        "spinor_minus_proxy": "#d62728",
    }
    bit_values = sorted(summary["bits"].unique())

    def _plot_metric(metric: str, ylabel: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for view in views:
            view_frame = summary[summary["view"] == view].sort_values("bits")
            ax.plot(
                view_frame["bits"],
                view_frame[metric],
                marker="o",
                linewidth=2,
                label=view,
                color=colors[view],
            )
        ax.set_yscale("log")
        ax.set_xticks(bit_values)
        ax.set_xlabel("KV bit setting")
        ax.set_ylabel(ylabel)
        ax.set_title("Triality learned SO(8) audit by bit and view")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False, ncols=1)
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    ortho_plot = _plot_metric(
        "max_effective_orthogonality_error",
        "max effective |R^T R - I|",
        "triality_so8_orthogonality_by_bit_view.png",
    )
    det_plot = _plot_metric(
        "max_effective_determinant_error",
        "max effective |det(B) - 1|",
        "triality_so8_determinant_by_bit_view.png",
    )
    return ortho_plot, det_plot


def main() -> int:
    args = parse_args()
    detail, summary, status = audit_manifest(
        args.manifest,
        orthogonality_atol=args.orthogonality_atol,
        determinant_atol=args.determinant_atol,
        z_threshold=args.z_threshold,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "triality_so8_rotation_audit_detail.csv", index=False)
    summary.to_csv(output_dir / "triality_so8_rotation_audit_summary.csv", index=False)
    (output_dir / "triality_so8_rotation_audit_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(summary, status, output_dir / "triality_so8_rotation_audit_summary.md")
    ortho_plot, det_plot = write_plots(summary, output_dir)
    print(f"status={status['status']}")
    print(f"detail={output_dir / 'triality_so8_rotation_audit_detail.csv'}")
    print(f"summary={output_dir / 'triality_so8_rotation_audit_summary.csv'}")
    print(f"orthogonality_plot={ortho_plot}")
    print(f"determinant_plot={det_plot}")
    if status["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
