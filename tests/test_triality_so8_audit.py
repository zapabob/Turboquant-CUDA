from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_triality_so8_rotations.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_triality_so8_rotations", AUDIT_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, *, bad_det: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    views = ("vector", "spinor_plus_proxy", "spinor_minus_proxy")
    for view_index, view in enumerate(views):
        rotation = torch.eye(8, dtype=torch.float64)
        if bad_det and view == "spinor_minus_proxy":
            rotation[0, 0] = -1.0
        rotation_path = tmp_path / f"layer_00_bits_2_{view}.pt"
        torch.save(rotation, rotation_path)
        rows.append(
            {
                "layer": 0,
                "bits": 2.0,
                "bit_setting": "2",
                "view": view,
                "mode": f"mode_{view_index}",
                "rotation_path": str(rotation_path),
                "rotation_seed": 100 + view_index,
                "qjl_seed": 200 + view_index,
            }
        )
    manifest_path = tmp_path / "triality_rotation_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_audit_manifest_passes_identity_views(tmp_path: Path) -> None:
    module = _load_audit_module()
    manifest_path = _write_manifest(tmp_path)

    detail, summary, status = module.audit_manifest(manifest_path)

    assert status["status"] == "pass"
    assert status["outliers"] == 0
    assert set(summary["view"]) == {"vector", "spinor_plus_proxy", "spinor_minus_proxy"}
    assert detail["effective_orthogonality_error"].max() == 0.0
    assert detail["effective_determinant_error"].max() == 0.0


def test_audit_manifest_rejects_det_outlier(tmp_path: Path) -> None:
    module = _load_audit_module()
    manifest_path = _write_manifest(tmp_path, bad_det=True)

    _detail, summary, status = module.audit_manifest(manifest_path)

    assert status["status"] == "fail"
    assert status["outliers"] > 0
    assert "fail" in set(summary["status"])
