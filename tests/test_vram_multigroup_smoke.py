"""Smoke test for research_vram_multigroup_qwen.py (CPU; optional triality rotation dir)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

from tests.test_capture import build_capture_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_vram_multigroup_qwen.py"
SPEC = importlib.util.spec_from_file_location("research_vram_multigroup_qwen", SCRIPT)
assert SPEC and SPEC.loader
_mod = importlib.util.module_from_spec(SPEC)
sys.modules["research_vram_multigroup_qwen"] = _mod
SPEC.loader.exec_module(_mod)


def test_vram_multigroup_script_cpu_smoke(tmp_path: Path) -> None:
    root = tmp_path / "kv"
    build_capture_dir(root, "p1")
    out = tmp_path / "out"
    argv = [
        "research_vram_multigroup_qwen.py",
        "--kv-dir",
        str(root),
        "--modes",
        "exact,multiscreen_relevance",
        "--trials",
        "1",
        "--eval-device",
        "cpu",
        "--output-dir",
        str(out),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = _mod.main()
    finally:
        sys.argv = old
    assert code == 0
    summary = pd.read_csv(out / "metrics" / "vram_summary_by_mode.csv")
    assert len(summary) == 2
    assert set(summary["mode"]) == {"exact", "multiscreen_relevance"}
    assert (out / "metrics" / "vram_summary_by_mode.md").is_file()
    plots = out / "plots"
    assert (plots / "vram_peak_mb_by_mode.png").is_file()
    assert (plots / "kv_memory_ratio_by_mode.png").is_file()
    assert (plots / "hidden_cosine_similarity_by_mode.png").is_file()


def _write_minimal_vector_rotation(path: Path, *, layer: int, bits: float) -> None:
    torch.save(
        {
            "layer": layer,
            "bits": bits,
            "view": "vector",
            "rotation": torch.eye(8, dtype=torch.float32),
            "rotation_seed": 1,
            "qjl_seed": 2,
        },
        path,
    )


def test_vram_multigroup_multiscreen_triality_cpu_smoke(tmp_path: Path) -> None:
    root = tmp_path / "kv"
    build_capture_dir(root, "p1")
    rot = tmp_path / "rotations"
    rot.mkdir(parents=True, exist_ok=True)
    _write_minimal_vector_rotation(rot / "L0_b3_vector.pt", layer=0, bits=3.0)
    out = tmp_path / "out"
    argv = [
        "research_vram_multigroup_qwen.py",
        "--kv-dir",
        str(root),
        "--modes",
        "exact,multiscreen_triality_vector",
        "--bits",
        "3",
        "--trials",
        "1",
        "--eval-device",
        "cpu",
        "--rotation-dir",
        str(rot),
        "--output-dir",
        str(out),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = _mod.main()
    finally:
        sys.argv = old
    assert code == 0
    summary = pd.read_csv(out / "metrics" / "vram_summary_by_mode.csv")
    assert len(summary) == 2
    assert set(summary["mode"]) == {"exact", "multiscreen_triality_vector"}
