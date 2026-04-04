"""Smoke-test ``research_validate_multiscreen_kv.py`` on a tiny synthetic capture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from tests.test_capture import build_capture_dir

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_validate_multiscreen_kv.py"
SPEC = importlib.util.spec_from_file_location("research_validate_multiscreen_kv", SCRIPT_PATH)
assert SPEC and SPEC.loader
_mod = importlib.util.module_from_spec(SPEC)
sys.modules["research_validate_multiscreen_kv"] = _mod
SPEC.loader.exec_module(_mod)


def test_multiscreen_script_exact_and_multiscreen_modes(tmp_path: Path) -> None:
    root = tmp_path / "kv"
    build_capture_dir(root, "p1")
    out = tmp_path / "out"
    for mode in ("exact", "multiscreen_relevance"):
        argv = [
            "research_validate_multiscreen_kv.py",
            "--captured-dir",
            str(root),
            "--mode",
            mode,
            "--bits",
            "3",
            "--trials",
            "1",
            "--output-dir",
            str(out / mode),
        ]
        old = sys.argv
        try:
            sys.argv = argv
            code = _mod.main()
        finally:
            sys.argv = old
        assert code == 0
        csv_path = out / mode / "metrics" / "multiscreen_kv_trials.csv"
        frame = pd.read_csv(csv_path)
        assert not frame.empty
        assert (frame["mode"] == mode).all()
