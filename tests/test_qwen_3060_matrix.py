from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from tests.test_capture import build_capture_dir
from turboquant.analysis import (
    QWEN_3060_MATRIX_MODES,
    QWEN_3060_PAIRWISE_BASELINES,
    QWEN_3060_STAT_METRICS,
    compute_qwen_3060_multigroup_statistics,
)
from turboquant.allocation import ChannelBitAllocation
from turboquant.research_extension.k_triality import TrialityRotationArtifact, save_triality_proxy_rotations
from turboquant.schema import build_turboquant_artifact_metadata


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_qwen_3060_matrix.py"
SPEC = importlib.util.spec_from_file_location("validate_qwen_3060_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_rotation_dir(path: Path, *, layer_idx: int, bits: float) -> Path:
    metadata = build_turboquant_artifact_metadata(
        total_bits=bits,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=8,
        allocation=ChannelBitAllocation.preset(effective_bits=bits - 1.0, width=8) if bits - 1.0 in {1.5, 2.5, 3.5} else None,
    )
    save_triality_proxy_rotations(
        [
            TrialityRotationArtifact(
                layer_idx=layer_idx,
                bits=bits,
                view="vector",
                rotation=torch.eye(8, dtype=torch.float32),
                rotation_seed=17,
                qjl_seed=71,
                metadata=metadata,
            )
        ],
        path,
    )
    return path


def test_qwen_3060_statistics_emit_required_groups() -> None:
    rows = []
    for trial in (0, 1, 2):
        for mode_index, mode in enumerate(QWEN_3060_MATRIX_MODES):
            rows.append(
                {
                    "dataset": "qwen3060:test",
                    "trial": trial,
                    "layer": 0,
                    "mode": mode,
                    "bit_setting": "3",
                    "bits": 3.0,
                    "model_name": "Qwen/Qwen3.5-9B",
                    "tokenizer_name": "Qwen/Qwen3.5-9B",
                    "prompt_hash": "hash",
                    "capture_id": "cap",
                    "lane_name": "rtx3060_desktop_12gb",
                    "logit_cosine_similarity": 0.99 - (0.005 * mode_index),
                    "next_logit_kl": 0.01 + (0.002 * mode_index),
                    "hidden_cosine_similarity": 0.98 - (0.01 * mode_index),
                    "attention_output_relative_error": 0.01 + (0.003 * mode_index),
                    "memory_ratio_vs_exact": 1.0 if mode == "exact" else 0.7 - (0.05 * mode_index),
                    "prefill_seconds": 0.01,
                    "decode_seconds": 0.02,
                    "peak_vram_mb": 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    friedman, pairwise = compute_qwen_3060_multigroup_statistics(frame)
    assert not friedman.empty
    assert not pairwise.empty
    assert set(friedman["metric"]) == set(QWEN_3060_STAT_METRICS)
    assert set(QWEN_3060_PAIRWISE_BASELINES).issubset(set(pairwise["baseline_mode"]))


def test_validate_qwen_3060_matrix_script_writes_outputs(tmp_path: Path) -> None:
    kv_root = tmp_path / "captures"
    build_capture_dir(kv_root, "prompt-a")
    rotation_dir = _write_rotation_dir(tmp_path / "rotations", layer_idx=0, bits=3.0)
    output_dir = tmp_path / "out"

    argv = [
        "validate_qwen_3060_matrix.py",
        "--kv-dir",
        str(kv_root),
        "--rotation-dir",
        str(rotation_dir),
        "--output-dir",
        str(output_dir),
        "--eval-device",
        "cpu",
        "--bits",
        "3",
        "--trials",
        "3",
        "--max-layers",
        "1",
        "--skip-plots",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = MODULE.main()
    finally:
        sys.argv = old

    assert code == 0
    trial_csv = output_dir / "metrics" / "qwen_3060_matrix_trials.csv"
    summary_csv = output_dir / "metrics" / "qwen_3060_matrix_summary.csv"
    friedman_csv = output_dir / "metrics" / "qwen_3060_matrix_friedman.csv"
    pairwise_csv = output_dir / "metrics" / "qwen_3060_matrix_pairwise.csv"
    run_meta = output_dir / "metrics" / "qwen_3060_matrix_run_meta.json"

    assert trial_csv.exists()
    assert summary_csv.exists()
    assert friedman_csv.exists()
    assert pairwise_csv.exists()
    assert run_meta.exists()

    trial_frame = pd.read_csv(trial_csv)
    assert set(QWEN_3060_MATRIX_MODES).issubset(set(trial_frame["mode"]))
    assert set(trial_frame["lane_name"]) == {"rtx3060_desktop_12gb"}

    payload = json.loads(run_meta.read_text(encoding="utf-8"))
    assert payload["lane_name"] == "rtx3060_desktop_12gb"
    assert payload["mode_count"] == len(QWEN_3060_MATRIX_MODES)
