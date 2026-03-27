from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from turboquant.research_extension.k_triality import compute_triality_statistics, load_triality_proxy_rotations


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_validate_k_triality.py"
SPEC = importlib.util.spec_from_file_location("research_validate_k_triality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_eval_checkpoint_writes_status_file(tmp_path: Path) -> None:
    checkpoint = MODULE.EvalCheckpoint(tmp_path)
    checkpoint.start("demo_stage")
    checkpoint.complete("demo_stage")
    payload = json.loads((tmp_path / "eval_status.json").read_text(encoding="utf-8"))
    assert payload["last_completed_stage"] == "demo_stage"
    assert payload["failed_stage"] is None


def test_load_triality_proxy_rotations_requires_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No triality rotation artifacts"):
        load_triality_proxy_rotations(tmp_path)


def test_compute_triality_statistics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty trial_frame"):
        compute_triality_statistics(pd.DataFrame())


def test_summarize_from_trials_reconstructs_summary_and_stats() -> None:
    rows = []
    for mode, hidden, kl in (
        ("key_only_block_so8_learned", 0.90, 0.10),
        ("key_only_block_so8_triality_vector", 0.91, 0.09),
        ("key_only_block_so8_triality_plus", 0.92, 0.08),
        ("key_only_block_so8_triality_minus", 0.89, 0.11),
        ("key_only_random", 0.88, 0.12),
        ("full_kv", 0.80, 0.20),
    ):
        rows.append(
            {
                "dataset": "research_captured:test",
                "trial": 0,
                "layer": 0,
                "mode": mode,
                "bit_setting": "2",
                "bits": 2.0,
                "capture_id": "cap",
                "prompt_label": "test",
                "prompt_hash": "hash",
                "logit_cosine_similarity": 0.99,
                "logit_mae": 0.1,
                "logit_mse": 0.2,
                "next_logit_kl": kl,
                "logit_spearman": 0.95,
                "logit_top1_match": 1.0,
                "logit_top5_match": 1.0,
                "logit_top5_overlap": 1.0,
                "hidden_cosine_similarity": hidden,
                "hidden_mae": 0.1,
                "hidden_mse": 0.2,
                "attention_output_relative_error": 0.3,
                "memory_bits": 10.0,
                "memory_ratio_vs_exact": 0.5,
                "prefill_seconds": 0.01,
                "decode_seconds": 0.02,
                "peak_vram_mb": 0.0,
                "key_mode": mode,
                "value_mode": "exact",
                "value_rotation_policy": "exact",
                "rotation_seed": 1,
                "qjl_seed": 2,
            }
        )
    trial_frame = pd.DataFrame(rows)
    summary_frame, mean_pm_sd, stats_frame = MODULE.summarize_from_trials(trial_frame)
    assert not summary_frame.empty
    assert not mean_pm_sd.empty
    assert not stats_frame.empty
    assert set(stats_frame["mode"]) >= {
        "key_only_block_so8_triality_vector",
        "key_only_block_so8_triality_plus",
        "key_only_block_so8_triality_minus",
    }
