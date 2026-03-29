from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from tests.test_capture import build_capture_dir
from turboquant.research_extension.k_triality import (
    TRIALITY_PROXY_VIEWS,
    compute_triality_statistics,
    evaluate_triality_proxy_captured,
    load_triality_proxy_rotations,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_validate_k_triality.py"
SPEC = importlib.util.spec_from_file_location("research_validate_k_triality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_eval_checkpoint_writes_status_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "eval_run.log"
    logger = MODULE.RunLogger(log_path)
    checkpoint = MODULE.EvalCheckpoint(tmp_path, logger)
    checkpoint.start("demo_stage")
    checkpoint.complete("demo_stage")
    payload = json.loads((tmp_path / "eval_status.json").read_text(encoding="utf-8"))
    assert payload["last_completed_stage"] == "demo_stage"
    assert payload["failed_stage"] is None
    assert log_path.exists()
    assert "start=demo_stage" in log_path.read_text(encoding="utf-8")


def test_log_line_swallows_oserror_on_stdout(tmp_path: Path) -> None:
    log_path = tmp_path / "x.log"

    def bad_print(*_a, **_k):
        raise OSError(22, "Invalid argument")

    with patch("builtins.print", bad_print):
        MODULE._log_line(log_path, "hello", also_stdout=True)
    assert "hello" in log_path.read_text(encoding="utf-8")


def test_rolling_checkpoint_rotates_slots(tmp_path: Path) -> None:
    status = tmp_path / "eval_status.json"
    status.write_text('{"ok": true}', encoding="utf-8")
    partial = tmp_path / "triality_trials_partial.csv"
    partial.write_text("a,b\n1,2\n", encoding="utf-8")
    cp_root = tmp_path / "checkpoints"
    mgr = MODULE.RollingCheckpointManager(
        checkpoints_root=cp_root,
        status_src=status,
        partial_csv_src=partial,
        interval_seconds=0.05,
        n_slots=3,
    )
    mgr.start()
    time.sleep(0.35)
    mgr.stop(final_snapshot=True)
    metas = list(cp_root.glob("cp_*"))
    assert metas, "expected at least one checkpoint slot directory"
    meta_path = cp_root / "rolling_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "slot" in meta and "generation" in meta


def _write_triality_rotation_pt(path: Path, *, layer: int, bits: float, view: str) -> None:
    payload = {
        "layer": layer,
        "bits": bits,
        "view": view,
        "rotation": torch.eye(8, dtype=torch.float32),
        "rotation_seed": 1,
        "qjl_seed": 2,
    }
    torch.save(payload, path)


def _write_triality_rotation_pt_with_dim(path: Path, *, layer: int, bits: float, view: str, dim: int) -> None:
    payload = {
        "layer": layer,
        "bits": bits,
        "view": view,
        "rotation": torch.eye(dim, dtype=torch.float32),
        "rotation_seed": 1,
        "qjl_seed": 2,
    }
    torch.save(payload, path)


def test_evaluate_triality_rejects_rotation_head_dim_mismatch(tmp_path: Path) -> None:
    build_capture_dir(tmp_path, "prompt-a")
    rot_dir = tmp_path / "rotations"
    rot_dir.mkdir(parents=True, exist_ok=True)
    for view in TRIALITY_PROXY_VIEWS:
        _write_triality_rotation_pt_with_dim(
            rot_dir / f"layer_00_bits_2_{view}.pt", layer=0, bits=2.0, view=view, dim=16
        )
    with pytest.raises(ValueError, match="do not match captured key head dimension"):
        evaluate_triality_proxy_captured(
            kv_root=tmp_path,
            trial_count=1,
            bit_grid=[2.0],
            rotation_dir=rot_dir,
            max_layers=0,
            eval_device="cpu",
        )


def test_evaluate_triality_writes_partial_after_first_bundle(tmp_path: Path) -> None:
    build_capture_dir(tmp_path, "prompt-a")
    build_capture_dir(tmp_path, "prompt-b")
    rot_dir = tmp_path / "rotations"
    rot_dir.mkdir(parents=True, exist_ok=True)
    for view in TRIALITY_PROXY_VIEWS:
        _write_triality_rotation_pt(rot_dir / f"L0_b2_{view}.pt", layer=0, bits=2.0, view=view)
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    partial = metrics / "triality_trials_partial.csv"
    state = metrics / "eval_resume_state.json"

    class AbortOne(Exception):
        pass

    bundle_finishes = {"n": 0}

    def on_done():
        bundle_finishes["n"] += 1
        if bundle_finishes["n"] >= 1:
            raise AbortOne()

    with pytest.raises(AbortOne):
        evaluate_triality_proxy_captured(
            kv_root=tmp_path,
            trial_count=1,
            bit_grid=[2.0],
            rotation_dir=rot_dir,
            max_layers=0,
            eval_device="cpu",
            metrics_dir=metrics,
            resume=False,
            force_fresh=True,
            on_bundle_done=on_done,
        )
    assert partial.exists()
    assert state.exists()


def test_evaluate_triality_resume_matches_full_run(tmp_path: Path) -> None:
    build_capture_dir(tmp_path, "prompt-a")
    build_capture_dir(tmp_path, "prompt-b")
    rot_dir = tmp_path / "rotations"
    rot_dir.mkdir(parents=True, exist_ok=True)
    for view in TRIALITY_PROXY_VIEWS:
        _write_triality_rotation_pt(rot_dir / f"L0_b2_{view}.pt", layer=0, bits=2.0, view=view)

    metrics_resume = tmp_path / "metrics_resume"
    metrics_resume.mkdir(parents=True, exist_ok=True)

    class AbortOne(Exception):
        pass

    bundle_finishes = {"n": 0}

    def on_done():
        bundle_finishes["n"] += 1
        if bundle_finishes["n"] >= 1:
            raise AbortOne()

    with pytest.raises(AbortOne):
        evaluate_triality_proxy_captured(
            kv_root=tmp_path,
            trial_count=1,
            bit_grid=[2.0],
            rotation_dir=rot_dir,
            max_layers=0,
            eval_device="cpu",
            metrics_dir=metrics_resume,
            resume=False,
            force_fresh=True,
            on_bundle_done=on_done,
        )

    n_partial = len(pd.read_csv(metrics_resume / "triality_trials_partial.csv"))
    assert n_partial > 0

    trial_resumed, _ = evaluate_triality_proxy_captured(
        kv_root=tmp_path,
        trial_count=1,
        bit_grid=[2.0],
        rotation_dir=rot_dir,
        max_layers=0,
        eval_device="cpu",
        metrics_dir=metrics_resume,
        resume=True,
        force_fresh=False,
    )

    metrics_fresh = tmp_path / "metrics_fresh"
    metrics_fresh.mkdir(parents=True, exist_ok=True)
    trial_full, _ = evaluate_triality_proxy_captured(
        kv_root=tmp_path,
        trial_count=1,
        bit_grid=[2.0],
        rotation_dir=rot_dir,
        max_layers=0,
        eval_device="cpu",
        metrics_dir=metrics_fresh,
        resume=False,
        force_fresh=True,
    )

    assert len(trial_resumed) == len(trial_full)
    assert len(trial_resumed) > n_partial


def test_load_triality_proxy_rotations_requires_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No triality rotation artifacts"):
        load_triality_proxy_rotations(tmp_path)


def test_compute_triality_statistics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty trial_frame"):
        compute_triality_statistics(pd.DataFrame())


def test_summarize_from_trials_reconstructs_summary_and_stats() -> None:
    rows = []
    for trial in (0, 1):
        for mode, hidden, kl in (
            ("key_only_block_so8_learned", 0.90, 0.10),
            ("key_only_block_so8_triality_vector", 0.91, 0.09),
            ("key_only_block_so8_triality_plus", 0.92, 0.08),
            ("key_only_block_so8_triality_minus", 0.89, 0.11),
            ("key_only_random", 0.88, 0.12),
            ("key_only_block_so8_static", 0.87, 0.13),
            ("full_kv", 0.80, 0.20),
        ):
            rows.append(
                {
                    "dataset": "research_captured:test",
                    "trial": trial,
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
    summary_frame, mean_pm_sd, stats_frame, friedman_frame, pairwise_frame = MODULE.summarize_from_trials(trial_frame)
    assert not summary_frame.empty
    assert not mean_pm_sd.empty
    assert not stats_frame.empty
    assert not friedman_frame.empty
    assert not pairwise_frame.empty
    assert set(stats_frame["mode"]) >= {
        "key_only_block_so8_triality_vector",
        "key_only_block_so8_triality_plus",
        "key_only_block_so8_triality_minus",
    }
