from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.schema import build_turboquant_artifact_metadata
from tests.test_capture import build_capture_dir
from turboquant.research_extension.k_triality import (
    TRIALITY_PROXY_VIEWS,
    TrialityRotationArtifact,
    _bundle_keys_filtered,
    _stable_sort_bundles,
    compute_triality_statistics,
    evaluate_triality_proxy_captured,
    load_captured_runs,
    load_triality_proxy_rotations,
    save_triality_proxy_rotations,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_validate_k_triality.py"
SPEC = importlib.util.spec_from_file_location("research_validate_k_triality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_train_k_triality.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("research_train_k_triality", TRAIN_SCRIPT_PATH)
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)


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


def test_bundle_keys_match_for_relative_and_absolute_kv_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume state must not depend on whether --kv-dir was passed relative or absolute."""
    monkeypatch.chdir(tmp_path)
    cap = tmp_path / "kv_captures"
    build_capture_dir(cap, "prompt-a")
    rel_bundles = _stable_sort_bundles(load_captured_runs(Path("kv_captures")))
    abs_bundles = _stable_sort_bundles(load_captured_runs(cap.resolve()))
    assert _bundle_keys_filtered(rel_bundles) == _bundle_keys_filtered(abs_bundles)
    assert all(":" in k for k in _bundle_keys_filtered(rel_bundles))
    assert str(tmp_path).replace("\\", "/") in _bundle_keys_filtered(rel_bundles)[0]


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


def test_save_and_load_triality_proxy_rotations_preserve_explicit_metadata(tmp_path: Path) -> None:
    metadata = build_turboquant_artifact_metadata(
        total_bits=2.5,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=8,
        allocation=ChannelBitAllocation.preset(effective_bits=1.5, width=8),
    )
    artifact = TrialityRotationArtifact(
        layer_idx=0,
        bits=2.5,
        view="vector",
        rotation=torch.eye(8, dtype=torch.float32),
        rotation_seed=17,
        qjl_seed=71,
        metadata=metadata,
    )
    frame = save_triality_proxy_rotations([artifact], tmp_path)
    assert frame.loc[0, "tq_triality_mode"] == "triality_proxy"
    loaded = load_triality_proxy_rotations(tmp_path)
    restored = loaded[(0, 2.5, "vector")]
    assert restored.metadata["tq_stage1_effective_bits"] == 1.25
    assert restored.metadata["tq_runtime_bits_per_channel"] == 2.25
    assert restored.metadata["tq_triality_view"] == "vector"


def test_train_script_writes_meta_and_dynamic_head_dim_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = build_turboquant_artifact_metadata(
        total_bits=2.5,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=8,
        allocation=ChannelBitAllocation.preset(effective_bits=1.5, width=8),
    )
    artifacts = [
        TrialityRotationArtifact(
            layer_idx=0,
            bits=2.5,
            view="vector",
            rotation=torch.eye(8, dtype=torch.float32),
            rotation_seed=17,
            qjl_seed=71,
            metadata=metadata,
        )
    ]
    training_summary = pd.DataFrame(
        [
            {
                "layer": 0,
                "bits": 2.5,
                "bit_setting": "2.5",
                "view": "vector",
                "mode": "key_only_block_so8_triality_vector",
            }
        ]
    )

    def fake_fit_triality_proxy_rotations(**_kwargs):
        return artifacts, training_summary

    monkeypatch.setattr(TRAIN_MODULE, "fit_triality_proxy_rotations", fake_fit_triality_proxy_rotations)

    out = tmp_path / "train_out"
    argv = [
        "research_train_k_triality.py",
        "--kv-dir",
        str(tmp_path / "kv"),
        "--bits",
        "2.5,3.5",
        "--output-dir",
        str(out),
        "--write-config",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = TRAIN_MODULE.main()
    finally:
        sys.argv = old

    assert code == 0
    meta = json.loads((out / "metrics" / "triality_training_run_meta.json").read_text(encoding="utf-8"))
    assert meta["head_dim"] == 8
    assert meta["tq_triality_mode"] == "triality_proxy"
    assert meta["bit_grid"] == [2.5, 3.5]
    config = json.loads((out / "turboquant_config.research.json").read_text(encoding="utf-8"))
    assert config["k_codec"]["head_dim"] == 8
    assert config["k_codec"]["qjl_dim"] == 8


def test_eval_script_writes_meta_and_dynamic_head_dim_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_in = tmp_path / "input_trials.csv"
    pd.DataFrame([{"mode": "key_only_block_so8_triality_vector"}]).to_csv(metrics_in, index=False)
    rot_dir = tmp_path / "rotations"
    rot_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_turboquant_artifact_metadata(
        total_bits=2.5,
        qjl_bits=1,
        qjl_dim=8,
        rotation_policy="block_so8_learned",
        rotation_seed=17,
        qjl_seed=71,
        triality_mode="triality_proxy",
        triality_view="vector",
        width=8,
        allocation=ChannelBitAllocation.preset(effective_bits=1.5, width=8),
    )
    save_triality_proxy_rotations(
        [
            TrialityRotationArtifact(
                layer_idx=0,
                bits=2.5,
                view="vector",
                rotation=torch.eye(8, dtype=torch.float32),
                rotation_seed=17,
                qjl_seed=71,
                metadata=metadata,
            )
        ],
        rot_dir,
    )

    summary_frame = pd.DataFrame([{"mode": "key_only_block_so8_triality_vector", "bits": 2.5}])
    mean_pm_sd = pd.DataFrame([{"mode": "key_only_block_so8_triality_vector", "bit_setting": "2.5"}])

    def fake_summarize_from_trials(_trial_frame, skip_statistics=False):
        return summary_frame, mean_pm_sd, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(MODULE, "summarize_from_trials", fake_summarize_from_trials)

    out = tmp_path / "eval_out"
    argv = [
        "research_validate_k_triality.py",
        "--from-existing-trials",
        str(metrics_in),
        "--rotation-dir",
        str(rot_dir),
        "--output-dir",
        str(out),
        "--skip-plots",
        "--skip-statistics",
        "--write-config",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = MODULE.main()
    finally:
        sys.argv = old

    assert code == 0
    meta = json.loads((out / "metrics" / "triality_eval_run_meta.json").read_text(encoding="utf-8"))
    assert meta["head_dim"] == 8
    assert meta["tq_triality_mode"] == "triality_proxy"
    config = json.loads((out / "turboquant_config.research.json").read_text(encoding="utf-8"))
    assert config["k_codec"]["head_dim"] == 8
    assert config["k_codec"]["qjl_dim"] == 8


def test_compute_triality_statistics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty trial_frame"):
        compute_triality_statistics(pd.DataFrame())


def test_summarize_from_trials_reconstructs_summary_and_stats() -> None:
    rows = []
    # Three trials so compute_pairwise_wilcoxon_rotation_modes gets len(paired) >= 3 per mode pair.
    for trial in (0, 1, 2):
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
