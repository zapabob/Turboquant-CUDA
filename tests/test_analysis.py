from __future__ import annotations

import pandas as pd
import torch

from turboquant.analysis import evaluate_layer_grid, melt_metric_rows, summarize_layer_thresholds


def test_evaluate_layer_grid_emits_research_modes() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    keys = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    values = torch.randn((1, 2, 8, 16), generator=generator, dtype=torch.float32)
    rows = evaluate_layer_grid(
        dataset="synthetic",
        keys=keys,
        values=values,
        trial=0,
        layer_idx=0,
        bit_grid=[2.0, 3.5],
    )
    modes = {row["mode"] for row in rows}
    assert {
        "exact",
        "key_only_random",
        "key_only_block_so8_static",
        "key_only_block_so8_learned",
        "protected_v",
        "protected_v_lowrank",
        "full_kv",
    } <= modes
    sample = rows[0]
    assert "peak_vram_mb" in sample
    assert "prefill_seconds" in sample
    assert "decode_seconds" in sample


def test_melt_metric_rows_exposes_long_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "trial": 0,
                "layer": 0,
                "mode": "key_only_block_so8_learned",
                "bit_setting": "4",
                "bits": 4.0,
                "logit_cosine_similarity": 0.9,
                "logit_mae": 0.1,
                "logit_mse": 0.02,
                "logit_spearman": 0.91,
                "logit_top1_match": 1.0,
                "logit_top5_match": 1.0,
                "logit_top5_overlap": 1.0,
                "hidden_cosine_similarity": 0.85,
                "hidden_mae": 0.2,
                "hidden_mse": 0.03,
                "memory_bits": 100.0,
                "memory_ratio_vs_exact": 0.4,
                "prefill_seconds": 0.01,
                "decode_seconds": 0.005,
                "peak_vram_mb": 0.0,
            }
        ]
    )
    long_frame = melt_metric_rows(frame)
    assert {"metric", "value"} <= set(long_frame.columns)
    assert "hidden_cosine_similarity" in set(long_frame["metric"])


def test_summarize_layer_thresholds_finds_first_crossing() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "synthetic", "mode": "key_only_block_so8_learned", "bit_setting": "4", "bits": 4.0, "trial": 0, "layer": 0, "hidden_cosine_similarity": 0.999},
            {"dataset": "synthetic", "mode": "key_only_block_so8_learned", "bit_setting": "4", "bits": 4.0, "trial": 0, "layer": 1, "hidden_cosine_similarity": 0.97},
            {"dataset": "synthetic", "mode": "key_only_block_so8_learned", "bit_setting": "4", "bits": 4.0, "trial": 0, "layer": 2, "hidden_cosine_similarity": 0.94},
        ]
    )
    summary = summarize_layer_thresholds(frame, metric="hidden_cosine_similarity", threshold=0.95)
    row = summary.iloc[0]
    assert row["mean"] == 2.0
