"""PyTorch-only toy attention benchmark helpers for the paper baseline."""

from __future__ import annotations

from typing import Iterable

from turboquant.analysis import evaluate_layer_grid


PAPER_BASELINE_MODES = {"exact", "key_only_random", "full_kv"}


def filter_paper_baseline_rows(rows: Iterable[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return [row for row in rows if str(row.get("mode")) in PAPER_BASELINE_MODES]


def evaluate_paper_attention_grid(
    *,
    dataset: str,
    keys,
    values,
    trial: int,
    layer_idx: int,
    bit_grid: list[float],
) -> list[dict[str, float | int | str]]:
    rows = evaluate_layer_grid(
        dataset=dataset,
        keys=keys,
        values=values,
        trial=trial,
        layer_idx=layer_idx,
        bit_grid=bit_grid,
    )
    return filter_paper_baseline_rows(rows)
