"""Research evaluation wrappers for K/V-separated experiments."""

from __future__ import annotations

from pathlib import Path

from turboquant.analysis import (
    compute_value_sensitivity_rows,
    evaluate_layer_grid,
    evaluate_value_protection_grid,
    load_captured_runs,
    synthetic_kv,
)


V_ABLATION_MODES = {
    "v_mse_random",
    "v_mse_block_so8",
    "v_prod_random",
    "v_prod_block_so8",
    "protected_v",
    "protected_v_lowrank",
}


def filter_v_ablation_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return [row for row in rows if str(row.get("mode")) in V_ABLATION_MODES]


def synthetic_v_ablation_rows(
    *,
    trial: int,
    layer_idx: int,
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    bit_grid: list[float],
) -> list[dict[str, float | int | str]]:
    keys, values = synthetic_kv(seed=6_000 + (trial * 31) + layer_idx, batch=batch, heads=heads, seq_len=seq_len, dim=dim)
    rows = evaluate_layer_grid(
        dataset="research_synthetic",
        keys=keys,
        values=values,
        trial=trial,
        layer_idx=layer_idx,
        bit_grid=bit_grid,
    )
    return filter_v_ablation_rows(rows)


def captured_v_ablation_rows(
    *,
    kv_root: Path,
    trial: int,
    bit_grid: list[float],
    max_layers: int = 0,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    bundles = load_captured_runs(kv_root)
    for bundle in bundles:
        if max_layers > 0 and bundle.layer_idx >= max_layers:
            continue
        rows.extend(
            filter_v_ablation_rows(
                evaluate_layer_grid(
                    dataset=f"research_captured:{bundle.metadata.prompt_label}",
                    keys=bundle.keys,
                    values=bundle.values,
                    trial=trial,
                    layer_idx=bundle.layer_idx,
                    bit_grid=bit_grid,
                )
            )
        )
    return rows


__all__ = [
    "V_ABLATION_MODES",
    "captured_v_ablation_rows",
    "compute_value_sensitivity_rows",
    "evaluate_value_protection_grid",
    "filter_v_ablation_rows",
    "synthetic_v_ablation_rows",
]
