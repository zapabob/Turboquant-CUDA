"""Shared statistical helpers for online/offline TurboQuant evaluation."""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = (total - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def summarize_continuous_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str = "value",
) -> pd.DataFrame:
    """Aggregate repeated continuous measurements with SEM and 95% CI.

    Shapes:
    - ``frame``: one row per repeated measurement
    - ``value_column``: scalar continuous value
    """

    if value_column not in frame.columns:
        raise ValueError(f"Expected '{value_column}' in frame columns")

    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        values = group[value_column].astype(float)
        n = int(values.count())
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem = float(stats.sem(values, nan_policy="omit")) if n > 1 else 0.0
        if n > 1 and math.isfinite(sem) and sem > 0.0:
            ci_low, ci_high = stats.t.interval(
                confidence=0.95,
                df=n - 1,
                loc=mean,
                scale=sem,
            )
        else:
            ci_low = mean
            ci_high = mean
        record.update(
            {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _wilson_interval(*, successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    z = float(stats.norm.ppf(1.0 - ((1.0 - confidence) / 2.0)))
    phat = successes / n
    denominator = 1.0 + ((z**2) / n)
    center = (phat + ((z**2) / (2.0 * n))) / denominator
    radius = (z / denominator) * math.sqrt((phat * (1.0 - phat) / n) + ((z**2) / (4.0 * (n**2))))
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_benchmark_items(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    outcome_column: str = "is_correct",
) -> pd.DataFrame:
    """Aggregate per-item benchmark correctness with Wilson 95% CI."""

    if outcome_column not in frame.columns:
        raise ValueError(f"Expected '{outcome_column}' in frame columns")

    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        values = group[outcome_column].astype(int)
        n = int(values.count())
        n_correct = int(values.sum())
        mean = float(n_correct / n) if n > 0 else float("nan")
        ci_low, ci_high = _wilson_interval(successes=n_correct, n=n) if n > 0 else (float("nan"), float("nan"))
        variance = mean * (1.0 - mean) if n > 0 else float("nan")
        std = math.sqrt(variance) if n > 0 else float("nan")
        sem = math.sqrt(variance / n) if n > 0 else float("nan")
        record.update(
            {
                "n": n,
                "n_correct": n_correct,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def compute_continuous_pairwise_statistics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    pairing_columns: list[str],
    mode_column: str = "mode",
    value_column: str = "value",
    baseline_modes: list[str] | tuple[str, ...] = ("exact",),
) -> pd.DataFrame:
    """Compute Wilcoxon signed-rank pairwise tests with Holm correction."""

    required = {value_column, mode_column, *group_columns, *pairing_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"compute_continuous_pairwise_statistics missing columns: {missing}")

    rows: list[dict[str, float | int | str | bool]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        raw_rows: list[dict[str, float | int | str | bool]] = []
        modes = sorted(str(mode) for mode in group[mode_column].dropna().unique())
        for baseline_mode in baseline_modes:
            if baseline_mode not in modes:
                continue
            for candidate_mode in modes:
                if candidate_mode == baseline_mode:
                    continue
                paired = group.loc[group[mode_column].isin((baseline_mode, candidate_mode))].copy()
                pivot = paired.pivot_table(
                    index=pairing_columns,
                    columns=mode_column,
                    values=value_column,
                    aggfunc="first",
                ).dropna()
                if len(pivot) < 3:
                    continue
                baseline_values = pivot[baseline_mode].astype(float)
                candidate_values = pivot[candidate_mode].astype(float)
                result = stats.wilcoxon(
                    baseline_values,
                    candidate_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )
                raw_rows.append(
                    {
                        **group_record,
                        "baseline_mode": baseline_mode,
                        "candidate_mode": candidate_mode,
                        "test": "wilcoxon_signed_rank",
                        "n_pairs": int(len(pivot)),
                        "statistic": float(result.statistic),
                        "p_value": float(result.pvalue),
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "delta_candidate_minus_baseline": float(candidate_values.mean() - baseline_values.mean()),
                    }
                )
        if raw_rows:
            adjusted = holm_bonferroni([float(row["p_value"]) for row in raw_rows])
            for row, adj_p in zip(raw_rows, adjusted, strict=True):
                row["p_value_holm"] = adj_p
                row["significant_0_05"] = adj_p < 0.05
            rows.extend(raw_rows)
    return pd.DataFrame(rows)


def compute_benchmark_pairwise_statistics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    pairing_columns: list[str],
    mode_column: str = "mode",
    outcome_column: str = "is_correct",
    baseline_modes: list[str] | tuple[str, ...] = ("exact",),
) -> pd.DataFrame:
    """Compute exact McNemar pairwise tests on item-level benchmark outcomes."""

    required = {outcome_column, mode_column, *group_columns, *pairing_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"compute_benchmark_pairwise_statistics missing columns: {missing}")

    rows: list[dict[str, float | int | str | bool]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        raw_rows: list[dict[str, float | int | str | bool]] = []
        modes = sorted(str(mode) for mode in group[mode_column].dropna().unique())
        for baseline_mode in baseline_modes:
            if baseline_mode not in modes:
                continue
            for candidate_mode in modes:
                if candidate_mode == baseline_mode:
                    continue
                paired = group.loc[group[mode_column].isin((baseline_mode, candidate_mode))].copy()
                pivot = paired.pivot_table(
                    index=pairing_columns,
                    columns=mode_column,
                    values=outcome_column,
                    aggfunc="first",
                ).dropna()
                if pivot.empty:
                    continue
                baseline_values = pivot[baseline_mode].astype(int)
                candidate_values = pivot[candidate_mode].astype(int)
                baseline_only = int(((baseline_values == 1) & (candidate_values == 0)).sum())
                candidate_only = int(((baseline_values == 0) & (candidate_values == 1)).sum())
                discordant = baseline_only + candidate_only
                p_value = (
                    float(stats.binomtest(min(baseline_only, candidate_only), n=discordant, p=0.5).pvalue)
                    if discordant > 0
                    else 1.0
                )
                raw_rows.append(
                    {
                        **group_record,
                        "baseline_mode": baseline_mode,
                        "candidate_mode": candidate_mode,
                        "test": "mcnemar_exact",
                        "n_pairs": int(len(pivot)),
                        "p_value": p_value,
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "delta_candidate_minus_baseline": float(candidate_values.mean() - baseline_values.mean()),
                        "discordant_baseline_only": baseline_only,
                        "discordant_candidate_only": candidate_only,
                    }
                )
        if raw_rows:
            adjusted = holm_bonferroni([float(row["p_value"]) for row in raw_rows])
            for row, adj_p in zip(raw_rows, adjusted, strict=True):
                row["p_value_holm"] = adj_p
                row["significant_0_05"] = adj_p < 0.05
            rows.extend(raw_rows)
    return pd.DataFrame(rows)
