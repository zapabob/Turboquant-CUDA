"""Statistical summaries for TurboQuant experiment metrics."""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats


def summarize_metric_trials(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate trial rows into summary statistics with 95% confidence intervals."""

    if "value" not in frame.columns:
        raise ValueError("Expected a 'value' column in the input frame")

    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        values = group["value"].astype(float)
        n = int(values.count())
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem = float(stats.sem(values, nan_policy="omit")) if n > 1 else 0.0
        if n > 1 and math.isfinite(sem) and sem > 0:
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
