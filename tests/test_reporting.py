from __future__ import annotations

import pandas as pd

from turboquant.reporting import summarize_metric_trials


def test_summarize_metric_trials_produces_confidence_columns() -> None:
    frame = pd.DataFrame(
        [
            {"experiment": "mse", "bits": 2.0, "metric": "mse", "trial": 0, "value": 1.0},
            {"experiment": "mse", "bits": 2.0, "metric": "mse", "trial": 1, "value": 3.0},
        ]
    )
    summary = summarize_metric_trials(frame, group_columns=["experiment", "bits", "metric"])
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["mean"] == 2.0
    assert row["std"] > 0
    assert row["ci95_high"] >= row["mean"] >= row["ci95_low"]
