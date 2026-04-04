from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.analysis import summarize_trial_metrics
from turboquant.io_utils import ensure_dir
from turboquant.research_extension import (
    KeyResearchConfig,
    TRIALITY_PROXY_VIEWS,
    ValueResearchConfig,
    build_best_per_layer_selector,
    compute_triality_selector_statistics,
)
from turboquant.schema import build_research_turboquant_config, write_turboquant_config


ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "triality_best_per_layer"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(frame.iloc[row_idx][column]) for column in columns) + " |"
        for row_idx in range(len(frame))
    ]
    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and gate Triality best-per-layer policy from existing trial CSV.")
    parser.add_argument(
        "--trial-csv",
        default="artifacts/research_extension/triality_full_eval/metrics/triality_trials_captured.csv",
    )
    parser.add_argument("--rotation-dir", default="artifacts/research_extension/triality_full_train/rotations")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    trial_frame = pd.read_csv(args.trial_csv)
    manifest, selected_frame = build_best_per_layer_selector(trial_frame)
    selector_stats = compute_triality_selector_statistics(selected_frame, trial_frame)
    selected_summary = summarize_trial_metrics(selected_frame)

    manifest_path = output_dir / "triality_best_per_layer_manifest.csv"
    summary_path = output_dir / "triality_best_per_layer_summary.csv"
    stats_path = output_dir / "triality_best_per_layer_gate.csv"
    manifest.to_csv(manifest_path, index=False)
    selected_summary.to_csv(summary_path, index=False)
    selector_stats.to_csv(stats_path, index=False)
    (output_dir / "triality_best_per_layer_manifest.md").write_text(markdown_table(manifest), encoding="utf-8")
    (output_dir / "triality_best_per_layer_summary.md").write_text(markdown_table(selected_summary), encoding="utf-8")
    (output_dir / "triality_best_per_layer_gate.md").write_text(markdown_table(selector_stats), encoding="utf-8")

    if args.write_config:
        config_out = Path(args.config_out) if args.config_out else output_dir / "triality_best_per_layer_config.json"
        payload = build_research_turboquant_config(
            key_config=KeyResearchConfig(
                head_dim=128,
                view_selection="best_per_layer",
                views=TRIALITY_PROXY_VIEWS,
            ),
            value_config=ValueResearchConfig(),
            artifact_refs={
                "trial_csv": str(Path(args.trial_csv)).replace("\\", "/"),
                "selector_manifest_csv": str(manifest_path).replace("\\", "/"),
                "selector_summary_csv": str(summary_path).replace("\\", "/"),
                "selector_gate_csv": str(stats_path).replace("\\", "/"),
                "rotation_dir": str(Path(args.rotation_dir)).replace("\\", "/"),
            },
        )
        write_turboquant_config(config_out, payload)

    print(selector_stats)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
