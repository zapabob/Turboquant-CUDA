from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from turboquant.analysis import load_captured_runs, summarize_trial_metrics
from turboquant.io_utils import ensure_dir
from turboquant.research_extension import (
    KeyResearchConfig,
    V_ABLATION_MODES,
    ValueResearchConfig,
    captured_v_ablation_rows,
    synthetic_v_ablation_rows,
)
from turboquant.schema import build_research_turboquant_config, write_turboquant_config


ARTIFACT_ROOT = Path("artifacts") / "research_extension"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K/V-separated value codec ablations.")
    parser.add_argument("--query-source", choices=["synthetic", "captured"], default="synthetic")
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--synthetic-layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--bits", default="2,2.5,3,3.5,4,8")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT / "metrics"))
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    frames: list[pd.DataFrame] = []

    if args.query_source == "synthetic":
        for trial in range(args.trials):
            for layer_idx in range(args.synthetic_layers):
                frames.append(
                    pd.DataFrame(
                        synthetic_v_ablation_rows(
                            trial=trial,
                            layer_idx=layer_idx,
                            batch=args.batch,
                            heads=args.heads,
                            seq_len=args.seq_len,
                            dim=args.head_dim,
                            bit_grid=bit_grid,
                        )
                    )
                )
    else:
        bundles = load_captured_runs(Path(args.kv_dir))
        if args.max_layers > 0:
            bundles = [bundle for bundle in bundles if bundle.layer_idx < args.max_layers]
        for trial in range(args.trials):
            frames.append(pd.DataFrame(captured_v_ablation_rows(kv_root=Path(args.kv_dir), trial=trial, bit_grid=bit_grid, max_layers=args.max_layers)))

    trial_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["mode"])
    if not trial_frame.empty:
        trial_frame = trial_frame.loc[trial_frame["mode"].isin(V_ABLATION_MODES)].copy()
    summary_frame = summarize_trial_metrics(trial_frame) if not trial_frame.empty else pd.DataFrame()

    suffix = args.query_source
    trial_frame.to_csv(output_dir / f"v_codec_trials_{suffix}.csv", index=False)
    summary_frame.to_csv(output_dir / f"v_codec_summary_{suffix}.csv", index=False)
    if args.write_config:
        config_out = Path(args.config_out) if args.config_out else output_dir / "turboquant_config.research.json"
        payload = build_research_turboquant_config(
            key_config=KeyResearchConfig(head_dim=args.head_dim),
            value_config=ValueResearchConfig(),
            artifact_refs={
                "trial_csv": str((output_dir / f"v_codec_trials_{suffix}.csv")).replace("\\", "/"),
                "summary_csv": str((output_dir / f"v_codec_summary_{suffix}.csv")).replace("\\", "/"),
            },
        )
        write_turboquant_config(config_out, payload)
    print(summary_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
