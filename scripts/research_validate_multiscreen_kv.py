"""Evaluate captured KV with paper-style attention metrics and Multiscreen / Triality modes.

Modes (``--mode``):
  **Default (production canonical / 実用正系):** ``key_only_block_so8_triality_vector`` —
  Triality SO(8) proxy + TurboQuant (vector view); requires trained rotations.

  Others: ``exact`` | ``key_only_random`` | ``key_only_block_so8_static`` |
  ``multiscreen_relevance``

Triality vector mode uses ``--rotation-dir`` (default: ``artifacts/research_extension/triality_full_train/rotations``)
with artifacts from ``research_train_k_triality.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from turboquant.allocation import ChannelBitAllocation
from turboquant.analysis import load_captured_runs
from turboquant.io_utils import ensure_dir
from turboquant.research_extension.captured_kv_modes import (
    CAPTURED_KEY_EVAL_MODES,
    eval_captured_key_mode_row,
)
from turboquant.research_extension.k_triality import (
    DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR,
    PRODUCTION_K_TURBOQUANT_MODE,
    load_triality_proxy_rotations,
)

ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "multiscreen_kv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Captured KV key-only eval: default mode is production Triality SO(8) + TurboQuant "
            f"({PRODUCTION_K_TURBOQUANT_MODE})."
        ),
    )
    parser.add_argument(
        "--captured-dir",
        default="artifacts/kv_4bit",
        help="Root with capture_manifest.json or subdirs per prompt (same as paper_validate --kv-dir).",
    )
    parser.add_argument(
        "--mode",
        default=PRODUCTION_K_TURBOQUANT_MODE,
        choices=sorted(CAPTURED_KEY_EVAL_MODES),
        help=(
            f"Evaluation mode (default: {PRODUCTION_K_TURBOQUANT_MODE} = 実用正系: "
            "Triality vector + TurboQuant; needs --rotation-dir)."
        ),
    )
    parser.add_argument(
        "--bits",
        type=float,
        default=3.0,
        help="Key TurboQuant total bits (floor for Stage-1 integer width). Ignored for exact.",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=0, help="0 = all layers.")
    parser.add_argument(
        "--eval-device",
        default="cpu",
        help="Device for tensors during eval (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument(
        "--rotation-dir",
        default="",
        help=(
            "Directory with triality *.pt rotations. "
            f"If empty and mode is triality vector, uses {DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR}."
        ),
    )
    parser.add_argument("--ms-regular-bits", type=int, default=2)
    parser.add_argument("--ms-outlier-bits", type=int, default=4)
    parser.add_argument(
        "--ms-outlier-count",
        type=int,
        default=64,
        help="Global top-K over flattened (batch, heads, seq) relevance positions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode not in CAPTURED_KEY_EVAL_MODES:
        raise SystemExit(f"Invalid mode {args.mode!r}")

    captured = Path(args.captured_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")

    bundles = load_captured_runs(captured)
    eval_device = torch.device(args.eval_device)

    triality_artifacts = None
    rotation_path: Path | None = None
    if args.mode == "key_only_block_so8_triality_vector":
        rot_str = args.rotation_dir.strip() or DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR
        rotation_path = Path(rot_str)
        if not rotation_path.is_dir():
            raise SystemExit(
                f"Missing triality rotation directory: {rotation_path.resolve()}.\n"
                "Train first with the same KV capture, e.g.:\n"
                "  uv run python scripts\\research_train_k_triality.py "
                "--kv-dir <your-capture-root> --output-dir artifacts\\research_extension\\triality_full_train\n"
                f"Expected *.pt under {DEFAULT_PRODUCTION_TRIALITY_ROTATION_DIR} (or pass --rotation-dir)."
            )
        triality_artifacts = load_triality_proxy_rotations(rotation_path)

    ms_alloc: ChannelBitAllocation | None = None
    if args.mode == "multiscreen_relevance":
        ms_alloc = ChannelBitAllocation.from_multiscreen_relevance(
            regular_bits=args.ms_regular_bits,
            outlier_bits=args.ms_outlier_bits,
            outlier_count=args.ms_outlier_count,
        )

    rows: list[dict[str, float | int | str]] = []
    bit_value = float(args.bits)

    for trial in range(args.trials):
        for bundle in bundles:
            if args.max_layers > 0 and bundle.layer_idx >= args.max_layers:
                continue
            if eval_device.type == "cuda":
                torch.cuda.empty_cache()
            row = eval_captured_key_mode_row(
                mode=args.mode,
                bundle=bundle,
                trial=trial,
                bit_value=bit_value,
                eval_device=eval_device,
                rotation_dir=rotation_path,
                triality_artifacts=triality_artifacts,
                ms_alloc=ms_alloc,
            )
            meta = bundle.metadata
            row["model_name"] = meta.model_name
            row["tokenizer_name"] = meta.tokenizer_name
            row["prompt_hash"] = meta.prompt_hash
            row["capture_dtype"] = meta.dtype
            row["capture_device"] = meta.device
            row["capture_id"] = meta.capture_id or ""
            row["capture_dir"] = str(bundle.capture_dir)
            rows.append(row)

    trial_frame = pd.DataFrame(rows)
    trial_path = metrics_dir / "multiscreen_kv_trials.csv"
    trial_frame.to_csv(trial_path, index=False)

    meta_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "research_validate_multiscreen_kv.py",
        "production_canonical_k_mode": PRODUCTION_K_TURBOQUANT_MODE,
        "mode": args.mode,
        "bits": bit_value,
        "trials": args.trials,
        "max_layers": args.max_layers,
        "eval_device": str(eval_device),
        "captured_dir": str(captured.resolve()),
        "output_dir": str(output_dir.resolve()),
        "rotation_dir": str(rotation_path.resolve()) if rotation_path else None,
        "multiscreen_regular_bits": args.ms_regular_bits,
        "multiscreen_outlier_bits": args.ms_outlier_bits,
        "multiscreen_outlier_count": args.ms_outlier_count,
        "reference_paper": "https://www.alphaxiv.org/abs/2604.01178",
        "rows_csv": str(trial_path).replace("\\", "/"),
    }
    if bundles:
        meta_payload["example_model"] = bundles[0].metadata.model_name
        meta_payload["example_prompt_hash"] = bundles[0].metadata.prompt_hash
    (metrics_dir / "multiscreen_kv_run_meta.json").write_text(
        json.dumps(meta_payload, indent=2),
        encoding="utf-8",
    )

    print(trial_frame.head())
    print(f"wrote {trial_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
