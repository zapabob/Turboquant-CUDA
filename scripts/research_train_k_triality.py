from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import ensure_dir
from turboquant.research_extension import (
    KeyResearchConfig,
    TRIALITY_PROXY_VIEWS,
    ValueResearchConfig,
    fit_triality_proxy_rotations,
    save_triality_proxy_rotations,
)
from turboquant.schema import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    TURBOQUANT_REFERENCE_PAPER_URL,
    build_research_turboquant_config,
    write_turboquant_config,
)


ARTIFACT_ROOT = Path("artifacts") / "research_extension" / "triality_full_train"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train K-only triality proxy block-SO(8) rotations on captured KV.")
    parser.add_argument("--kv-dir", default="artifacts/kv")
    parser.add_argument("--bits", default="2,2.5,3,3.5,4,8")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config-out", default=None)
    parser.add_argument(
        "--log-rotation-fit-trace",
        action="store_true",
        help="Append per-optimizer-step orthogonality and SO(8) block det error to metrics/triality_rotation_fit_trace.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bit_grid = [float(item) for item in args.bits.split(",") if item]
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    rotation_dir = ensure_dir(output_dir / "rotations")

    rotation_fit_trace: list[dict[str, float | int | str]] | None = [] if args.log_rotation_fit_trace else None
    artifacts, training_summary = fit_triality_proxy_rotations(
        kv_root=Path(args.kv_dir),
        bit_grid=bit_grid,
        max_layers=args.max_layers,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        rotation_fit_trace=rotation_fit_trace,
    )
    rotation_manifest = save_triality_proxy_rotations(artifacts, rotation_dir)
    training_summary.to_csv(metrics_dir / "triality_training_summary.csv", index=False)
    if rotation_fit_trace is not None:
        pd.DataFrame(rotation_fit_trace).to_csv(metrics_dir / "triality_rotation_fit_trace.csv", index=False)
    rotation_manifest.to_csv(metrics_dir / "triality_rotation_manifest.csv", index=False)
    if not artifacts:
        raise ValueError("fit_triality_proxy_rotations returned no artifacts")
    head_dim = int(artifacts[0].rotation.shape[-1])
    run_meta_path = metrics_dir / "triality_training_run_meta.json"
    run_meta = {
        "timestamp_utc": _utc_now(),
        "script": "research_train_k_triality.py",
        "kv_dir": str(Path(args.kv_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "rotation_dir": str(rotation_dir.resolve()),
        "device": args.device,
        "steps": args.steps,
        "lr": args.lr,
        "max_layers": args.max_layers,
        "bit_grid": bit_grid,
        "head_dim": head_dim,
        "tq_schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "tq_triality_mode": "triality_proxy",
        "tq_triality_views": list(TRIALITY_PROXY_VIEWS),
        "reference_paper": TURBOQUANT_REFERENCE_PAPER_URL,
        "training_summary_csv": str((metrics_dir / "triality_training_summary.csv").resolve()),
        "rotation_manifest_csv": str((metrics_dir / "triality_rotation_manifest.csv").resolve()),
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    if args.write_config:
        config_out = Path(args.config_out) if args.config_out else output_dir / "turboquant_config.research.json"
        payload = build_research_turboquant_config(
            key_config=KeyResearchConfig(
                head_dim=head_dim,
                qjl_dim=head_dim,
                views=TRIALITY_PROXY_VIEWS,
            ),
            value_config=ValueResearchConfig(),
            artifact_refs={
                "training_summary_csv": str((metrics_dir / "triality_training_summary.csv")).replace("\\", "/"),
                "rotation_manifest_csv": str((metrics_dir / "triality_rotation_manifest.csv")).replace("\\", "/"),
                "rotation_dir": str(rotation_dir).replace("\\", "/"),
                "training_run_meta_json": str(run_meta_path).replace("\\", "/"),
            },
        )
        write_turboquant_config(config_out, payload)

    print(training_summary)
    print(rotation_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
