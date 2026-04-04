#!/usr/bin/env python3
"""Captured KV 上で triality ローテーション学習 → フル評価を連続実行する。

前提: `scripts/capture_qwen_kv.py` 等で `capture_manifest.json` 付きの KV が `--kv-dir` にあること。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train triality proxy rotations on captured KV, then run research_validate_k_triality."
    )
    parser.add_argument(
        "--kv-dir",
        type=Path,
        required=True,
        help="Root with prompt-scoped captures (or single capture dir with capture_manifest.json).",
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        default=Path("artifacts") / "research_extension" / "triality_full_train",
        help="Where fit_triality writes metrics/ and rotations/ (default: artifacts/research_extension/triality_full_train; matches PRODUCTION rotation path).",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("artifacts") / "research_extension" / "triality_full_eval",
        help="research_validate_k_triality --output-dir (default: artifacts/research_extension/triality_full_eval).",
    )
    parser.add_argument("--bits", default="2,2.5,3,3.5,4,8")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--train-lr", type=float, default=5e-2)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--trials", type=int, default=3)
    return parser.parse_args(argv)


def main() -> int:
    raw = sys.argv[1:]
    if "--" in raw:
        split_at = raw.index("--")
        main_argv = raw[:split_at]
        eval_tail = raw[split_at + 1 :]
    else:
        main_argv = raw
        eval_tail = []
    args = parse_args(main_argv)
    repo = _repo_root()
    kv = args.kv_dir.resolve()
    train_out = args.train_output_dir
    rotation_dir = train_out / "rotations"
    eval_out = args.eval_output_dir

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    train_script = repo / "scripts" / "research_train_k_triality.py"
    eval_script = repo / "scripts" / "research_validate_k_triality.py"

    train_cmd = [
        sys.executable,
        str(train_script),
        "--kv-dir",
        str(kv),
        "--bits",
        args.bits,
        "--max-layers",
        str(args.max_layers),
        "--steps",
        str(args.train_steps),
        "--lr",
        str(args.train_lr),
        "--device",
        args.train_device,
        "--output-dir",
        str(train_out),
    ]
    print("[triality-pipeline] train:", " ".join(train_cmd), flush=True)
    r1 = subprocess.run(train_cmd, cwd=repo)
    if r1.returncode != 0:
        return r1.returncode

    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--kv-dir",
        str(kv),
        "--rotation-dir",
        str(rotation_dir),
        "--bits",
        args.bits,
        "--max-layers",
        str(args.max_layers),
        "--trials",
        str(args.trials),
        "--eval-device",
        args.eval_device,
        "--output-dir",
        str(eval_out),
    ]
    if eval_tail:
        eval_cmd.extend(eval_tail)
    print("[triality-pipeline] eval:", " ".join(eval_cmd), flush=True)
    r2 = subprocess.run(eval_cmd, cwd=repo)
    return r2.returncode


if __name__ == "__main__":
    raise SystemExit(main())
