from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.gguf_profiles import build_hypura_serve_command, read_hypura_gguf_bridge_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for launching a Kobold-compatible Hypura proxy."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch Hypura as a Kobold-compatible proxy for a TurboQuant-aware GGUF. "
            "This is the compatibility path for clients such as unsupported KoboldCpp builds."
        )
    )
    parser.add_argument("--gguf", required=True, help="Path to the packaged GGUF file.")
    parser.add_argument("--host", default="127.0.0.1", help="Host passed to `hypura serve`.")
    parser.add_argument("--port", type=int, default=5001, help="Kobold-compatible API port.")
    parser.add_argument("--context", type=int, default=8192, help="Context length passed to Hypura.")
    parser.add_argument(
        "--turboquant-mode",
        default="gguf-auto",
        choices=["gguf-auto", "exact", "paper-key-only", "paper-full-kv", "research-kv-split"],
        help=(
            "TurboQuant runtime mode for Hypura. 'gguf-auto' reads the embedded "
            "hypura.turboquant.* metadata from the GGUF and fails if it is missing."
        ),
    )
    parser.add_argument("--release", action="store_true", help="Use `cargo run --release`.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> int:
    """Launch Hypura serve with a Kobold-compatible API surface."""

    args = parse_args()
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file does not exist: {gguf_path}")

    if args.turboquant_mode == "gguf-auto":
        bridge = read_hypura_gguf_bridge_config(gguf_path)
        if bridge is None:
            raise ValueError(
                "GGUF does not contain embedded Hypura bridge metadata. "
                "Repackage with `--hypura-compatible-profile auto` or pass an explicit --turboquant-mode."
            )
        print(
            "hypura_bridge="
            f"profile:{bridge.source_profile},mode:{bridge.mode},rotation_policy:{bridge.rotation_policy},"
            f"triality_view:{bridge.triality_view},rotation_seed:{bridge.rotation_seed}"
        )

    command = build_hypura_serve_command(
        gguf_path=gguf_path,
        host=args.host,
        port=args.port,
        context=args.context,
        turboquant_mode=args.turboquant_mode,
        release=args.release,
    )
    print("launch_command=" + subprocess.list2cmdline(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT / "rust", check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
