from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.triality_live_gguf import (  # noqa: E402
    summary_dict,
    verify_triality_live_gguf,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a Triality schema-v2 identity live GGUF and sidecar."
    )
    parser.add_argument("--source-gguf", required=True)
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = verify_triality_live_gguf(
        source_path=Path(args.source_gguf),
        model_path=Path(args.model_gguf),
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    for key, value in summary_dict(summary).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
