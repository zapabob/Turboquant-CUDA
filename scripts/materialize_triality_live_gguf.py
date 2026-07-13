from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.triality_live_gguf import (  # noqa: E402
    TRIALITY_LIVE_DEFAULT_TIMESTAMP,
    materialize_triality_live_gguf,
    summary_dict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a llama Q4_0 GGUF with a complete development-only "
            "Triality schema-v2 identity bundle."
        )
    )
    parser.add_argument("--input-gguf", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument("--profile-id", default="liveq4")
    parser.add_argument("--generated-at-utc", default=TRIALITY_LIVE_DEFAULT_TIMESTAMP)
    parser.add_argument("--development-identity-views", action="store_true")
    parser.add_argument("--disable-weight-conversion", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = materialize_triality_live_gguf(
        source_path=Path(args.input_gguf),
        output_path=Path(args.output_gguf),
        profile_id=args.profile_id,
        generated_at_utc=args.generated_at_utc,
        development_identity_views=args.development_identity_views,
        disable_weight_conversion=args.disable_weight_conversion,
    )
    for key, value in summary_dict(summary).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
