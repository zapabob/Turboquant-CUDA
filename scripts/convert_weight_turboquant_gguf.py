from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.weight_gguf import convert_weight_turboquant_gguf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GGUF Q8_0 weight tensors into offline TQ4_1S tensor blocks."
    )
    parser.add_argument("--input-gguf", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument(
        "--model-family",
        default=None,
        help="Optional explicit model family override, for example google/gemma-4-e4b-it.",
    )
    parser.add_argument(
        "--mode",
        default="triality-proxy-so8-pareto",
        help="Triality metadata mode to record alongside the converted weight artifact.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite --output-gguf if it already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_gguf = Path(args.input_gguf)
    output_gguf = Path(args.output_gguf)
    if output_gguf.exists():
        if not args.force:
            raise FileExistsError(f"Output GGUF already exists: {output_gguf}")
        output_gguf.unlink()

    summary = convert_weight_turboquant_gguf(
        source_path=input_gguf,
        output_path=output_gguf,
        model_family=args.model_family,
        mode=args.mode,
    )
    print(f"converted_gguf={summary.output_path}")
    print(f"model_family={summary.model_family}")
    print(f"converted_tensor_count={summary.converted_tensor_count}")
    print(f"preserved_tensor_count={summary.preserved_tensor_count}")
    print(f"weight_plan={summary.weight_plan['tensor_plan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
