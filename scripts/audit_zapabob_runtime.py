from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import ensure_dir
from turboquant.runtime_eval import audit_zapabob_runtime_checkout, render_runtime_audit_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current-main zapabob/llama.cpp TurboQuant runtime wiring.")
    parser.add_argument("--llama-cpp-dir", required=True, help="Path to the authoritative zapabob/llama.cpp checkout.")
    parser.add_argument("--output-dir", default="artifacts/runtime_eval/current_main", help="Artifact output directory.")
    parser.add_argument("--perplexity-bin", default=None, help="Optional built llama-perplexity binary path.")
    parser.add_argument("--llama-bench-bin", default=None, help="Optional built llama-bench binary path.")
    parser.add_argument("--server-bin", default=None, help="Optional built llama-server binary path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    binary_paths = {
        name: Path(value)
        for name, value in {
            "llama-perplexity": args.perplexity_bin,
            "llama-bench": args.llama_bench_bin,
            "llama-server": args.server_bin,
        }.items()
        if value
    }
    audit_payload = audit_zapabob_runtime_checkout(
        Path(args.llama_cpp_dir),
        binary_paths=binary_paths,
    )
    (output_dir / "runtime_code_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    (output_dir / "runtime_code_audit.md").write_text(render_runtime_audit_markdown(audit_payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
