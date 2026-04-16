"""Validate that the workspace stays aligned with the expected upstream repos."""

from __future__ import annotations

import argparse
from pathlib import Path

from turboquant.repo_contract import (
    load_repository_contract,
    resolve_llama_cpp_checkout,
    validate_documentation,
    validate_gitmodules,
    validate_llama_cpp_checkout,
    validate_qwen_runtime_contract,
    validate_rust_build_script,
    validate_vendor_remote,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for repository contract validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=None,
        help="Optional explicit llama.cpp checkout to validate for Rust builds.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the repository contract checks and print a human-readable result."""

    args = parse_args()
    contract = load_repository_contract()
    resolved_llama_cpp = resolve_llama_cpp_checkout(contract, llama_cpp_dir=args.llama_cpp_dir)
    errors: list[str] = []
    errors.extend(validate_gitmodules(contract))
    errors.extend(validate_vendor_remote(contract))
    errors.extend(validate_llama_cpp_checkout(contract, llama_cpp_dir=args.llama_cpp_dir))
    errors.extend(validate_qwen_runtime_contract(contract, llama_cpp_dir=args.llama_cpp_dir))
    errors.extend(validate_rust_build_script(contract))
    errors.extend(validate_documentation(contract))

    print(f"Repository contract: {contract.sources.turboquant_name} + {contract.sources.llama_cpp_name}")
    print(f"Vendored llama.cpp path: {contract.paths.vendored_llama_cpp.as_posix()}")
    print(f"Resolved Rust llama.cpp checkout: {resolved_llama_cpp.as_posix()}")
    if errors:
        print("Contract violations:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository contract OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
