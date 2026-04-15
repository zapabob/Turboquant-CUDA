from __future__ import annotations

from turboquant.repo_contract import (
    collect_repository_contract_errors,
    load_repository_contract,
    validate_llama_cpp_checkout,
)


def test_repository_contract_is_consistent() -> None:
    contract = load_repository_contract()
    errors = collect_repository_contract_errors(contract.repo_root)
    assert errors == []


def test_vendored_llama_cpp_checkout_matches_contract() -> None:
    contract = load_repository_contract()
    vendored_llama_cpp = contract.repo_root / contract.paths.vendored_llama_cpp
    errors = validate_llama_cpp_checkout(contract, llama_cpp_dir=vendored_llama_cpp)
    assert errors == []
