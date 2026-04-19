"""Repository contract helpers for downstream build consistency checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tomllib


@dataclass(frozen=True)
class RepositorySourceContract:
    """Source-of-truth repositories that this workspace must stay aligned with."""

    turboquant_name: str
    turboquant_url: str
    llama_cpp_name: str
    llama_cpp_url: str


@dataclass(frozen=True)
class RepositoryPathContract:
    """Important workspace paths that participate in the build contract."""

    vendored_llama_cpp: Path
    rust_build_script: Path
    readme: Path
    claude: Path


@dataclass(frozen=True)
class RustBuildContract:
    """Rust-side path and environment contract for locating llama.cpp."""

    llama_cpp_env_vars: tuple[str, ...]
    llama_cpp_candidate_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class LlamaCppCompatibilityContract:
    """Markers that distinguish the expected zapabob llama.cpp checkout."""

    required_header: Path
    required_header_markers: tuple[str, ...]
    required_source: Path
    required_source_markers: tuple[str, ...]


@dataclass(frozen=True)
class QwenRuntimeContract:
    """Markers that define the 12GB-only Qwen TurboQuant runtime lane."""

    kv_cache_source: Path
    kv_cache_markers: tuple[str, ...]
    qwen_model_source: Path
    qwen_model_markers: tuple[str, ...]


@dataclass(frozen=True)
class DocumentationContract:
    """Markers that must stay present in user-facing documentation."""

    required_readme_markers: tuple[str, ...]
    required_claude_markers: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryContract:
    """Machine-readable repository contract used by scripts and tests."""

    sources: RepositorySourceContract
    paths: RepositoryPathContract
    rust: RustBuildContract
    llama_cpp_compat: LlamaCppCompatibilityContract
    qwen_runtime: QwenRuntimeContract
    docs: DocumentationContract
    repo_root: Path


def _repo_root_from(start: Path | None = None) -> Path:
    """Resolve the repository root by walking upward to ``repo_contract.toml``."""

    anchor = (start if start is not None else Path(__file__)).resolve()
    search_root = anchor if anchor.is_dir() else anchor.parent

    for candidate in (search_root, *search_root.parents):
        if (candidate / "repo_contract.toml").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate repo_contract.toml while resolving the repository root from '{anchor}'."
    )


def load_repository_contract(repo_root: Path | None = None) -> RepositoryContract:
    """Load the repository contract from ``repo_contract.toml``."""

    root = _repo_root_from(repo_root)
    contract_path = root / "repo_contract.toml"
    with contract_path.open("rb") as handle:
        raw = tomllib.load(handle)

    sources = RepositorySourceContract(
        turboquant_name=raw["repositories"]["turboquant_name"],
        turboquant_url=raw["repositories"]["turboquant_url"],
        llama_cpp_name=raw["repositories"]["llama_cpp_name"],
        llama_cpp_url=raw["repositories"]["llama_cpp_url"],
    )
    paths = RepositoryPathContract(
        vendored_llama_cpp=Path(raw["paths"]["vendored_llama_cpp"]),
        rust_build_script=Path(raw["paths"]["rust_build_script"]),
        readme=Path(raw["paths"]["readme"]),
        claude=Path(raw["paths"]["claude"]),
    )
    rust = RustBuildContract(
        llama_cpp_env_vars=tuple(raw["rust"]["llama_cpp_env_vars"]),
        llama_cpp_candidate_suffixes=tuple(raw["rust"]["llama_cpp_candidate_suffixes"]),
    )
    llama_cpp_compat = LlamaCppCompatibilityContract(
        required_header=Path(raw["llama_cpp_compat"]["required_header"]),
        required_header_markers=tuple(raw["llama_cpp_compat"]["required_header_markers"]),
        required_source=Path(raw["llama_cpp_compat"]["required_source"]),
        required_source_markers=tuple(raw["llama_cpp_compat"]["required_source_markers"]),
    )
    qwen_runtime = QwenRuntimeContract(
        kv_cache_source=Path(raw["qwen_runtime"]["kv_cache_source"]),
        kv_cache_markers=tuple(raw["qwen_runtime"]["kv_cache_markers"]),
        qwen_model_source=Path(raw["qwen_runtime"]["qwen_model_source"]),
        qwen_model_markers=tuple(raw["qwen_runtime"]["qwen_model_markers"]),
    )
    docs = DocumentationContract(
        required_readme_markers=tuple(raw["docs"]["required_readme_markers"]),
        required_claude_markers=tuple(raw["docs"]["required_claude_markers"]),
    )
    return RepositoryContract(
        sources=sources,
        paths=paths,
        rust=rust,
        llama_cpp_compat=llama_cpp_compat,
        qwen_runtime=qwen_runtime,
        docs=docs,
        repo_root=root,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate_gitmodules(contract: RepositoryContract) -> list[str]:
    """Validate the vendored llama.cpp submodule declaration."""

    gitmodules_path = contract.repo_root / ".gitmodules"
    gitmodules = _read_text(gitmodules_path)
    errors: list[str] = []
    if str(contract.paths.vendored_llama_cpp).replace("\\", "/") not in gitmodules.replace("\\", "/"):
        errors.append(
            f".gitmodules does not declare vendored llama.cpp path '{contract.paths.vendored_llama_cpp.as_posix()}'."
        )
    if contract.sources.llama_cpp_url not in gitmodules:
        errors.append(
            f".gitmodules does not point {contract.paths.vendored_llama_cpp.as_posix()} at '{contract.sources.llama_cpp_url}'."
        )
    return errors


def validate_vendor_remote(contract: RepositoryContract) -> list[str]:
    """Validate the checked-out vendored llama.cpp origin when the submodule exists."""

    vendored_path = contract.repo_root / contract.paths.vendored_llama_cpp
    if not vendored_path.exists():
        return [f"Vendored llama.cpp path is missing: '{vendored_path}'."]

    dot_git_path = vendored_path / ".git"
    if not dot_git_path.exists():
        return []

    try:
        origin = subprocess.run(
            ["git", "-C", str(vendored_path), "remote", "get-url", "origin"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "unknown git error"
        return [f"Failed to read {contract.paths.vendored_llama_cpp.as_posix()} origin remote: {message}"]

    if origin != contract.sources.llama_cpp_url:
        return [
            f"{contract.paths.vendored_llama_cpp.as_posix()} origin remote is "
            f"'{origin}', expected '{contract.sources.llama_cpp_url}'."
        ]
    return []


def resolve_llama_cpp_checkout(
    contract: RepositoryContract,
    llama_cpp_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the llama.cpp checkout that Rust builds will consume."""

    if llama_cpp_dir is not None:
        return llama_cpp_dir.resolve()

    environment = os.environ if env is None else env
    for env_var in contract.rust.llama_cpp_env_vars:
        candidate = environment.get(env_var)
        if candidate and candidate.strip():
            return Path(candidate.strip()).expanduser().resolve()

    return (contract.repo_root / contract.paths.vendored_llama_cpp).resolve()


def validate_llama_cpp_checkout(
    contract: RepositoryContract,
    llama_cpp_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Validate that the Rust build path points at a zapabob-compatible llama.cpp checkout."""

    checkout_path = resolve_llama_cpp_checkout(contract, llama_cpp_dir=llama_cpp_dir, env=env)
    errors: list[str] = []
    if not checkout_path.exists():
        return [f"Resolved llama.cpp checkout path is missing: '{checkout_path}'."]

    required_files = (
        contract.llama_cpp_compat.required_header,
        contract.llama_cpp_compat.required_source,
    )
    for relative_path in required_files:
        candidate = checkout_path / relative_path
        if not candidate.is_file():
            errors.append(
                f"Resolved llama.cpp checkout is missing required file '{relative_path.as_posix()}': '{checkout_path}'."
            )

    header_path = checkout_path / contract.llama_cpp_compat.required_header
    if header_path.is_file():
        header_text = _read_text(header_path)
        for marker in contract.llama_cpp_compat.required_header_markers:
            if marker not in header_text:
                errors.append(
                    f"{header_path.as_posix()} is missing required TurboQuant marker '{marker}'."
                )

    source_path = checkout_path / contract.llama_cpp_compat.required_source
    if source_path.is_file():
        source_text = _read_text(source_path)
        for marker in contract.llama_cpp_compat.required_source_markers:
            if marker not in source_text:
                errors.append(
                    f"{source_path.as_posix()} is missing required TurboQuant marker '{marker}'."
                )

    return errors


def validate_qwen_runtime_contract(
    contract: RepositoryContract,
    llama_cpp_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Validate that the Qwen runtime lane markers are present in vendored llama.cpp."""

    checkout_path = resolve_llama_cpp_checkout(contract, llama_cpp_dir=llama_cpp_dir, env=env)
    errors: list[str] = []
    kv_cache_path = checkout_path / contract.qwen_runtime.kv_cache_source
    qwen_model_path = checkout_path / contract.qwen_runtime.qwen_model_source
    if not kv_cache_path.is_file():
        return [f"Resolved llama.cpp checkout is missing KV cache runtime file '{contract.qwen_runtime.kv_cache_source.as_posix()}': '{checkout_path}'."]
    if not qwen_model_path.is_file():
        return [f"Resolved llama.cpp checkout is missing Qwen runtime file '{contract.qwen_runtime.qwen_model_source.as_posix()}': '{checkout_path}'."]

    kv_cache_text = _read_text(kv_cache_path)
    qwen_model_text = _read_text(qwen_model_path)
    for marker in contract.qwen_runtime.kv_cache_markers:
        if marker not in kv_cache_text:
            errors.append(f"{kv_cache_path.as_posix()} is missing required Qwen runtime marker '{marker}'.")
    for marker in contract.qwen_runtime.qwen_model_markers:
        if marker not in qwen_model_text:
            errors.append(f"{qwen_model_path.as_posix()} is missing required Qwen runtime marker '{marker}'.")
    return errors


def validate_rust_build_script(contract: RepositoryContract) -> list[str]:
    """Validate that Rust build glue still advertises the expected llama.cpp contract."""

    build_script = _read_text(contract.repo_root / contract.paths.rust_build_script)
    errors: list[str] = []
    if contract.sources.llama_cpp_name not in build_script:
        errors.append(
            f"{contract.paths.rust_build_script.as_posix()} does not mention '{contract.sources.llama_cpp_name}'."
        )
    for env_var in contract.rust.llama_cpp_env_vars:
        if env_var not in build_script:
            errors.append(
                f"{contract.paths.rust_build_script.as_posix()} is missing env override '{env_var}'."
            )
    for suffix in contract.rust.llama_cpp_candidate_suffixes:
        if suffix not in build_script:
            errors.append(
                f"{contract.paths.rust_build_script.as_posix()} is missing vendored candidate '{suffix}'."
            )
    return errors


def validate_documentation(contract: RepositoryContract) -> list[str]:
    """Validate that README and CLAUDE keep the build contract visible."""

    errors: list[str] = []
    readme = _read_text(contract.repo_root / contract.paths.readme)
    claude = _read_text(contract.repo_root / contract.paths.claude)

    for marker in contract.docs.required_readme_markers:
        if marker not in readme:
            errors.append(f"README.md is missing required marker '{marker}'.")
    for marker in contract.docs.required_claude_markers:
        if marker not in claude:
            errors.append(f"CLAUDE.md is missing required marker '{marker}'.")
    return errors


def collect_repository_contract_errors(repo_root: Path | None = None) -> list[str]:
    """Collect all repository contract violations."""

    contract = load_repository_contract(repo_root)
    errors: list[str] = []
    errors.extend(validate_gitmodules(contract))
    errors.extend(validate_vendor_remote(contract))
    errors.extend(validate_llama_cpp_checkout(contract))
    errors.extend(validate_qwen_runtime_contract(contract))
    errors.extend(validate_rust_build_script(contract))
    errors.extend(validate_documentation(contract))
    return errors


def assert_repository_contract(repo_root: Path | None = None) -> None:
    """Raise ``RuntimeError`` when the repository contract is violated."""

    errors = collect_repository_contract_errors(repo_root)
    if errors:
        raise RuntimeError("Repository contract validation failed:\n- " + "\n- ".join(errors))
