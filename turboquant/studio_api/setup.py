"""Setup snapshot helpers for TurboQuant Studio."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import subprocess

from turboquant.capture import DEFAULT_PROMPT_PANEL
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
from turboquant.runtime import MODEL_PRESETS, REQUIRED_CUDA, torch_cuda_version
from turboquant.studio_api.config import StudioSettings
from turboquant.studio_api.models import ModelPresetInfo, PromptPanelEntry, SetupCheck, SetupSnapshot


def _command_version(*command: str) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return completed.stdout.strip() or completed.stderr.strip() or None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _repo_contract_snapshot() -> tuple[bool, list[str]]:
    contract = load_repository_contract()
    errors: list[str] = []
    errors.extend(validate_gitmodules(contract))
    errors.extend(validate_vendor_remote(contract))
    errors.extend(validate_llama_cpp_checkout(contract))
    errors.extend(validate_qwen_runtime_contract(contract))
    errors.extend(validate_rust_build_script(contract))
    errors.extend(validate_documentation(contract))
    _ = resolve_llama_cpp_checkout(contract)
    return (not errors, errors)


def build_setup_snapshot(settings: StudioSettings) -> SetupSnapshot:
    """Collect a current setup snapshot for the Studio shell."""

    torch_version = None
    torch_cuda = None
    cuda_available = False
    gpu_names: list[str] = []
    checks: list[SetupCheck] = []
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch_cuda_version(torch)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception as exc:  # pragma: no cover - environment-specific
        checks.append(SetupCheck(name="torch_import", ok=False, detail=repr(exc)))

    repo_contract_ok, repo_contract_errors = _repo_contract_snapshot()
    vendored_runtime_ready = not any("llama.cpp" in error.lower() for error in repo_contract_errors)
    checks.extend(
        [
            SetupCheck(name="python_3_12", ok=platform.python_version().startswith("3.12."), detail=platform.python_version()),
            SetupCheck(name="uv", ok=_command_version("uv", "--version") is not None, detail=_command_version("uv", "--version") or "missing"),
            SetupCheck(
                name="cuda_target",
                ok=(torch_cuda == REQUIRED_CUDA and cuda_available) if torch_cuda else False,
                detail=f"target={REQUIRED_CUDA}, current={torch_cuda}, available={cuda_available}",
            ),
            SetupCheck(
                name="repo_contract",
                ok=repo_contract_ok,
                detail="Repository contract OK." if repo_contract_ok else "; ".join(repo_contract_errors),
            ),
            SetupCheck(
                name="vendored_runtime",
                ok=vendored_runtime_ready,
                detail="zapabob llama.cpp contract surface present" if vendored_runtime_ready else "vendored runtime contract issue",
            ),
        ]
    )
    return SetupSnapshot(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        repo_root=str(settings.repo_root),
        artifact_root=str(settings.artifact_root),
        active_artifact_root=str(settings.artifact_root),
        python_version=platform.python_version(),
        uv_version=_command_version("uv", "--version"),
        node_version=_command_version("node", "--version"),
        npm_version=_command_version("npm", "--version"),
        target_cuda=REQUIRED_CUDA,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        gpu_names=gpu_names,
        repo_contract_ok=repo_contract_ok,
        repo_contract_errors=repo_contract_errors,
        vendored_runtime_ready=vendored_runtime_ready,
        checks=checks,
        model_presets=[
            ModelPresetInfo(
                name=preset.name,
                model_id=preset.model_id,
                lane_name=preset.lane_name,
                model_source=preset.resolved_model_source(),
                default_weight_load=preset.default_weight_load,
                default_dtype=preset.default_dtype,
            )
            for preset in MODEL_PRESETS.values()
        ],
        prompt_panel=[PromptPanelEntry(label=item.label, prompt=item.prompt) for item in DEFAULT_PROMPT_PANEL],
    )
