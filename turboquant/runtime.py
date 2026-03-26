"""Runtime guards and model defaults for the Windows + uv workflow."""

from __future__ import annotations

import sys
from typing import Any


REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 12
REQUIRED_CUDA = "12.8"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_MODEL_ID = "Qwen/Qwen3.5-9B-Base"
SUPPORTED_MODEL_IDS = {DEFAULT_MODEL_ID, BASE_MODEL_ID}


def python_is_supported() -> bool:
    return sys.version_info[:2] == (REQUIRED_PYTHON_MAJOR, REQUIRED_PYTHON_MINOR)


def require_supported_python() -> None:
    if not python_is_supported():
        raise RuntimeError(
            "This project requires Python 3.12.x. "
            f"Found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
        )


def torch_cuda_version(torch_mod: Any) -> str | None:
    return getattr(getattr(torch_mod, "version", None), "cuda", None)


def cuda_matches_target(torch_mod: Any) -> bool:
    return torch_cuda_version(torch_mod) == REQUIRED_CUDA and bool(torch_mod.cuda.is_available())


def model_preset_to_id(name: str) -> str:
    if name == "default":
        return DEFAULT_MODEL_ID
    if name == "base":
        return BASE_MODEL_ID
    raise ValueError(f"Unsupported model preset: {name}")
