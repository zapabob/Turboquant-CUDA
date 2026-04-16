"""Runtime guards and model defaults for the Windows + uv workflow."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any


REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 12
REQUIRED_CUDA = "12.8"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_MODEL_ID = "Qwen/Qwen3.5-9B-Base"
QWEN35_4B_MODEL_ID = "Qwen/Qwen3.5-4B"

LOCAL_CAPTURE_MODEL_PATH = r"H:\Qwen3.5-9B-official-hf"

RTX3060_12GB_LANE = "rtx3060_desktop_12gb"
RTX3060_6GB_LANE = "rtx3060_notebook_6gb"

DEFAULT_CAPTURE_MODEL_PRESET = "qwen35_9b_12gb"
QWEN35_9B_BASE_PRESET = "qwen35_9b_base_12gb"
QWEN35_4B_6GB_PRESET = "qwen35_4b_6gb"


@dataclass(frozen=True, slots=True)
class CaptureModelPreset:
    """Named capture preset for a reproducible model / VRAM lane."""

    name: str
    model_id: str
    lane_name: str
    local_model_path: str | None = None
    default_weight_load: str = "4bit"
    default_dtype: str = "float16"

    def resolved_model_source(self) -> str:
        """Return the actual model reference used for loading."""

        return self.local_model_path or self.model_id


MODEL_PRESETS: dict[str, CaptureModelPreset] = {
    DEFAULT_CAPTURE_MODEL_PRESET: CaptureModelPreset(
        name=DEFAULT_CAPTURE_MODEL_PRESET,
        model_id=DEFAULT_MODEL_ID,
        lane_name=RTX3060_12GB_LANE,
        local_model_path=LOCAL_CAPTURE_MODEL_PATH,
    ),
    QWEN35_9B_BASE_PRESET: CaptureModelPreset(
        name=QWEN35_9B_BASE_PRESET,
        model_id=BASE_MODEL_ID,
        lane_name=RTX3060_12GB_LANE,
    ),
    QWEN35_4B_6GB_PRESET: CaptureModelPreset(
        name=QWEN35_4B_6GB_PRESET,
        model_id=QWEN35_4B_MODEL_ID,
        lane_name=RTX3060_6GB_LANE,
    ),
}

MODEL_PRESET_ALIASES: dict[str, str] = {
    "default": DEFAULT_CAPTURE_MODEL_PRESET,
    "base": QWEN35_9B_BASE_PRESET,
}

SUPPORTED_MODEL_IDS = {DEFAULT_MODEL_ID, BASE_MODEL_ID, QWEN35_4B_MODEL_ID}


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


def model_preset_names() -> tuple[str, ...]:
    """Return supported preset names including legacy aliases."""

    names = [*MODEL_PRESETS.keys(), *MODEL_PRESET_ALIASES.keys()]
    return tuple(dict.fromkeys(names))


def get_model_preset(name: str) -> CaptureModelPreset:
    """Resolve a named capture preset, preserving legacy aliases."""

    canonical_name = MODEL_PRESET_ALIASES.get(name, name)
    try:
        return MODEL_PRESETS[canonical_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model preset: {name}") from exc


def model_preset_to_id(name: str) -> str:
    """Return the model source used for a preset."""

    return get_model_preset(name).resolved_model_source()
