"""Helpers for KV capture and artifact metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import subprocess
from typing import Any

import torch

from turboquant.io_utils import write_json


@dataclass(slots=True)
class LayerCaptureRecord:
    layer_index: int
    key_shape: list[int]
    value_shape: list[int]
    key_file: str
    value_file: str


@dataclass(slots=True)
class CaptureMetadata:
    model_name: str
    tokenizer_name: str
    prompt_hash: str
    prompt_length: int
    timestamp_utc: str
    device: str
    dtype: str
    layer_count: int
    package_versions: dict[str, str]
    git_commit_hash: str | None
    layers: list[LayerCaptureRecord]


def package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def normalize_past_key_values(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")
    normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for item in past_key_values:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise TypeError(f"Unexpected cache layer entry: {type(item)!r}")
        key, value = item[0], item[1]
        normalized.append((key.detach(), value.detach()))
    return normalized


def save_capture_metadata(path: Path, metadata: CaptureMetadata) -> None:
    write_json(path=path, payload=asdict(metadata))
