"""Helpers for KV capture and artifact metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import torch

from turboquant.io_utils import write_json


@dataclass(slots=True)
class PromptCaptureSpec:
    label: str
    prompt: str


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
    model_source: str | None = None
    prompt_label: str | None = None
    capture_id: str | None = None


DEFAULT_PROMPT_PANEL: tuple[PromptCaptureSpec, ...] = (
    PromptCaptureSpec(
        label="explain",
        prompt="Explain TurboQuant KV cache compression in one short paragraph.",
    ),
    PromptCaptureSpec(
        label="reasoning",
        prompt="Give a concise numbered list of three reasons why key-only KV compression can be more stable than full-KV compression.",
    ),
    PromptCaptureSpec(
        label="coding",
        prompt="Write a minimal Python function that computes softmax attention scores from query and key tensors.",
    ),
    PromptCaptureSpec(
        label="summary",
        prompt="Summarize the tradeoff between memory savings and hidden-state drift in KV cache quantization in two sentences.",
    ),
)


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
    elif hasattr(past_key_values, "layers"):
        normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                normalized.append((keys.detach(), values.detach()))
        if normalized:
            return normalized
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        normalized = []
        for key, value in zip(past_key_values.key_cache, past_key_values.value_cache, strict=True):
            if key is None or value is None:
                continue
            normalized.append((key.detach(), value.detach()))
        if normalized:
            return normalized
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


def capture_slug(label: str, prompt_hash: str) -> str:
    safe_label = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not safe_label:
        safe_label = "prompt"
    return f"{safe_label}-{prompt_hash[:8]}"


def load_capture_metadata(path: Path) -> CaptureMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = [LayerCaptureRecord(**record) for record in payload.get("layers", [])]
    payload["layers"] = layers
    return CaptureMetadata(**payload)
