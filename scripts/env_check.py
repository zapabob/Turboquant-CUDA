from __future__ import annotations

from datetime import datetime, timezone
import importlib
import os
import platform
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REPORT_PATH = Path(os.environ.get("TURBOQUANT_ENV_CHECK_PATH", "artifacts/reports/env_check.txt"))


def try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - exercised in real envs
        return None, repr(exc)


from turboquant.runtime import (
    DEFAULT_MODEL_ID,
    REQUIRED_CUDA,
    cuda_matches_target,
    python_is_supported,
    torch_cuda_version,
)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    failures: list[str] = []

    lines.append(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"python: {sys.version}")
    lines.append(f"python_supported: {python_is_supported()}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"default_model: {DEFAULT_MODEL_ID}")
    lines.append("default_weight_load: 4bit")

    torch_mod, torch_error = try_import("torch")
    if torch_mod is None:
        failures.append(f"torch import failed: {torch_error}")
    else:
        lines.append(f"torch: {torch_mod.__version__}")
        lines.append(f"torch_cuda: {torch_cuda_version(torch_mod)}")
        cuda_available = bool(torch_mod.cuda.is_available())
        lines.append(f"cuda_available: {cuda_available}")
        lines.append(f"target_cuda: {REQUIRED_CUDA}")
        lines.append(f"target_cuda_match: {cuda_matches_target(torch_mod)}")
        device_count = int(torch_mod.cuda.device_count()) if cuda_available else 0
        lines.append(f"cuda_device_count: {device_count}")
        for idx in range(device_count):
            lines.append(f"gpu_{idx}: {torch_mod.cuda.get_device_name(idx)}")
        if not python_is_supported():
            failures.append("python version is not 3.12.x")
        if not cuda_matches_target(torch_mod):
            failures.append(f"CUDA backend mismatch: expected {REQUIRED_CUDA} with CUDA available")

    transformers_mod, transformers_error = try_import("transformers")
    if transformers_mod is None:
        failures.append(f"transformers import failed: {transformers_error}")
    else:
        lines.append(f"transformers: {transformers_mod.__version__}")

    bitsandbytes_mod, bitsandbytes_error = try_import("bitsandbytes")
    if bitsandbytes_mod is None:
        lines.append(f"bitsandbytes: unavailable ({bitsandbytes_error})")
    else:
        lines.append(f"bitsandbytes: {getattr(bitsandbytes_mod, '__version__', 'imported')}")

    status = "ok" if not failures else "error"
    lines.append(f"status: {status}")
    if failures:
        lines.append("failures:")
        lines.extend(f"- {failure}" for failure in failures)

    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
