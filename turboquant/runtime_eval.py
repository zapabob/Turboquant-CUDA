"""Runtime evaluation helpers for llama.cpp-based TurboQuant experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd


MCQ_RUNTIME_TASKS = ("hellaswag", "piqa", "arc_easy", "arc_challenge", "mmlu")
CHAT_RUNTIME_TASKS = ("gsm8k",)
CURRENT_MAIN_RUNTIME_PROFILES: dict[str, dict[str, str]] = {
    "exact": {},
    "turboquant_enabled_audit": {
        "LLAMA_TURBOQUANT": "1",
        "LLAMA_TURBOQUANT_SO8": "1",
        "LLAMA_TURBOQUANT_SO8_LEARNED": "0",
        "LLAMA_TURBOQUANT_TRIALITY": "1",
        "LLAMA_TURBOQUANT_TRIALITY_MIX": "0.5",
        "LLAMA_TURBOQUANT_ROTATION_SEED": "0",
    },
}
TURBOQUANT_ENV_KNOBS = (
    "LLAMA_TURBOQUANT",
    "LLAMA_TURBOQUANT_SO8",
    "LLAMA_TURBOQUANT_SO8_LEARNED",
    "LLAMA_TURBOQUANT_TRIALITY",
    "LLAMA_TURBOQUANT_TRIALITY_MIX",
    "LLAMA_TURBOQUANT_ROTATION_SEED",
)

_PPL_BRACKET_RE = re.compile(r"\[(?P<chunk>\d+)\](?P<ppl>\d+(?:\.\d+)?)")
_PPL_TABLE_RE = re.compile(
    r"^\s*(?P<offset>\d+)\s+(?P<ppl>\d+(?:\.\d+)?)\s+(?P<log_ppl>\d+(?:\.\d+)?)\s+(?P<log_ppl_std>\d+(?:\.\d+)?)\s*$"
)
_PPL_FINAL_RE = re.compile(r"Final estimate:\s+PPL\s*=\s*(?P<ppl>\d+(?:\.\d+)?)\s+\+/-\s+(?P<unc>\d+(?:\.\d+)?)")
_LM_EVAL_TIMESTAMP_RE = re.compile(r"_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)$")
_LM_EVAL_SAMPLE_RE = re.compile(r"^samples_(?P<task>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)$")


@dataclass(frozen=True, slots=True)
class RuntimeEvalCommandConfig:
    model_path: Path
    mode: str
    run_id: str
    backend: str = "runtime"


def resolve_runtime_profile_env(profile_name: str | None) -> dict[str, str]:
    """Resolve a named current-main runtime profile into concrete env overrides."""

    if profile_name is None or not profile_name.strip():
        return {}
    resolved = CURRENT_MAIN_RUNTIME_PROFILES.get(profile_name)
    if resolved is None:
        supported = ", ".join(sorted(CURRENT_MAIN_RUNTIME_PROFILES))
        raise ValueError(f"Unsupported runtime profile '{profile_name}'. Expected one of: {supported}")
    return dict(resolved)


def parse_runtime_env_json(runtime_env_json: str | None) -> dict[str, str]:
    """Parse inline JSON or a JSON file path into stringified environment overrides."""

    if runtime_env_json is None or not runtime_env_json.strip():
        return {}
    candidate_path = Path(runtime_env_json)
    payload_text = candidate_path.read_text(encoding="utf-8-sig") if candidate_path.exists() else runtime_env_json
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("runtime_env_json must decode to a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def merge_runtime_env_overrides(
    *,
    profile_name: str | None,
    runtime_env_json: str | None,
) -> dict[str, str]:
    """Merge a named runtime profile with optional JSON overrides."""

    merged = resolve_runtime_profile_env(profile_name)
    merged.update(parse_runtime_env_json(runtime_env_json))
    return merged


def build_llama_perplexity_command(
    *,
    binary_path: Path,
    model_path: Path,
    corpus_path: Path,
    context_size: int,
    batch_size: int,
    stride: int,
    chunks: int | None = None,
) -> list[str]:
    """Build a llama.cpp perplexity command."""

    command = [
        str(binary_path),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "--ctx-size",
        str(context_size),
        "--batch-size",
        str(batch_size),
        "--ppl-stride",
        str(stride),
    ]
    if chunks is not None:
        command.extend(["--chunks", str(chunks)])
    return command


def build_llama_bench_command(
    *,
    binary_path: Path,
    model_path: Path,
    repetitions: int,
    threads: int,
    n_prompt: int,
    n_gen: int,
    device: str = "auto",
) -> list[str]:
    """Build a llama-bench command that emits JSON for downstream parsing."""

    return [
        str(binary_path),
        "--model",
        str(model_path),
        "--repetitions",
        str(repetitions),
        "--threads",
        str(threads),
        "--n-prompt",
        str(n_prompt),
        "--n-gen",
        str(n_gen),
        "--device",
        device,
        "--output",
        "json",
    ]


def build_llama_server_command(
    *,
    binary_path: Path,
    model_path: Path,
    host: str,
    port: int,
    context_size: int,
    threads: int,
    model_alias: str,
    n_gpu_layers: int | str = 99,
) -> list[str]:
    """Build a llama-server command for OpenAI-compatible runtime evaluation."""

    return [
        str(binary_path),
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--threads",
        str(threads),
        "--alias",
        model_alias,
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]


def build_lm_eval_command(
    *,
    model_type: str,
    model_args: dict[str, str | int | float],
    tasks: list[str] | tuple[str, ...],
    output_path: Path,
    python_executable: str | Path | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    log_samples: bool = False,
    extra_flags: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Build an lm-eval-harness command for local OpenAI-compatible endpoints."""

    serialized_args = ",".join(f"{key}={value}" for key, value in model_args.items())
    command = [
        str(python_executable or sys.executable),
        "-m",
        "lm_eval",
        "--model",
        model_type,
        "--model_args",
        serialized_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if log_samples:
        command.append("--log_samples")
    if extra_flags:
        command.extend(extra_flags)
    return command


def wait_for_server_ready(
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
) -> dict[str, str | int]:
    """Wait until llama-server exposes the model list endpoint.

    Some current-main builds expose ``/health`` while others do not, so health is
    treated as best-effort metadata instead of a hard readiness requirement.
    """

    base_url = f"http://{host}:{port}"
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            health_status: int | None = None
            health_error: str | None = None
            try:
                with urllib_request.urlopen(f"{base_url}/health", timeout=5) as response:
                    health_status = int(response.status)
            except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError, ConnectionError) as exc:
                health_error = str(exc)
            with urllib_request.urlopen(f"{base_url}/v1/models", timeout=5) as models_response:
                if 200 <= int(models_response.status) < 300:
                    return {
                        "base_url": base_url,
                        "health_url": f"{base_url}/health",
                        "health_status": int(health_status) if health_status is not None else -1,
                        "health_error": health_error or "",
                        "models_url": f"{base_url}/v1/models",
                        "status": int(models_response.status),
                    }
        except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for llama-server readiness at {base_url}: {last_error}")


def probe_local_completions_loglikelihood_support(
    *,
    completions_url: str,
    model_name: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Probe whether the local completions endpoint exposes prompt logprobs.

    Current-main `llama-server` returns OpenAI-compatible completion objects, but
    prompt logprob support is required for lm-eval multiple-choice tasks. This
    helper records the exact response shape so audit-mode runs can distinguish
    "unsupported by runtime schema" from ordinary command failures.
    """

    payload = {
        "model": model_name,
        "prompt": "Question: 2+2 =",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "echo": True,
        "seed": 1234,
    }
    request = urllib_request.Request(
        completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        response_text = response.read().decode("utf-8")
        response_json = json.loads(response_text)

    first_choice = (response_json.get("choices") or [{}])[0]
    logprobs_payload = first_choice.get("logprobs") or {}
    content_items = logprobs_payload.get("content")
    token_logprobs = logprobs_payload.get("token_logprobs")
    usage_payload = response_json.get("usage") or {}
    prompt_tokens = int(usage_payload.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage_payload.get("completion_tokens", 0) or 0)
    content_count = len(content_items) if isinstance(content_items, list) else 0
    token_logprobs_count = len(token_logprobs) if isinstance(token_logprobs, list) else 0
    supports_prompt_logprobs = bool(
        isinstance(token_logprobs, list)
        and prompt_tokens > 0
        and token_logprobs_count >= (prompt_tokens + completion_tokens)
    )
    return {
        "completions_url": completions_url,
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "has_logprobs_content": isinstance(content_items, list),
        "logprobs_content_count": content_count,
        "has_token_logprobs": isinstance(token_logprobs, list),
        "token_logprobs_count": token_logprobs_count,
        "supports_prompt_logprobs": supports_prompt_logprobs,
        "response_excerpt": response_json,
    }


def parse_llama_bench_json(
    text: str,
    *,
    mode: str,
    backend: str,
    run_id: str,
) -> pd.DataFrame:
    """Expand llama-bench JSON output into one row per throughput sample."""

    payload = json.loads(text)
    rows: list[dict[str, str | int | float]] = []
    for entry_idx, entry in enumerate(payload):
        samples = entry.get("samples_ts", [])
        for sample_idx, value in enumerate(samples):
            rows.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "run_id": run_id,
                    "sample_group": entry_idx,
                    "sample_idx": sample_idx,
                    "test": str(entry.get("test", "")),
                    "n_prompt": int(entry.get("n_prompt", 0)),
                    "n_gen": int(entry.get("n_gen", 0)),
                    "metric": "tokens_per_second",
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def parse_perplexity_output(
    text: str,
    *,
    mode: str,
    backend: str,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse llama.cpp perplexity stdout into chunk rows and final summary rows."""

    chunk_rows: list[dict[str, str | int | float]] = []
    final_rows: list[dict[str, str | float]] = []
    for line in text.splitlines():
        bracket_matches = list(_PPL_BRACKET_RE.finditer(line))
        if bracket_matches:
            for bracket_match in bracket_matches:
                ppl = float(bracket_match.group("ppl"))
                chunk_rows.append(
                    {
                        "backend": backend,
                        "mode": mode,
                        "run_id": run_id,
                        "chunk_id": int(bracket_match.group("chunk")),
                        "perplexity": ppl,
                        "log_perplexity": math.log(ppl),
                    }
                )
            continue

        table_match = _PPL_TABLE_RE.match(line)
        if table_match is not None:
            ppl = float(table_match.group("ppl"))
            chunk_rows.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "run_id": run_id,
                    "chunk_id": int(table_match.group("offset")),
                    "perplexity": ppl,
                    "log_perplexity": float(table_match.group("log_ppl")),
                    "log_perplexity_std": float(table_match.group("log_ppl_std")),
                }
            )
            continue

        final_match = _PPL_FINAL_RE.search(line)
        if final_match is not None:
            final_ppl = float(final_match.group("ppl"))
            final_rows.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "run_id": run_id,
                    "metric": "perplexity",
                    "mean": final_ppl,
                    "uncertainty": float(final_match.group("unc")),
                    "log_perplexity": math.log(final_ppl),
                }
            )
    if chunk_rows and not final_rows:
        mean_ppl = sum(float(row["perplexity"]) for row in chunk_rows) / len(chunk_rows)
        final_rows.append(
            {
                "backend": backend,
                "mode": mode,
                "run_id": run_id,
                "metric": "perplexity",
                "mean": mean_ppl,
                "uncertainty": float("nan"),
                "log_perplexity": math.log(mean_ppl),
            }
        )
    return pd.DataFrame(chunk_rows), pd.DataFrame(final_rows)


def load_lm_eval_results(path: Path) -> pd.DataFrame:
    """Load lm-eval JSON output into a standardized task summary frame."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    rows: list[dict[str, str | float]] = []
    for task, metrics in results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            if "_stderr" in metric_name:
                continue
            metric_head, separator, metric_tail = metric_name.partition(",")
            stderr_name = f"{metric_head}_stderr{separator}{metric_tail}" if separator else f"{metric_head}_stderr"
            sem = float(metrics[stderr_name]) if stderr_name in metrics and isinstance(metrics[stderr_name], (int, float)) else float("nan")
            ci95_low = float(value) - (1.96 * sem) if not math.isnan(sem) else float("nan")
            ci95_high = float(value) + (1.96 * sem) if not math.isnan(sem) else float("nan")
            rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "mean": float(value),
                    "sem": sem,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                }
            )
    return pd.DataFrame(rows)


def resolve_lm_eval_results_path(output_stub: Path) -> Path | None:
    """Resolve the timestamped lm-eval aggregate JSON path for an output stub."""

    if output_stub.exists():
        return output_stub
    candidates = sorted(
        output_stub.parent.glob(f"{output_stub.stem}_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def resolve_lm_eval_sample_paths(*, output_stub: Path, task_names: list[str] | tuple[str, ...]) -> list[Path]:
    """Resolve per-task sample JSONL files that belong to one lm-eval run."""

    aggregate_path = resolve_lm_eval_results_path(output_stub)
    if aggregate_path is None:
        return []
    stamp_match = _LM_EVAL_TIMESTAMP_RE.search(aggregate_path.stem)
    if stamp_match is None:
        return []
    stamp = stamp_match.group("stamp")
    paths: list[Path] = []
    for task_name in task_names:
        sample_path = output_stub.parent / f"samples_{task_name}_{stamp}.jsonl"
        if sample_path.exists():
            paths.append(sample_path)
    return paths


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_mcq_prediction(filtered_resps: list[Any]) -> str:
    scored: list[tuple[int, float]] = []
    for index, item in enumerate(filtered_resps):
        if isinstance(item, (list, tuple)) and item:
            score = _coerce_float(item[0])
        else:
            score = _coerce_float(item)
        if score is not None:
            scored.append((index, score))
    if not scored:
        return ""
    return str(max(scored, key=lambda pair: pair[1])[0])


def _infer_text_prediction(filtered_resps: list[Any]) -> str:
    if not filtered_resps:
        return ""
    first = filtered_resps[0]
    if isinstance(first, (list, tuple)) and first:
        return str(first[0])
    return str(first)


def _task_name_from_sample_path(path: Path) -> str:
    match = _LM_EVAL_SAMPLE_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Could not infer task name from sample path: {path}")
    return match.group("task")


def load_lm_eval_sample_results(
    paths: list[Path] | tuple[Path, ...],
    *,
    mode: str,
    backend: str,
    run_id: str,
    preferred_metrics: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Load per-item lm-eval sample logs into a standardized item frame."""

    rows: list[dict[str, str | int | float]] = []
    for path in paths:
        task_name = _task_name_from_sample_path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            filtered_resps = payload.get("filtered_resps", [])
            reference = str(payload.get("target", ""))
            metric_name = next((name for name in preferred_metrics if name in payload), "derived")
            metric_value = payload.get(metric_name, None)
            prediction = ""
            if reference.isdigit():
                prediction = _infer_mcq_prediction(filtered_resps)
            if not prediction:
                prediction = _infer_text_prediction(filtered_resps)
            if isinstance(metric_value, (int, float)):
                is_correct = int(float(metric_value) >= 0.5)
            else:
                is_correct = int(prediction == reference)
            rows.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "run_id": run_id,
                    "task": task_name,
                    "doc_id": str(payload.get("doc_id", "")),
                    "prediction": prediction,
                    "reference": reference,
                    "is_correct": is_correct,
                    "correctness_metric": metric_name,
                }
            )
    return pd.DataFrame(rows)


def _collect_source_matches(
    *,
    root_dir: Path,
    needles: list[str] | tuple[str, ...],
    suffixes: tuple[str, ...] = (".cpp", ".h", ".cu", ".cuh", ".md"),
) -> dict[str, list[str]]:
    matches = {needle: [] for needle in needles}
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        relative = path.relative_to(root_dir).as_posix()
        for line_number, line in enumerate(lines, start=1):
            for needle in needles:
                if needle in line:
                    matches[needle].append(f"{relative}:{line_number}")
    return matches


def _git_head_or_unknown(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def audit_zapabob_runtime_checkout(
    llama_cpp_dir: Path,
    *,
    binary_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Inspect a zapabob/llama.cpp checkout for current-main TurboQuant runtime limitations."""

    src_dir = llama_cpp_dir / "src"
    kv_cache_path = src_dir / "llama-kv-cache.cpp"
    kv_cache_lines = kv_cache_path.read_text(encoding="utf-8").splitlines() if kv_cache_path.exists() else []
    env_locations = _collect_source_matches(root_dir=llama_cpp_dir, needles=TURBOQUANT_ENV_KNOBS)
    config_locations = _collect_source_matches(
        root_dir=src_dir,
        needles=(
            "turboquant_cfg.enabled",
            "turboquant_cfg.so8_enabled",
            "turboquant_cfg.so8_learned",
            "turboquant_cfg.triality_enabled",
            "turboquant_cfg.triality_mix",
            "turboquant_cfg.rotation_seed",
        ),
        suffixes=(".cpp", ".h"),
    )
    helper_symbol_locations = _collect_source_matches(
        root_dir=llama_cpp_dir,
        needles=(
            "llama_turboquant_apply_so8_rotation",
            "llama_turboquant_train_triality_codebook",
            "llama_turboquant_evaluate_triality",
            "llama_turboquant_load_artifact",
        ),
        suffixes=(".cpp", ".h"),
    )

    src_helper_runtime_hits = {
        symbol: [
            location
            for location in helper_symbol_locations[symbol]
            if location.startswith("src/")
            and not location.startswith("src/llama-turboquant.cpp")
            and not location.startswith("src/llama-turboquant.h")
        ]
        for symbol in helper_symbol_locations
    }
    tools_helper_hits = {
        symbol: [location for location in helper_symbol_locations[symbol] if location.startswith("tools/")]
        for symbol in helper_symbol_locations
    }

    log_evidence = {
        "enabled_banner": [
            f"src/llama-kv-cache.cpp:{line_number}"
            for line_number, line in enumerate(kv_cache_lines, start=1)
            if "TurboQuant enabled (so8=%d, so8_learned=%d, triality=%d, mix=%.3f, seed=%u)" in line
        ],
        "k_path_log": [
            f"src/llama-kv-cache.cpp:{line_number}"
            for line_number, line in enumerate(kv_cache_lines, start=1)
            if "TurboQuant K-path active" in line
        ],
        "v_path_log": [
            f"src/llama-kv-cache.cpp:{line_number}"
            for line_number, line in enumerate(kv_cache_lines, start=1)
            if "TurboQuant V-path active" in line
        ],
    }

    config_parse_log_only = all(
        len(config_locations[field]) <= 1 for field in config_locations if field != "turboquant_cfg.enabled"
    )
    enabled_gate_log_only = (
        len(config_locations["turboquant_cfg.enabled"]) <= 3
        and not any(src_helper_runtime_hits.values())
        and bool(log_evidence["k_path_log"])
        and bool(log_evidence["v_path_log"])
    )

    audit: dict[str, Any] = {
        "llama_cpp_dir": str(llama_cpp_dir),
        "git_commit": _git_head_or_unknown(llama_cpp_dir),
        "binary_paths": {name: str(path) for name, path in sorted((binary_paths or {}).items())},
        "env_knobs": list(TURBOQUANT_ENV_KNOBS),
        "env_knob_locations": env_locations,
        "config_field_locations": config_locations,
        "helper_symbol_locations": helper_symbol_locations,
        "src_helper_runtime_hits": src_helper_runtime_hits,
        "tools_helper_hits": tools_helper_hits,
        "log_evidence": log_evidence,
        "findings": {
            "so8_triality_parse_log_only": config_parse_log_only,
            "llama_turboquant_enabled_is_log_only_gate": enabled_gate_log_only,
            "artifact_loader_only_used_by_tooling": not src_helper_runtime_hits["llama_turboquant_load_artifact"]
            and bool(tools_helper_hits["llama_turboquant_load_artifact"]),
            "mode_selectable_runtime_present": False,
        },
        "notes": [
            "Current external fork main exposes LLAMA_TURBOQUANT* env knobs and logs them, but the audited source tree does not call TurboQuant transform/artifact helpers from src/ runtime code.",
            "tools/turboquant contains artifact training/readback helpers, so embedded TurboQuant GGUF packaging remains a secondary validation path rather than an active llama-server selector path.",
        ],
    }
    return audit


def render_runtime_audit_markdown(audit_payload: dict[str, Any]) -> str:
    """Render a human-readable current-main runtime audit summary."""

    findings = audit_payload["findings"]
    lines = [
        "# zapabob llama.cpp Current-main Runtime Audit",
        "",
        f"- git_commit: `{audit_payload['git_commit']}`",
        f"- llama_cpp_dir: `{audit_payload['llama_cpp_dir']}`",
        "",
        "## Findings",
        "",
        f"- `so8_triality_parse_log_only`: `{findings['so8_triality_parse_log_only']}`",
        f"- `llama_turboquant_enabled_is_log_only_gate`: `{findings['llama_turboquant_enabled_is_log_only_gate']}`",
        f"- `artifact_loader_only_used_by_tooling`: `{findings['artifact_loader_only_used_by_tooling']}`",
        f"- `mode_selectable_runtime_present`: `{findings['mode_selectable_runtime_present']}`",
        "",
        "## Binary Paths",
        "",
    ]
    binary_paths = audit_payload.get("binary_paths", {})
    if binary_paths:
        for name, path in binary_paths.items():
            lines.append(f"- `{name}`: `{path}`")
    else:
        lines.append("- _No binary paths recorded._")
    lines.extend(
        [
            "",
            "## Log Evidence",
            "",
        ]
    )
    for label, locations in audit_payload["log_evidence"].items():
        if locations:
            lines.append(f"- `{label}`: {', '.join(f'`{location}`' for location in locations)}")
        else:
            lines.append(f"- `{label}`: _not found_")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in audit_payload["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


__all__ = [
    "CHAT_RUNTIME_TASKS",
    "CURRENT_MAIN_RUNTIME_PROFILES",
    "MCQ_RUNTIME_TASKS",
    "RuntimeEvalCommandConfig",
    "TURBOQUANT_ENV_KNOBS",
    "audit_zapabob_runtime_checkout",
    "build_llama_bench_command",
    "build_llama_perplexity_command",
    "build_llama_server_command",
    "build_lm_eval_command",
    "load_lm_eval_sample_results",
    "load_lm_eval_results",
    "merge_runtime_env_overrides",
    "parse_llama_bench_json",
    "parse_perplexity_output",
    "parse_runtime_env_json",
    "probe_local_completions_loglikelihood_support",
    "render_runtime_audit_markdown",
    "resolve_runtime_profile_env",
    "resolve_lm_eval_results_path",
    "resolve_lm_eval_sample_paths",
    "wait_for_server_ready",
]
