from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.io_utils import ensure_dir
from turboquant.runtime_eval import (
    CHAT_RUNTIME_TASKS,
    MCQ_RUNTIME_TASKS,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
    build_lm_eval_command,
    load_lm_eval_sample_results,
    load_lm_eval_results,
    merge_runtime_env_overrides,
    parse_llama_bench_json,
    parse_perplexity_output,
    probe_local_completions_loglikelihood_support,
    resolve_lm_eval_results_path,
    resolve_lm_eval_sample_paths,
    wait_for_server_ready,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current-main runtime-side Qwen evaluation via llama.cpp + lm-eval.")
    parser.add_argument("--mode", required=True, help="Mode label for the evaluated runtime.")
    parser.add_argument("--model-path", required=True, help="GGUF model path.")
    parser.add_argument("--perplexity-bin", default=None, help="llama.cpp perplexity binary path.")
    parser.add_argument("--llama-bench-bin", default=None, help="llama.cpp llama-bench binary path.")
    parser.add_argument("--server-bin", default=None, help="Optional llama-server binary path to launch locally.")
    parser.add_argument("--corpus-file", default=None, help="Plain-text corpus for perplexity evaluation.")
    parser.add_argument("--server-base-url", default=None, help="OpenAI-compatible llama-server base URL.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Host for a spawned llama-server instance.")
    parser.add_argument("--server-port", type=int, default=8080, help="Port for a spawned llama-server instance.")
    parser.add_argument("--server-ready-timeout-sec", type=float, default=120.0, help="Startup timeout for llama-server readiness checks.")
    parser.add_argument("--server-log-prefix", default=None, help="Optional custom prefix for llama-server stdout/stderr logs.")
    parser.add_argument("--server-context-size", type=int, default=4096, help="Context size passed to llama-server.")
    parser.add_argument("--server-n-gpu-layers", default="99", help="GPU layer setting passed to llama-server.")
    parser.add_argument("--server-model-name", default="qwen-runtime", help="Model alias exposed by llama-server.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional local tokenizer path for lm-eval local-completions backends.")
    parser.add_argument("--runtime-profile", default=None, help="Named current-main runtime env profile.")
    parser.add_argument("--runtime-env-json", default=None, help="Inline JSON or JSON file path for env overrides.")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count for runtime tools.")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions for llama-bench.")
    parser.add_argument("--context-size", type=int, default=512, help="PPL context length.")
    parser.add_argument("--batch-size", type=int, default=128, help="PPL batch size.")
    parser.add_argument("--stride", type=int, default=256, help="PPL stride.")
    parser.add_argument("--chunks", type=int, default=0, help="Optional PPL chunk cap (0=all).")
    parser.add_argument("--n-prompt", type=int, default=256, help="Prompt tokens for llama-bench.")
    parser.add_argument("--n-gen", type=int, default=64, help="Generation tokens for llama-bench.")
    parser.add_argument("--mcq-tasks", default=",".join(MCQ_RUNTIME_TASKS), help="Comma-separated MCQ task list.")
    parser.add_argument("--chat-tasks", default=",".join(CHAT_RUNTIME_TASKS), help="Comma-separated chat task list.")
    parser.add_argument("--lm-eval-limit", type=int, default=0, help="Optional lm-eval task limit (0=all).")
    parser.add_argument("--allow-mcq-unavailable", action="store_true", help="Treat missing prompt-logprob support as an audit artifact instead of a hard failure.")
    parser.add_argument("--output-dir", default="artifacts/runtime_eval", help="Artifact output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write command manifests without executing tools.")
    return parser.parse_args(argv)


def _write_exit_code(path: Path, code: int) -> None:
    path.write_text(str(code), encoding="utf-8")


def _run_command(
    command: list[str],
    *,
    log_prefix: Path,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log_prefix.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
    _write_exit_code(log_prefix.with_suffix(".exit_code.txt"), int(result.returncode))
    return result


def _serialize_command(command: list[str], *, env_updates: dict[str, str]) -> dict[str, object]:
    return {
        "command": command,
        "env": env_updates,
    }


def _start_llama_server(
    command: list[str],
    *,
    env_updates: dict[str, str],
    log_prefix: Path,
) -> tuple[subprocess.Popen[str], TextIO, TextIO]:
    env = os.environ.copy()
    env.update(env_updates)
    stdout_handle = log_prefix.with_suffix(".stdout.log").open("w", encoding="utf-8")
    stderr_handle = log_prefix.with_suffix(".stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return process, stdout_handle, stderr_handle


def _stop_llama_server(
    process: subprocess.Popen[str],
    *,
    stdout_handle: TextIO,
    stderr_handle: TextIO,
    log_prefix: Path,
) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    exit_code = int(process.returncode if process.returncode is not None else process.wait(timeout=5))
    stdout_handle.close()
    stderr_handle.close()
    _write_exit_code(log_prefix.with_suffix(".exit_code.txt"), exit_code)
    return exit_code


def _split_tasks(serialized: str) -> list[str]:
    return [task.strip() for task in serialized.split(",") if task.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output_dir))
    metrics_dir = ensure_dir(output_dir / "metrics")
    logs_dir = ensure_dir(output_dir / "logs")

    runtime_env = merge_runtime_env_overrides(
        profile_name=args.runtime_profile,
        runtime_env_json=args.runtime_env_json,
    )
    mcq_tasks = _split_tasks(args.mcq_tasks)
    chat_tasks = _split_tasks(args.chat_tasks)
    limit = int(args.lm_eval_limit) if int(args.lm_eval_limit) > 0 else None

    effective_base_url = args.server_base_url.rstrip("/") if args.server_base_url else None
    if args.server_bin and effective_base_url is None:
        effective_base_url = f"http://{args.server_host}:{int(args.server_port)}"

    run_meta = {
        "mode": args.mode,
        "model_path": args.model_path,
        "perplexity_bin": args.perplexity_bin,
        "llama_bench_bin": args.llama_bench_bin,
        "server_bin": args.server_bin,
        "corpus_file": args.corpus_file,
        "server_base_url": effective_base_url,
        "server_host": args.server_host,
        "server_port": int(args.server_port),
        "server_model_name": args.server_model_name,
        "server_ready_timeout_sec": float(args.server_ready_timeout_sec),
        "server_context_size": int(args.server_context_size),
        "server_n_gpu_layers": str(args.server_n_gpu_layers),
        "tokenizer_path": args.tokenizer_path,
        "runtime_profile": args.runtime_profile,
        "runtime_env": runtime_env,
        "threads": args.threads,
        "repetitions": args.repetitions,
        "context_size": args.context_size,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "chunks": args.chunks,
        "n_prompt": args.n_prompt,
        "n_gen": args.n_gen,
        "mcq_tasks": mcq_tasks,
        "chat_tasks": chat_tasks,
        "lm_eval_limit": args.lm_eval_limit,
        "allow_mcq_unavailable": args.allow_mcq_unavailable,
        "dry_run": args.dry_run,
    }
    (metrics_dir / "runtime_eval_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    commands: dict[str, dict[str, object]] = {}
    if args.perplexity_bin and args.corpus_file:
        commands["perplexity"] = _serialize_command(
            build_llama_perplexity_command(
                binary_path=Path(args.perplexity_bin),
                model_path=Path(args.model_path),
                corpus_path=Path(args.corpus_file),
                context_size=int(args.context_size),
                batch_size=int(args.batch_size),
                stride=int(args.stride),
                chunks=int(args.chunks) if int(args.chunks) > 0 else None,
            ),
            env_updates=runtime_env,
        )
    if args.llama_bench_bin:
        commands["llama_bench"] = _serialize_command(
            build_llama_bench_command(
                binary_path=Path(args.llama_bench_bin),
                model_path=Path(args.model_path),
                repetitions=int(args.repetitions),
                threads=int(args.threads),
                n_prompt=int(args.n_prompt),
                n_gen=int(args.n_gen),
            ),
            env_updates=runtime_env,
        )
    if args.server_bin:
        commands["server_start"] = _serialize_command(
            build_llama_server_command(
                binary_path=Path(args.server_bin),
                model_path=Path(args.model_path),
                host=args.server_host,
                port=int(args.server_port),
                context_size=int(args.server_context_size),
                threads=int(args.threads),
                model_alias=args.server_model_name,
                n_gpu_layers=args.server_n_gpu_layers,
            ),
            env_updates=runtime_env,
        )
    if effective_base_url:
        if mcq_tasks:
            mcq_model_args: dict[str, str | int | bool] = {
                "base_url": f"{effective_base_url.rstrip('/')}/v1/completions",
                "model": args.server_model_name,
                "num_concurrent": 1,
                "max_retries": 3,
                "tokenized_requests": False,
            }
            if args.tokenizer_path:
                mcq_model_args["tokenizer"] = args.tokenizer_path
                mcq_model_args["tokenizer_backend"] = "huggingface"
            commands["lm_eval_mcq"] = _serialize_command(
                build_lm_eval_command(
                    model_type="local-completions",
                    model_args=mcq_model_args,
                    tasks=mcq_tasks,
                    output_path=logs_dir / "lm_eval_mcq.json",
                    batch_size=1,
                    limit=limit,
                    log_samples=True,
                ),
                env_updates=runtime_env,
            )
        if chat_tasks:
            chat_model_args: dict[str, str | int | bool] = {
                "base_url": f"{effective_base_url.rstrip('/')}/v1/chat/completions",
                "model": args.server_model_name,
                "num_concurrent": 1,
                "max_retries": 3,
            }
            if args.tokenizer_path:
                chat_model_args["tokenizer"] = args.tokenizer_path
                chat_model_args["tokenizer_backend"] = "huggingface"
            commands["lm_eval_chat"] = _serialize_command(
                build_lm_eval_command(
                    model_type="local-chat-completions",
                    model_args=chat_model_args,
                    tasks=chat_tasks,
                    output_path=logs_dir / "lm_eval_chat.json",
                    batch_size=1,
                    limit=limit,
                    log_samples=True,
                    extra_flags=["--fewshot_as_multiturn", "--apply_chat_template"],
                ),
                env_updates=runtime_env,
            )

    (logs_dir / "runtime_eval_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    if args.dry_run:
        return 0

    for name in ("perplexity", "llama_bench"):
        payload = commands.get(name)
        if payload is None:
            continue
        result = _run_command(
            list(payload["command"]),
            log_prefix=logs_dir / name,
            env_updates=dict(payload["env"]),
        )
        if result.returncode != 0:
            return int(result.returncode)
        if name == "perplexity":
            chunk_frame, summary_frame = parse_perplexity_output(
                result.stdout,
                mode=args.mode,
                backend="runtime",
                run_id=name,
            )
            if not chunk_frame.empty:
                chunk_frame.to_csv(metrics_dir / "runtime_ppl_chunks.csv", index=False)
            if not summary_frame.empty:
                summary_frame.to_csv(metrics_dir / "runtime_ppl_tool_summary.csv", index=False)
        elif name == "llama_bench":
            bench_frame = parse_llama_bench_json(
                result.stdout,
                mode=args.mode,
                backend="runtime",
                run_id=name,
            )
            if not bench_frame.empty:
                bench_frame.to_csv(metrics_dir / "runtime_bench_samples.csv", index=False)

    server_process: subprocess.Popen[str] | None = None
    stdout_handle: TextIO | None = None
    stderr_handle: TextIO | None = None
    server_log_prefix = Path(args.server_log_prefix) if args.server_log_prefix else logs_dir / "llama_server"
    mcq_probe: dict[str, object] | None = None
    try:
        if "server_start" in commands:
            payload = commands["server_start"]
            server_process, stdout_handle, stderr_handle = _start_llama_server(
                list(payload["command"]),
                env_updates=dict(payload["env"]),
                log_prefix=server_log_prefix,
            )
            try:
                ready = wait_for_server_ready(
                    host=args.server_host,
                    port=int(args.server_port),
                    timeout_seconds=float(args.server_ready_timeout_sec),
                )
            except TimeoutError as exc:
                (logs_dir / "llama_server_ready_error.txt").write_text(str(exc), encoding="utf-8")
                if server_process.poll() is not None:
                    return int(server_process.returncode or 1)
                return 1
            ready_error_path = logs_dir / "llama_server_ready_error.txt"
            if ready_error_path.exists():
                ready_error_path.unlink()
            (logs_dir / "llama_server_ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            if mcq_tasks:
                mcq_probe = probe_local_completions_loglikelihood_support(
                    completions_url=f"{effective_base_url.rstrip('/')}/v1/completions",
                    model_name=args.server_model_name,
                )
                (logs_dir / "llama_server_loglikelihood_probe.json").write_text(
                    json.dumps(mcq_probe, indent=2),
                    encoding="utf-8",
                )

        for name in ("lm_eval_mcq", "lm_eval_chat"):
            payload = commands.get(name)
            if payload is None:
                continue
            if name == "lm_eval_mcq" and mcq_probe is not None and not bool(mcq_probe.get("supports_prompt_logprobs")):
                unavailable_payload = {
                    "backend": "runtime",
                    "mode": args.mode,
                    "run_id": name,
                    "reason": "llama-server completions response does not expose prompt token_logprobs for lm-eval loglikelihood tasks",
                    "probe_path": str(logs_dir / "llama_server_loglikelihood_probe.json"),
                }
                (metrics_dir / "lm_eval_mcq_unavailable.json").write_text(
                    json.dumps(unavailable_payload, indent=2),
                    encoding="utf-8",
                )
                (logs_dir / "lm_eval_mcq.skip_reason.txt").write_text(
                    unavailable_payload["reason"],
                    encoding="utf-8",
                )
                if args.allow_mcq_unavailable:
                    continue
                return 1
            result = _run_command(
                list(payload["command"]),
                log_prefix=logs_dir / name,
                env_updates=dict(payload["env"]),
            )
            if result.returncode != 0:
                return int(result.returncode)
            output_stub = logs_dir / f"{name}.json"
            json_path = resolve_lm_eval_results_path(output_stub)
            if json_path is not None and json_path.exists():
                summary_frame = load_lm_eval_results(json_path)
                if not summary_frame.empty:
                    summary_frame = summary_frame.assign(backend="runtime", mode=args.mode, run_id=name)
                    summary_frame.to_csv(metrics_dir / f"{name}_summary.csv", index=False)
                task_names = mcq_tasks if name == "lm_eval_mcq" else chat_tasks
                preferred_metrics = ("acc", "acc_norm") if name == "lm_eval_mcq" else ("exact_match", "em", "strict-match")
                sample_paths = resolve_lm_eval_sample_paths(output_stub=output_stub, task_names=task_names)
                if sample_paths:
                    sample_frame = load_lm_eval_sample_results(
                        sample_paths,
                        mode=args.mode,
                        backend="runtime",
                        run_id=name,
                        preferred_metrics=preferred_metrics,
                    )
                    if not sample_frame.empty:
                        sample_frame.to_csv(metrics_dir / f"{name}_items.csv", index=False)
    finally:
        if server_process is not None and stdout_handle is not None and stderr_handle is not None:
            _stop_llama_server(
                server_process,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                log_prefix=server_log_prefix,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
