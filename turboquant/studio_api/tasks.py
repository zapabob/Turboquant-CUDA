"""Studio job handlers built on top of existing TurboQuant scripts and modules."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
from typing import Any

from turboquant.capture import DEFAULT_PROMPT_PANEL, PromptCaptureSpec, capture_slug
from turboquant.gguf_profiles import build_hypura_serve_command, read_hypura_gguf_bridge_config
from turboquant.io_utils import stable_hash
from turboquant.runtime_eval import build_llama_server_command, merge_runtime_env_overrides
from turboquant.studio_api.models import (
    CaptureFormState,
    ExportOnlineReportSpec,
    ExportReportSpec,
    MatrixRunSpec,
    PackageSpec,
    PaperValidateSpec,
    RuntimeEvalSpec,
    ServeSpec,
)


class JobCancelledError(RuntimeError):
    """Raised when a Studio job is canceled."""


def _cli_command(script_name: str, args: list[str]) -> list[str]:
    return ["uv", "run", "python", f"scripts/{script_name}", *args]


def _resolved_prompts(spec: CaptureFormState) -> list[PromptCaptureSpec]:
    if spec.prompt is not None:
        return [PromptCaptureSpec(label=spec.prompt_label, prompt=spec.prompt)]
    return list(DEFAULT_PROMPT_PANEL)


def build_capture_preview(spec: CaptureFormState) -> dict[str, Any]:
    prompts = _resolved_prompts(spec)
    capture_targets = [
        {
            "label": prompt.label,
            "prompt_hash": stable_hash(prompt.prompt),
            "capture_id": capture_slug(prompt.label, stable_hash(prompt.prompt)),
        }
        for prompt in prompts
    ]
    args = []
    if spec.model_id is not None:
        args.extend(["--model-id", spec.model_id])
    if spec.model_preset is not None:
        args.extend(["--model-preset", spec.model_preset])
    if spec.lane_name is not None:
        args.extend(["--lane-name", spec.lane_name])
    if spec.prompt is not None:
        args.extend(["--prompt", spec.prompt, "--prompt-label", spec.prompt_label])
    args.extend(
        [
            "--output-dir",
            spec.output_dir,
            "--weight-load",
            spec.weight_load,
            "--dtype",
            spec.dtype,
            "--max-length",
            str(spec.max_length),
            "--seed",
            str(spec.seed),
        ]
    )
    if spec.trust_remote_code:
        args.append("--trust-remote-code")
    return {
        "command": _cli_command("capture_qwen_kv.py", args),
        "expected_outputs": [str(Path(spec.output_dir) / target["capture_id"]) for target in capture_targets],
        "capture_targets": capture_targets,
    }


def build_paper_preview(spec: PaperValidateSpec) -> dict[str, Any]:
    if spec.variant == "captured_qwen":
        args = [
            "--kv-dir",
            spec.kv_dir,
            "--trials",
            str(spec.trials),
            "--max-layers",
            str(spec.max_layers),
            "--bits",
            spec.bits,
        ]
        output_dir = spec.output_dir or "artifacts/paper_baseline/qwen_captured"
        args.extend(["--output-dir", output_dir])
        if spec.write_config:
            args.append("--write-config")
        if spec.config_out:
            args.extend(["--config-out", spec.config_out])
        return {
            "command": _cli_command("paper_validate_captured_qwen.py", args),
            "expected_outputs": [
                str(Path(output_dir) / "metrics" / "attention_summary_captured.csv"),
                str(Path(output_dir) / "plots" / "attention_tradeoffs_captured.png"),
            ],
        }
    args = [
        "--trials",
        str(spec.trials),
        "--synthetic-layers",
        str(spec.synthetic_layers),
        "--batch",
        str(spec.batch),
        "--heads",
        str(spec.heads),
        "--seq-len",
        str(spec.seq_len),
        "--head-dim",
        str(spec.head_dim),
        "--bits",
        spec.bits,
    ]
    return {
        "command": _cli_command("paper_validate_attention.py", args),
        "expected_outputs": [
            "artifacts/paper_baseline/metrics/attention_summary.csv",
        ],
    }


def build_matrix_preview(spec: MatrixRunSpec) -> dict[str, Any]:
    args = [
        "--kv-dir",
        spec.kv_dir,
        "--rotation-dir",
        spec.rotation_dir,
        "--bits",
        spec.bits,
        "--trials",
        str(spec.trials),
        "--max-layers",
        str(spec.max_layers),
        "--eval-device",
        spec.eval_device,
        "--output-dir",
        spec.output_dir,
        "--ms-regular-bits",
        str(spec.ms_regular_bits),
        "--ms-outlier-bits",
        str(spec.ms_outlier_bits),
        "--ms-outlier-count",
        str(spec.ms_outlier_count),
    ]
    if spec.skip_statistics:
        args.append("--skip-statistics")
    if spec.skip_plots:
        args.append("--skip-plots")
    return {
        "command": _cli_command("validate_qwen_3060_matrix.py", args),
        "expected_outputs": [
            str(Path(spec.output_dir) / "metrics" / "qwen_3060_matrix_summary.csv"),
            str(Path(spec.output_dir) / "qwen_3060_matrix_report.md"),
        ],
    }


def build_export_report_preview(spec: ExportReportSpec) -> dict[str, Any]:
    args: list[str] = []
    if spec.matrix_dir:
        args.extend(["--matrix-dir", spec.matrix_dir])
    return {
        "command": _cli_command("export_report.py", args),
        "expected_outputs": (
            [
                str(Path(spec.matrix_dir) / "plots" / "qwen_3060_matrix_attention.png"),
                str(Path(spec.matrix_dir) / "reports" / "qwen_3060_matrix_summary.md"),
            ]
            if spec.matrix_dir
            else ["artifacts/reports/summary.md"]
        ),
    }


def build_export_online_preview(spec: ExportOnlineReportSpec) -> dict[str, Any]:
    args = [
        "--hf-dir",
        spec.hf_dir,
        "--runtime-dir",
        spec.runtime_dir,
        "--replay-summary-csv",
        spec.replay_summary_csv,
        "--output-dir",
        spec.output_dir,
    ]
    return {
        "command": _cli_command("export_online_eval_report.py", args),
        "expected_outputs": [
            str(Path(spec.output_dir) / "reports" / "online_eval_summary.md"),
            str(Path(spec.output_dir) / "plots" / "online_perplexity.png"),
        ],
    }


def build_package_preview(spec: PackageSpec) -> dict[str, Any]:
    args = [
        "--input-gguf",
        spec.input_gguf,
        "--output-gguf",
        spec.output_gguf,
        "--profiles",
        spec.profiles,
        "--default-profile",
        spec.default_profile,
        "--hypura-compatible-profile",
        spec.hypura_compatible_profile,
        "--bits",
        str(spec.bits),
        "--rotation-dir",
        spec.rotation_dir,
        "--paper-rotation-seed",
        str(spec.paper_rotation_seed),
        "--paper-qjl-seed",
        str(spec.paper_qjl_seed),
    ]
    if spec.force:
        args.append("--force")
    return {
        "command": _cli_command("pack_turboquant_gguf.py", args),
        "expected_outputs": [spec.output_gguf],
    }


def build_runtime_preview(spec: RuntimeEvalSpec) -> dict[str, Any]:
    args = [
        "--mode",
        spec.mode,
        "--model-path",
        spec.model_path,
        "--threads",
        str(spec.threads),
        "--repetitions",
        str(spec.repetitions),
        "--context-size",
        str(spec.context_size),
        "--batch-size",
        str(spec.batch_size),
        "--stride",
        str(spec.stride),
        "--chunks",
        str(spec.chunks),
        "--n-prompt",
        str(spec.n_prompt),
        "--n-gen",
        str(spec.n_gen),
        "--mcq-tasks",
        spec.mcq_tasks,
        "--chat-tasks",
        spec.chat_tasks,
        "--lm-eval-limit",
        str(spec.lm_eval_limit),
        "--output-dir",
        spec.output_dir,
        "--dry-run",
    ]
    optional_pairs = {
        "--perplexity-bin": spec.perplexity_bin,
        "--llama-bench-bin": spec.llama_bench_bin,
        "--server-bin": spec.server_bin,
        "--corpus-file": spec.corpus_file,
        "--server-base-url": spec.server_base_url,
        "--server-log-prefix": spec.server_log_prefix,
        "--tokenizer-path": spec.tokenizer_path,
        "--runtime-profile": spec.runtime_profile,
        "--runtime-env-json": spec.runtime_env_json,
    }
    for flag, value in optional_pairs.items():
        if value is not None:
            args.extend([flag, value])
    args.extend(
        [
            "--server-host",
            spec.server_host,
            "--server-port",
            str(spec.server_port),
            "--server-ready-timeout-sec",
            str(spec.server_ready_timeout_sec),
            "--server-context-size",
            str(spec.server_context_size),
            "--server-n-gpu-layers",
            spec.server_n_gpu_layers,
            "--server-model-name",
            spec.server_model_name,
        ]
    )
    if spec.allow_mcq_unavailable:
        args.append("--allow-mcq-unavailable")
    return {
        "command": _cli_command("eval_runtime_qwen.py", args),
        "expected_outputs": [
            str(Path(spec.output_dir) / "logs" / "runtime_eval_commands.json"),
            str(Path(spec.output_dir) / "metrics" / "runtime_eval_run_meta.json"),
        ],
    }


def build_serve_preview(spec: ServeSpec, *, kind: str) -> dict[str, Any]:
    if kind == "serve-llama":
        if spec.server_bin is None or spec.model_path is None:
            raise ValueError("serve-llama requires server_bin and model_path")
        env_updates = merge_runtime_env_overrides(
            profile_name=spec.runtime_profile,
            runtime_env_json=spec.runtime_env_json,
        )
        command = build_llama_server_command(
            binary_path=Path(spec.server_bin),
            model_path=Path(spec.model_path),
            host=spec.host,
            port=spec.port,
            context_size=spec.context_size,
            threads=spec.threads,
            model_alias=spec.model_alias,
            n_gpu_layers=spec.n_gpu_layers,
        )
        return {"command": command, "env": env_updates}

    if spec.gguf is None:
        raise ValueError("serve-hypura requires gguf")
    bridge = read_hypura_gguf_bridge_config(Path(spec.gguf)) if spec.turboquant_mode == "gguf-auto" else None
    command = build_hypura_serve_command(
        gguf_path=Path(spec.gguf),
        host=spec.host,
        port=spec.port,
        context=spec.context_size,
        turboquant_mode=spec.turboquant_mode,
        release=spec.release,
    )
    payload: dict[str, Any] = {"command": command}
    if bridge is not None:
        payload["bridge"] = asdict(bridge)
    return payload


def run_script_main(module_name: str, args: list[str]) -> dict[str, Any]:
    """Import a script module and execute its `main(argv)` entrypoint."""

    script_path = REPO_ROOT / (module_name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(module_name.replace(".", "_"), script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return_code = module.main(args)
    return {"exit_code": int(return_code)}


def capture_task(spec: CaptureFormState) -> dict[str, Any]:
    preview = build_capture_preview(spec)
    if spec.dry_run:
        return preview
    result = run_script_main("scripts.capture_qwen_kv", preview["command"][4:])
    result.update(preview)
    return result


def paper_validate_task(spec: PaperValidateSpec) -> dict[str, Any]:
    preview = build_paper_preview(spec)
    if spec.dry_run:
        return preview
    module_name = "scripts.paper_validate_captured_qwen" if spec.variant == "captured_qwen" else "scripts.paper_validate_attention"
    result = run_script_main(module_name, preview["command"][4:])
    result.update(preview)
    return result


def matrix_validate_task(spec: MatrixRunSpec) -> dict[str, Any]:
    preview = build_matrix_preview(spec)
    if spec.dry_run:
        return preview
    result = run_script_main("scripts.validate_qwen_3060_matrix", preview["command"][4:])
    result.update(preview)
    return result


def export_report_task(spec: ExportReportSpec) -> dict[str, Any]:
    preview = build_export_report_preview(spec)
    if spec.dry_run:
        return preview
    result = run_script_main("scripts.export_report", preview["command"][4:])
    result.update(preview)
    return result


def export_online_report_task(spec: ExportOnlineReportSpec) -> dict[str, Any]:
    preview = build_export_online_preview(spec)
    if spec.dry_run:
        return preview
    result = run_script_main("scripts.export_online_eval_report", preview["command"][4:])
    result.update(preview)
    return result


def package_gguf_task(spec: PackageSpec) -> dict[str, Any]:
    preview = build_package_preview(spec)
    if spec.dry_run:
        return preview
    result = run_script_main("scripts.pack_turboquant_gguf", preview["command"][4:])
    result.update(preview)
    return result


def runtime_eval_task(spec: RuntimeEvalSpec) -> dict[str, Any]:
    preview = build_runtime_preview(spec)
    args = preview["command"][4:]
    if not spec.dry_run:
        args = [item for item in args if item != "--dry-run"]
    result = run_script_main("scripts.eval_runtime_qwen", args)
    result.update(preview)
    return result


REPO_ROOT = Path(__file__).resolve().parents[2]
