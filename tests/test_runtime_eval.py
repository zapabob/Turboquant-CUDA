from __future__ import annotations

import json
from pathlib import Path
import sys

from turboquant.runtime_eval import (
    audit_zapabob_runtime_checkout,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
    build_lm_eval_command,
    load_lm_eval_sample_results,
    load_lm_eval_results,
    merge_runtime_env_overrides,
    parse_llama_bench_json,
    parse_perplexity_output,
    render_runtime_audit_markdown,
    resolve_runtime_profile_env,
    resolve_lm_eval_results_path,
    resolve_lm_eval_sample_paths,
)


def test_build_llama_perplexity_command_includes_expected_flags() -> None:
    command = build_llama_perplexity_command(
        binary_path=Path("vendor/llama.cpp/build/bin/llama-perplexity.exe"),
        model_path=Path("models/qwen.gguf"),
        corpus_path=Path("data/wiki.test.raw"),
        context_size=512,
        batch_size=128,
        stride=256,
        chunks=4,
    )

    assert "--ctx-size" in command
    assert "--batch-size" in command
    assert "--ppl-stride" in command
    assert "--chunks" in command


def test_build_llama_bench_command_requests_json_output() -> None:
    command = build_llama_bench_command(
        binary_path=Path("vendor/llama.cpp/build/bin/llama-bench.exe"),
        model_path=Path("models/qwen.gguf"),
        repetitions=3,
        threads=4,
        n_prompt=256,
        n_gen=64,
        device="cuda",
    )

    assert command[-2:] == ["--output", "json"]
    assert "--device" in command


def test_build_llama_server_command_includes_alias_and_gpu_layers() -> None:
    command = build_llama_server_command(
        binary_path=Path("vendor/llama.cpp/build/bin/llama-server.exe"),
        model_path=Path("models/qwen.gguf"),
        host="127.0.0.1",
        port=8080,
        context_size=4096,
        threads=4,
        model_alias="qwen-runtime",
        n_gpu_layers=99,
    )

    assert "--alias" in command
    assert "--n-gpu-layers" in command
    assert "--ctx-size" in command


def test_build_lm_eval_command_supports_local_completion_backends() -> None:
    command = build_lm_eval_command(
        model_type="local-completions",
        model_args={"base_url": "http://127.0.0.1:8080/v1/completions", "model": "qwen"},
        tasks=["hellaswag", "piqa"],
        output_path=Path("artifacts/runtime_eval/lm_eval.json"),
        batch_size=1,
        limit=10,
        extra_flags=["--apply_chat_template"],
    )

    assert command[0:4] == [sys.executable, "-m", "lm_eval", "--model"]
    assert "local-completions" in command
    assert "--limit" in command
    assert "--apply_chat_template" in command


def test_runtime_profile_and_json_env_overrides_merge_cleanly(tmp_path: Path) -> None:
    override_path = tmp_path / "env.json"
    override_path.write_text('{"LLAMA_TURBOQUANT_ROTATION_SEED": 7}', encoding="utf-8-sig")

    profile = resolve_runtime_profile_env("turboquant_enabled_audit")
    merged = merge_runtime_env_overrides(
        profile_name="turboquant_enabled_audit",
        runtime_env_json=str(override_path),
    )

    assert profile["LLAMA_TURBOQUANT"] == "1"
    assert merged["LLAMA_TURBOQUANT"] == "1"
    assert merged["LLAMA_TURBOQUANT_ROTATION_SEED"] == "7"


def test_parse_llama_bench_json_expands_samples() -> None:
    text = json.dumps(
        [
            {
                "test": "pp256",
                "n_prompt": 256,
                "n_gen": 0,
                "samples_ts": [1234.5, 1250.0, 1241.0],
            }
        ]
    )

    frame = parse_llama_bench_json(text, mode="exact", backend="runtime", run_id="bench-1")

    assert len(frame) == 3
    assert set(frame["metric"]) == {"tokens_per_second"}
    assert float(frame["value"].iloc[0]) == 1234.5


def test_parse_perplexity_output_extracts_chunk_and_final_rows() -> None:
    text = "\n".join(
        [
            "       0  13.5106  2.6031  0.0132",
            "     256  13.4200  2.5964  0.0129",
            "Final estimate: PPL = 13.4650 +/- 0.12000",
        ]
    )

    chunks, summary = parse_perplexity_output(text, mode="exact", backend="runtime", run_id="ppl-1")

    assert len(chunks) == 2
    assert float(chunks["perplexity"].iloc[0]) == 13.5106
    assert "log_perplexity" in chunks.columns
    assert float(summary["mean"].iloc[0]) == 13.4650


def test_parse_perplexity_output_supports_multiple_bracket_chunks_on_one_line() -> None:
    text = "\n".join(
        [
            "0.03 minutes",
            "[1]3.9596,[2]5.2452,",
        ]
    )

    chunks, summary = parse_perplexity_output(text, mode="exact", backend="runtime", run_id="ppl-inline")

    assert list(chunks["chunk_id"]) == [1, 2]
    assert list(chunks["perplexity"]) == [3.9596, 5.2452]
    assert len(summary) == 1
    assert float(summary["mean"].iloc[0]) == (3.9596 + 5.2452) / 2


def test_load_lm_eval_results_flattens_metric_rows(tmp_path: Path) -> None:
    path = tmp_path / "lm_eval.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "hellaswag": {"acc,none": 0.74, "acc_stderr,none": 0.01},
                    "gsm8k": {"exact_match,strict-match": 0.42},
                }
            }
        ),
        encoding="utf-8",
    )

    frame = load_lm_eval_results(path)
    assert set(frame["task"]) == {"hellaswag", "gsm8k"}
    assert len(frame) == 2
    assert "sem" in frame.columns


def test_resolve_lm_eval_timestamped_paths_and_parse_samples(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "lm_eval_mcq_2026-04-17T20-18-28.440238.json"
    aggregate_path.write_text("{}", encoding="utf-8")
    sample_path = tmp_path / "samples_hellaswag_2026-04-17T20-18-28.440238.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "doc_id": 0,
                "target": "3",
                "filtered_resps": [["-0.4", "False"], ["-0.5", "False"], ["-0.9", "False"], ["-0.1", "False"]],
                "acc": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_lm_eval_results_path(tmp_path / "lm_eval_mcq.json")
    assert resolved == aggregate_path

    sample_paths = resolve_lm_eval_sample_paths(output_stub=tmp_path / "lm_eval_mcq.json", task_names=["hellaswag"])
    assert sample_paths == [sample_path]

    frame = load_lm_eval_sample_results(
        sample_paths,
        mode="exact",
        backend="runtime",
        run_id="lm_eval_mcq",
        preferred_metrics=["acc"],
    )
    assert len(frame) == 1
    assert frame["task"].iloc[0] == "hellaswag"
    assert frame["prediction"].iloc[0] == "3"
    assert int(frame["is_correct"].iloc[0]) == 1


def test_audit_zapabob_runtime_checkout_marks_parse_log_only_fixture(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    tools_dir = tmp_path / "tools" / "turboquant"
    src_dir.mkdir(parents=True)
    tools_dir.mkdir(parents=True)
    (src_dir / "llama-turboquant.cpp").write_text(
        "\n".join(
            [
                'cfg.so8_enabled = env_flag("LLAMA_TURBOQUANT_SO8", true);',
                'cfg.so8_learned = env_flag("LLAMA_TURBOQUANT_SO8_LEARNED", false);',
                'cfg.triality_enabled = env_flag("LLAMA_TURBOQUANT_TRIALITY", true);',
                'cfg.triality_mix = env_float("LLAMA_TURBOQUANT_TRIALITY_MIX", 0.5f);',
                'cfg.rotation_seed = env_u32("LLAMA_TURBOQUANT_ROTATION_SEED", 0);',
            ]
        ),
        encoding="utf-8",
    )
    (src_dir / "llama-kv-cache.cpp").write_text(
        "\n".join(
            [
                'if (turboquant_cfg.enabled) {',
                '    LLAMA_LOG_INFO("%s: TurboQuant enabled (so8=%d, so8_learned=%d, triality=%d, mix=%.3f, seed=%u)\\n", __func__);',
                '    turboquant_cfg.so8_enabled ? 1 : 0;',
                '    turboquant_cfg.so8_learned ? 1 : 0;',
                '    turboquant_cfg.triality_enabled ? 1 : 0;',
                '    turboquant_cfg.triality_mix;',
                '    turboquant_cfg.rotation_seed;',
                '}',
                'if (turboquant_cfg.enabled) { LLAMA_LOG_INFO("%s: TurboQuant K-path active at layer %d\\n", __func__, il); }',
                'if (turboquant_cfg.enabled) { LLAMA_LOG_INFO("%s: TurboQuant V-path active at layer %d\\n", __func__, il); }',
            ]
        ),
        encoding="utf-8",
    )
    (tools_dir / "turboquant.cpp").write_text(
        "if (!llama_turboquant_load_artifact(path, artifact, &err)) { return 1; }\n",
        encoding="utf-8",
    )

    audit = audit_zapabob_runtime_checkout(tmp_path)
    markdown = render_runtime_audit_markdown(audit)

    assert audit["findings"]["so8_triality_parse_log_only"] is True
    assert audit["findings"]["llama_turboquant_enabled_is_log_only_gate"] is True
    assert audit["findings"]["artifact_loader_only_used_by_tooling"] is True
    assert "Current-main Runtime Audit" in markdown
