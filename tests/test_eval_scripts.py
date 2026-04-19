from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_hf_online_qwen.py"
RUNTIME_SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_runtime_qwen.py"
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_online_eval_report.py"
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_zapabob_runtime.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_hf_online_qwen_dry_run_writes_metadata(tmp_path: Path) -> None:
    module = _load_module(HF_SCRIPT_PATH, "eval_hf_online_qwen")
    output_dir = tmp_path / "hf"
    argv = [
        "eval_hf_online_qwen.py",
        "--mode",
        "key_only_random",
        "--bits",
        "3",
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    run_meta = json.loads((output_dir / "metrics" / "hf_online_run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["mode"] == "key_only_random"
    assert run_meta["weight_load"] == "4bit"
    assert run_meta["dry_run"] is True


def test_eval_runtime_qwen_dry_run_writes_commands(tmp_path: Path) -> None:
    module = _load_module(RUNTIME_SCRIPT_PATH, "eval_runtime_qwen")
    output_dir = tmp_path / "runtime"
    argv = [
        "eval_runtime_qwen.py",
        "--mode",
        "turboquant_enabled_audit",
        "--model-path",
        "models/qwen.gguf",
        "--llama-bench-bin",
        "zapabob/llama.cpp/build/bin/llama-bench.exe",
        "--server-bin",
        "zapabob/llama.cpp/build/bin/llama-server.exe",
        "--tokenizer-path",
        "models/qwen-tokenizer",
        "--runtime-profile",
        "turboquant_enabled_audit",
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    commands = json.loads((output_dir / "logs" / "runtime_eval_commands.json").read_text(encoding="utf-8"))
    assert "llama_bench" in commands
    assert commands["llama_bench"]["env"]["LLAMA_TURBOQUANT"] == "1"
    assert "server_start" in commands
    assert commands["lm_eval_mcq"]["command"][0].endswith("python.exe")
    assert "base_url=http://127.0.0.1:8080/v1/completions" in ",".join(commands["lm_eval_mcq"]["command"])
    assert "tokenizer=models/qwen-tokenizer" in ",".join(commands["lm_eval_mcq"]["command"])
    assert "--apply_chat_template" in commands["lm_eval_chat"]["command"]


def test_audit_zapabob_runtime_script_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_module(AUDIT_SCRIPT_PATH, "audit_zapabob_runtime")
    llama_cpp_dir = tmp_path / "llama.cpp-zapabob"
    (llama_cpp_dir / "src").mkdir(parents=True)
    (llama_cpp_dir / "tools" / "turboquant").mkdir(parents=True)
    (llama_cpp_dir / "src" / "llama-turboquant.cpp").write_text(
        'cfg.so8_enabled = env_flag("LLAMA_TURBOQUANT_SO8", true);\n',
        encoding="utf-8",
    )
    (llama_cpp_dir / "src" / "llama-kv-cache.cpp").write_text(
        "\n".join(
            [
                'if (turboquant_cfg.enabled) {',
                '    LLAMA_LOG_INFO("%s: TurboQuant enabled (so8=%d, so8_learned=%d, triality=%d, mix=%.3f, seed=%u)\\n", __func__);',
                '}',
                'if (turboquant_cfg.enabled) { LLAMA_LOG_INFO("%s: TurboQuant K-path active at layer %d\\n", __func__, il); }',
                'if (turboquant_cfg.enabled) { LLAMA_LOG_INFO("%s: TurboQuant V-path active at layer %d\\n", __func__, il); }',
            ]
        ),
        encoding="utf-8",
    )
    (llama_cpp_dir / "tools" / "turboquant" / "turboquant.cpp").write_text(
        "if (!llama_turboquant_load_artifact(path, artifact, &err)) { return 1; }\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    argv = [
        "audit_zapabob_runtime.py",
        "--llama-cpp-dir",
        str(llama_cpp_dir),
        "--output-dir",
        str(output_dir),
        "--server-bin",
        "C:/tmp/llama-server.exe",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    assert (output_dir / "runtime_code_audit.json").exists()
    assert (output_dir / "runtime_code_audit.md").exists()


def test_export_online_eval_report_writes_markdown_and_plots(tmp_path: Path) -> None:
    module = _load_module(EXPORT_SCRIPT_PATH, "export_online_eval_report")
    replay_csv = tmp_path / "replay.csv"
    replay_csv.write_text(
        "\n".join(
            [
                "mode,bit_setting,metric,mean,std,sem,ci95_low,ci95_high",
                "key_only_random,4,memory_ratio_vs_exact,0.63,0,0,0.63,0.63",
                "key_only_random,4,hidden_cosine_similarity,0.999,0.001,0.0005,0.998,1.0",
                "key_only_random,4,logit_cosine_similarity,0.998,0.001,0.0005,0.997,0.999",
            ]
        ),
        encoding="utf-8",
    )
    hf_metrics = tmp_path / "hf" / "metrics"
    hf_metrics.mkdir(parents=True)
    (hf_metrics / "hf_online_ppl_summary.csv").write_text(
        "\n".join(
            [
                "backend,mode,metric,n,mean,std,sem,ci95_low,ci95_high",
                "hf,exact,perplexity,4,9.5,0.5,0.25,9.0,10.0",
                "hf,key_only_random,perplexity,4,10.0,1.0,0.5,9.0,11.0",
            ]
        ),
        encoding="utf-8",
    )
    (hf_metrics / "hf_online_ppl_chunks.csv").write_text(
        "\n".join(
            [
                "backend,mode,chunk_id,token_count,log_perplexity,perplexity",
                "hf,exact,0,128,2.197225,9.0",
                "hf,exact,1,128,2.251292,9.5",
                "hf,exact,2,128,2.302585,10.0",
                "hf,key_only_random,0,128,2.302585,10.0",
                "hf,key_only_random,1,128,2.351375,10.5",
                "hf,key_only_random,2,128,2.397895,11.0",
            ]
        ),
        encoding="utf-8",
    )
    (hf_metrics / "hf_online_benchmark_summary.csv").write_text(
        "\n".join(
            [
                "backend,mode,task,n,n_correct,mean,std,sem,ci95_low,ci95_high",
                "hf,key_only_random,piqa,20,15,0.75,0.43,0.096,0.53,0.89",
            ]
        ),
        encoding="utf-8",
    )
    runtime_metrics = tmp_path / "runtime" / "metrics"
    runtime_metrics.mkdir(parents=True)
    (runtime_metrics / "runtime_bench_samples.csv").write_text(
        "\n".join(
            [
                "backend,mode,run_id,sample_group,sample_idx,test,n_prompt,n_gen,metric,value",
                "runtime,key_only_random,bench,0,0,pp256,256,0,tokens_per_second,1234.0",
                "runtime,key_only_random,bench,0,1,pp256,256,0,tokens_per_second,1250.0",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"
    argv = [
        "export_online_eval_report.py",
        "--hf-dir",
        str(tmp_path / "hf"),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--replay-summary-csv",
        str(replay_csv),
        "--output-dir",
        str(output_dir),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    assert (output_dir / "reports" / "online_eval_summary.md").exists()
    assert (output_dir / "plots" / "online_perplexity.png").exists()
    assert (output_dir / "metrics" / "online_eval_ppl_pairwise.csv").exists()


def test_export_online_eval_report_supports_nested_runtime_mode_dirs(tmp_path: Path) -> None:
    module = _load_module(EXPORT_SCRIPT_PATH, "export_online_eval_report_nested")
    runtime_metrics = tmp_path / "runtime" / "current_main" / "exact" / "metrics"
    runtime_metrics.mkdir(parents=True)
    (runtime_metrics / "runtime_ppl_chunks.csv").write_text(
        "\n".join(
            [
                "backend,mode,run_id,chunk_id,perplexity,log_perplexity",
                "runtime,exact,ppl,0,9.0,2.197225",
                "runtime,exact,ppl,1,9.5,2.251292",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report_nested"
    argv = [
        "export_online_eval_report.py",
        "--runtime-dir",
        str(tmp_path / "runtime" / "current_main"),
        "--output-dir",
        str(output_dir),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old

    assert code == 0
    report_text = (output_dir / "reports" / "online_eval_summary.md").read_text(encoding="utf-8")
    assert "Pairwise not available for perplexity" in report_text
