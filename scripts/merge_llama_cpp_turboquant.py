"""Plan and rehearse llama.cpp upstream merges while preserving TurboQuant SO8.

The script is intentionally conservative. By default it fetches the three
authoritative lines, inventories changed files, classifies conflicts that touch
TurboQuant / Triality surfaces, and writes a reproducible JSON report. Optional
merge rehearsal runs in a separate git worktree so the vendored checkout is not
mutated while we inspect the blast radius.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


LLAMA_CPP_DIR = Path("zapabob/llama.cpp")
DEFAULT_REPORT = Path("_docs/llama_cpp_turboquant_merge_plan.json")

REFS = {
    "zapabob": {
        "url": "https://github.com/zapabob/llama.cpp.git",
        "remote_ref": "refs/heads/main",
        "local_ref": "refs/remotes/origin/main",
    },
    "official": {
        "url": "https://github.com/ggml-org/llama.cpp.git",
        "remote_ref": "refs/heads/master",
        "local_ref": "refs/remotes/official/master",
    },
    "thetom": {
        "url": "https://github.com/TheTom/llama-cpp-turboquant.git",
        "remote_ref": "refs/heads/feature/turboquant-kv-cache",
        "local_ref": "refs/remotes/thetom/feature/turboquant-kv-cache",
    },
}

PROTECTED_PATTERNS = (
    "llama-turboquant",
    "turboquant",
    "triality",
    "hypura.turboquant",
    "tq4_1s",
    "tq3_1s",
    "tqcuda",
    "cache_type_k",
    "cache_type_v",
)

API_PATTERNS = (
    "tools/server",
    "common/chat",
    "common/json-schema",
    "common/peg",
    "function-calling",
    "include/llama.h",
    "src/llama-chat",
    "src/llama-context",
    "src/llama-model",
)

CUDA_PATTERNS = (
    "ggml/src/ggml-cuda",
    "fattn",
    "flash",
    "kv-cache",
    "llama-kv-cache",
)

SECURITY_PATTERNS = (
    "security",
    "cve",
    "overflow",
    "heap-buffer",
    "sanitize",
    "crash",
    "oob",
    "rpc",
)

CONFLICT_RESOLUTION_HINTS = {
    ".devops/nix/package.nix": {
        "action": "prefer_theirs",
        "reason": "Official packaging inputs are not part of the Windows TurboQuant runtime surface.",
    },
    "conversion/base.py": {
        "action": "auto_union_converter_base_turboquant",
        "reason": "Keep official converter API changes while preserving TurboQuant metadata and artifact options.",
    },
    "convert_hf_to_gguf.py": {
        "action": "auto_union_convert_hf_cli_turboquant",
        "reason": "Keep official CLI options while preserving TurboQuant triality/vector defaults and metadata writing.",
    },
    "ggml/src/ggml-cuda/fattn-common.cuh": {
        "action": "prefer_theirs_then_review",
        "reason": "Official f16_extra workspace replaces the local HIP pool bypass with the current flash-attention API shape.",
    },
    "ggml/src/ggml-cuda/mmvq.cu": {
        "action": "auto_union_mmvq_rdna_tq4",
        "reason": "Adopt official dispatch behavior while retaining TQ4_1S vector-dot support.",
    },
    "ggml/src/ggml-metal/ggml-metal.metal": {
        "action": "auto_union_metal_turboquant_copy_kernels",
        "reason": "Keep official Metal additions while retaining TQ3_1S/TQ4_1S kernels.",
    },
    "ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl": {
        "action": "auto_union_vulkan_nvfp4_turbo3",
        "reason": "Keep official shader additions while retaining TurboQuant dequant functions.",
    },
    "src/llama-context.cpp": {
        "action": "auto_union_llama_context_turbo_padding",
        "reason": "Track official context API shape while preserving TurboQuant KV cache type handling and FA guardrails.",
    },
    "src/llama-kv-cache.cpp": {
        "action": "auto_union_kv_cache_official_turbo",
        "reason": "Track official KV cache layout while preserving TurboQuant/InnerQ state and zero-padding behavior.",
    },
    "tests/CMakeLists.txt": {
        "action": "auto_union_tests_cmake",
        "reason": "Keep official test-col2im-1d registration and local TurboQuant conditional test exclusions.",
    },
    "tools/perplexity/perplexity.cpp": {
        "action": "prefer_theirs_then_review",
        "reason": "Conflict is outside core runtime; prefer official bugfix shape and re-add only necessary local comments.",
    },
}

THEIRS_RESOLVE_ACTIONS = frozenset({"prefer_theirs", "prefer_theirs_then_review"})
AUTO_RESOLVE_ACTIONS = THEIRS_RESOLVE_ACTIONS | frozenset(
    {
        "auto_union_metal_turboquant_copy_kernels",
        "auto_union_convert_hf_cli_turboquant",
        "auto_union_converter_base_turboquant",
        "auto_union_kv_cache_official_turbo",
        "auto_union_llama_context_turbo_padding",
        "auto_union_tests_cmake",
        "auto_union_vulkan_nvfp4_turbo3",
        "auto_union_mmvq_rdna_tq4",
    }
)


@dataclasses.dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str
    old_path: str | None = None

    @property
    def normalized_paths(self) -> tuple[str, ...]:
        if self.old_path is None:
            return (self.path,)
        return (self.old_path, self.path)


@dataclasses.dataclass(frozen=True)
class MergeLine:
    name: str
    ref: str
    head: str
    merge_base: str
    ahead: int
    behind: int
    changed_files: list[ChangedFile]
    local_changed_file_count: int
    overlapping_files: list[str]
    protected_files: list[str]
    protected_overlaps: list[str]
    api_files: list[str]
    cuda_files: list[str]
    security_commits: list[str]
    recent_commits: list[str]
    recommendation: str


class GitError(RuntimeError):
    pass


def _run_git(repo: Path, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(repo), *args]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        raise GitError(
            "git command failed: "
            + " ".join(command)
            + f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lower()


def _parse_name_status(output: str) -> list[ChangedFile]:
    files: list[ChangedFile] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") or status.startswith("C"):
            if len(parts) != 3:
                raise ValueError(f"Unexpected rename/copy line: {line}")
            files.append(ChangedFile(status=status, old_path=parts[1], path=parts[2]))
            continue
        if len(parts) != 2:
            raise ValueError(f"Unexpected name-status line: {line}")
        files.append(ChangedFile(status=status, path=parts[1]))
    return files


def _path_matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize_path(path)
    return any(pattern.lower() in normalized for pattern in patterns)


def _classify_paths(files: Sequence[ChangedFile], patterns: Iterable[str]) -> list[str]:
    matched: set[str] = set()
    for changed in files:
        for path in changed.normalized_paths:
            if _path_matches(path, patterns):
                matched.add(path)
    return sorted(matched)


def _classify_plain_paths(paths: Iterable[str], patterns: Iterable[str]) -> list[str]:
    return sorted({path for path in paths if _path_matches(path, patterns)})


def _status_counts(status_lines: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in status_lines:
        code = line[:2].strip()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _conflict_categories(path: str) -> list[str]:
    categories: list[str] = []
    if _path_matches(path, PROTECTED_PATTERNS):
        categories.append("protected")
    if _path_matches(path, API_PATTERNS):
        categories.append("api")
    if _path_matches(path, CUDA_PATTERNS):
        categories.append("cuda")
    return categories or ["noncore"]


def _conflict_marker_count(worktree: Path, path: str) -> int:
    conflict_path = worktree / Path(path)
    if not conflict_path.exists():
        return 0
    text = conflict_path.read_text(encoding="utf-8", errors="replace")
    return text.count("<<<<<<< ")


def _conflict_stage_plan(worktree: Path, conflicts: Sequence[str]) -> list[dict[str, object]]:
    conflict_set = set(conflicts)
    stages_by_path: dict[str, dict[str, str]] = {path: {} for path in conflicts}
    ls_files = _run_git(worktree, ["ls-files", "-u"], check=False)
    stage_names = {"1": "base", "2": "ours", "3": "theirs"}
    for line in ls_files.stdout.splitlines():
        if "\t" not in line:
            continue
        metadata, path = line.split("\t", 1)
        if path not in conflict_set:
            continue
        parts = metadata.split()
        if len(parts) != 3:
            continue
        _mode, blob, stage = parts
        stage_name = stage_names.get(stage)
        if stage_name is not None:
            stages_by_path[path][stage_name] = blob

    plan: list[dict[str, object]] = []
    for path in conflicts:
        hint = CONFLICT_RESOLUTION_HINTS.get(
            path,
            {
                "action": "manual_review",
                "reason": "No file-specific rule exists; inspect base/ours/theirs before resolving.",
            },
        )
        plan.append(
            {
                "path": path,
                "categories": _conflict_categories(path),
                "stages": stages_by_path[path],
                "conflict_marker_count": _conflict_marker_count(worktree, path),
                "recommended_action": hint["action"],
                "reason": hint["reason"],
            }
        )
    return plan


def _safe_artifact_name(path: str) -> str:
    safe = path.replace("\\", "/").replace("/", "__")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in safe)


def _blob_text(repo: Path, blob: str) -> str:
    return _run_git(repo, ["show", blob]).stdout


def _write_conflict_artifacts(
    worktree: Path,
    conflict_plan: Sequence[dict[str, object]],
    artifact_root: Path,
) -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    readme_lines = [
        "# Official llama.cpp Merge Conflict Artifacts",
        "",
        "Generated from a no-commit merge rehearsal. Each conflict directory contains",
        "`base.txt`, `ours.txt`, `theirs.txt` where available, plus",
        "`conflicted.txt` with Git conflict markers from the rehearsal worktree.",
        "",
        "| # | Path | Action | Categories | Markers |",
        "|---|---|---|---|---|",
    ]

    for index, item in enumerate(conflict_plan, 1):
        path = str(item["path"])
        conflict_dir = artifact_root / f"{index:02d}__{_safe_artifact_name(path)}"
        conflict_dir.mkdir(parents=True, exist_ok=True)

        stages = item.get("stages", {})
        if not isinstance(stages, dict):
            stages = {}
        written_stages: dict[str, str] = {}
        for stage_name in ("base", "ours", "theirs"):
            blob = stages.get(stage_name)
            if not isinstance(blob, str):
                continue
            stage_path = conflict_dir / f"{stage_name}.txt"
            stage_path.write_text(_blob_text(worktree, blob), encoding="utf-8", newline="\n")
            written_stages[stage_name] = str(stage_path)

        conflicted_source = worktree / Path(path)
        conflicted_path = conflict_dir / "conflicted.txt"
        if conflicted_source.exists():
            conflicted_path.write_text(
                conflicted_source.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
                newline="\n",
            )

        categories = item.get("categories", [])
        if not isinstance(categories, list):
            categories = []
        marker_count = item.get("conflict_marker_count", 0)
        action = str(item.get("recommended_action", "manual_review"))
        readme_lines.append(
            f"| {index} | `{path}` | `{action}` | `{', '.join(map(str, categories))}` | `{marker_count}` |"
        )
        entries.append(
            {
                "path": path,
                "directory": str(conflict_dir),
                "written_stages": written_stages,
                "conflicted": str(conflicted_path) if conflicted_path.exists() else None,
            }
        )

    readme_path = artifact_root / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8", newline="\n")
    return {"root": str(artifact_root), "readme": str(readme_path), "entries": entries}


def _conflicted_paths(status_lines: Iterable[str]) -> list[str]:
    return [
        line[3:]
        for line in status_lines
        if line[:2] in {"UU", "AA", "DD", "AU", "UA", "DU", "UD"}
    ]


def _auto_resolve_safe_conflicts(
    worktree: Path,
    conflict_plan: Sequence[dict[str, object]],
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for item in conflict_plan:
        action = str(item.get("recommended_action", ""))
        if action not in AUTO_RESOLVE_ACTIONS:
            continue
        path = str(item["path"])
        if action in THEIRS_RESOLVE_ACTIONS:
            _run_git(worktree, ["checkout", "--theirs", "--", path])
        elif action == "auto_union_converter_base_turboquant":
            _resolve_converter_base_turboquant_union(worktree, path)
        elif action == "auto_union_convert_hf_cli_turboquant":
            _resolve_convert_hf_cli_turboquant_union(worktree, path)
        elif action == "auto_union_kv_cache_official_turbo":
            _resolve_kv_cache_official_turbo_union(worktree, path)
        elif action == "auto_union_llama_context_turbo_padding":
            _resolve_llama_context_turbo_padding_union(worktree, path)
        elif action == "auto_union_metal_turboquant_copy_kernels":
            _resolve_metal_turboquant_copy_kernels(worktree, path)
        elif action == "auto_union_mmvq_rdna_tq4":
            _resolve_mmvq_rdna_tq4_union(worktree, path)
        elif action == "auto_union_tests_cmake":
            _resolve_tests_cmake_union(worktree, path)
        elif action == "auto_union_vulkan_nvfp4_turbo3":
            _resolve_vulkan_nvfp4_turbo3_union(worktree, path)
        else:
            raise ValueError(f"Unsupported auto-resolve action: {action}")
        _run_git(worktree, ["add", "--", path])
        resolved.append({"path": path, "action": action})
    return resolved


def _single_conflict_parts(text: str, path: str) -> tuple[int, int, str, str]:
    start_marker = "<<<<<<< HEAD\n"
    middle_marker = "=======\n"
    end_marker_prefix = ">>>>>>> "
    if text.count(start_marker) != 1 or text.count(end_marker_prefix) != 1:
        raise ValueError(f"Unexpected conflict marker shape in {path}")

    start = text.index(start_marker)
    middle = text.index(middle_marker, start)
    end = text.index(end_marker_prefix, middle)
    end_line_end = text.index("\n", end) + 1
    ours = text[start + len(start_marker):middle]
    theirs = text[middle + len(middle_marker):end]
    return start, end_line_end, ours, theirs


def _replace_conflicts(
    text: str,
    path: str,
    replacements: Sequence[Callable[[int, str, str], str]],
) -> str:
    start_marker = "<<<<<<< HEAD\n"
    middle_marker = "=======\n"
    end_marker_prefix = ">>>>>>> "
    if text.count(start_marker) != len(replacements) or text.count(end_marker_prefix) != len(replacements):
        raise ValueError(f"Unexpected conflict count in {path}")

    result: list[str] = []
    pos = 0
    for index, replacement_for in enumerate(replacements):
        start = text.index(start_marker, pos)
        middle = text.index(middle_marker, start)
        end = text.index(end_marker_prefix, middle)
        end_line_end = text.index("\n", end) + 1
        ours = text[start + len(start_marker):middle]
        theirs = text[middle + len(middle_marker):end]
        result.append(text[pos:start])
        result.append(replacement_for(index, ours, theirs))
        pos = end_line_end
    result.append(text[pos:])
    return "".join(result)


def _join_blocks(*blocks: str) -> str:
    joined = ""
    for block in blocks:
        if not block:
            continue
        joined += block
        if not joined.endswith("\n"):
            joined += "\n"
    return joined


def _resolve_converter_base_turboquant_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")

    def replacement_for(index: int, ours: str, theirs: str) -> str:
        if index == 0:
            if "turboquant_mode: str | None" not in ours:
                raise ValueError(f"Missing TurboQuant converter fields in {path}")
            if "target_model_dir: Path | None" not in theirs:
                raise ValueError(f"Missing official target_model_dir field in {path}")
            return _join_blocks(theirs, ours)
        if index == 1:
            if "turboquant_mode: str | None = None" not in ours:
                raise ValueError(f"Missing TurboQuant converter constructor args in {path}")
            if "target_model_dir: Path | None = None" not in theirs or "fp8_as_q8: bool = False" not in theirs:
                raise ValueError(f"Missing official converter constructor args in {path}")
            return _join_blocks(theirs.rstrip().removesuffix("):") + ",\n", ours.lstrip())
        raise ValueError(f"Unexpected conflict index {index} in {path}")

    target.write_text(_replace_conflicts(text, path, (replacement_for, replacement_for)), encoding="utf-8", newline="\n")


def _resolve_convert_hf_cli_turboquant_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")

    def replacement_for(index: int, ours: str, theirs: str) -> str:
        if index == 0:
            if '--tq-mode' not in ours or '--tq-rotation-policy' not in ours:
                raise ValueError(f"Missing TurboQuant CLI arguments in {path}")
            if '--fp8-as-q8' not in theirs or '--target-model-dir' not in theirs:
                raise ValueError(f"Missing official CLI arguments in {path}")
            return _join_blocks(theirs, ours)
        if index == 1:
            if "turboquant_mode=args.tq_mode" not in ours:
                raise ValueError(f"Missing TurboQuant model constructor kwargs in {path}")
            if "target_model_dir=Path(args.target_model_dir)" not in theirs or "fp8_as_q8=args.fp8_as_q8" not in theirs:
                raise ValueError(f"Missing official model constructor kwargs in {path}")
            return _join_blocks(theirs, ours)
        raise ValueError(f"Unexpected conflict index {index} in {path}")

    target.write_text(_replace_conflicts(text, path, (replacement_for, replacement_for)), encoding="utf-8", newline="\n")


def _resolve_llama_context_turbo_padding_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")

    def replacement_for(index: int, ours: str, theirs: str) -> str:
        if "model->hparams.n_layer()" not in theirs:
            raise ValueError(f"Missing official hparams n_layer accessor in {path}")
        if index == 0:
            if "k_is_turbo" not in ours or "head_k = ((head_k + 127) / 128) * 128;" not in ours:
                raise ValueError(f"Missing TurboQuant K padding guard in {path}")
            return (
                "        const bool k_is_turbo = (params.type_k == GGML_TYPE_TURBO2_0 ||\n"
                "                                 params.type_k == GGML_TYPE_TURBO3_0 ||\n"
                "                                 params.type_k == GGML_TYPE_TURBO4_0);\n"
                "        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {\n"
                "            uint32_t head_k = model->hparams.n_embd_head_k(il);\n"
                "            // Turbo types zero-pad heads to next multiple of 128 in llama-kv-cache.cpp.\n"
                "            if (k_is_turbo && head_k % 128 != 0) {\n"
                "                head_k = ((head_k + 127) / 128) * 128;\n"
                "            }\n"
                "            if (head_k % blck_size != 0) {\n"
            )
        if index == 1:
            if "v_is_turbo" not in ours or "head_v = ((head_v + 127) / 128) * 128;" not in ours:
                raise ValueError(f"Missing TurboQuant V padding guard in {path}")
            return (
                "        const bool v_is_turbo = (params.type_v == GGML_TYPE_TURBO2_0 ||\n"
                "                                 params.type_v == GGML_TYPE_TURBO3_0 ||\n"
                "                                 params.type_v == GGML_TYPE_TURBO4_0);\n"
                "        const bool is_mla = model->hparams.is_mla();\n"
                "        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {\n"
                "            uint32_t head_v = model->hparams.n_embd_head_v(il);\n"
                "            // Turbo types zero-pad; MLA has no separate V cache (V = view of K).\n"
                "            if (v_is_turbo && !is_mla && head_v % 128 != 0) {\n"
                "                head_v = ((head_v + 127) / 128) * 128;\n"
                "            }\n"
                "            if (head_v % blck_size != 0) {\n"
            )
        raise ValueError(f"Unexpected conflict index {index} in {path}")

    target.write_text(_replace_conflicts(text, path, (replacement_for, replacement_for)), encoding="utf-8", newline="\n")


def _resolve_kv_cache_official_turbo_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")

    def replacement_for(index: int, ours: str, theirs: str) -> str:
        if index == 0:
            if "TURBO_AUTO_ASYMMETRIC" not in ours or "type_k = GGML_TYPE_Q8_0;" not in ours:
                raise ValueError(f"Missing TurboQuant auto-asymmetric block in {path}")
            if "const uint32_t n_layer = hparams.n_layer_all;" not in theirs:
                raise ValueError(f"Missing official n_layer_all initialization in {path}")
            return _join_blocks(ours, theirs)
        if index == 1:
            if "+3 for turbo rotation matrices" not in ours:
                raise ValueError(f"Missing TurboQuant KV tensor overhead comment in {path}")
            if "2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()" not in theirs:
                raise ValueError(f"Missing official KV tensor overhead expression in {path}")
            return (
                "                // +3 for turbo rotation matrices (turbo_rotation + turbo_rotation_inv + turbo_innerq_scale_inv)\n"
                "                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer + 3)*ggml_tensor_overhead()),\n"
            )
        if index == 2:
            if "LLAMA_ATTN_ROT_K_OVERRIDE" not in ours or "attn_rot_k_supported" not in ours:
                raise ValueError(f"Missing TurboQuant attention rotation override block in {path}")
            if "TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]" not in theirs or "LLM_ARCH_DEEPSEEK32" not in theirs:
                raise ValueError(f"Missing official shared-cache or DeepSeek attention rotation block in {path}")
            return (
                "    // TurboQuant keeps attention rotation enabled by default on supported K/V cache sides.\n"
                "    // The official shared-cache path is preserved: views inherit the source cache rotation tensors.\n"
                "    // LLAMA_ATTN_ROT_DISABLE remains a hard lock-out; per-side overrides can opt supported sides in or out.\n"
                "    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]\n"
                "    if (other) {\n"
                "        n_embd_head_k_all = other->n_embd_head_k_all;\n"
                "        n_embd_head_v_all = other->n_embd_head_v_all;\n"
                "\n"
                "        attn_rot_k = other->attn_rot_k;\n"
                "        attn_rot_v = other->attn_rot_v;\n"
                "    } else {\n"
                "        const char * LLAMA_ATTN_ROT_DISABLE = getenv(\"LLAMA_ATTN_ROT_DISABLE\");\n"
                "        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;\n"
                "        if (attn_rot_disable) {\n"
                "            LLAMA_LOG_WARN(\"%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\\n\", __func__);\n"
                "        }\n"
                "\n"
                "        const bool attn_rot_k_supported =\n"
                "            n_embd_head_k_all > 0 &&\n"
                "            ggml_is_quantized(type_k) &&\n"
                "            hparams.n_embd_head_k() % 64 == 0;\n"
                "\n"
                "        const bool attn_rot_k_deepseek_indexer =\n"
                "            model.arch == LLM_ARCH_DEEPSEEK32 &&\n"
                "            hparams.n_embd_head_k_full == hparams.indexer_head_size;\n"
                "\n"
                "        attn_rot_k =\n"
                "            !attn_rot_disable &&\n"
                "            (attn_rot_k_supported || attn_rot_k_deepseek_indexer);\n"
                "\n"
                "        attn_rot_v =\n"
                "            !attn_rot_disable &&\n"
                "            n_embd_head_v_all > 0 &&\n"
                "            ggml_is_quantized(type_v) &&\n"
                "            hparams.n_embd_head_v() % 64 == 0;\n"
                "\n"
                "        const char * ROT_K_OV = getenv(\"LLAMA_ATTN_ROT_K_OVERRIDE\");\n"
                "        if (ROT_K_OV && !attn_rot_disable) {\n"
                "            attn_rot_k = (atoi(ROT_K_OV) != 0) && (attn_rot_k_supported || attn_rot_k_deepseek_indexer);\n"
                "        }\n"
                "\n"
                "        const char * ROT_V_OV = getenv(\"LLAMA_ATTN_ROT_V_OVERRIDE\");\n"
                "        if (ROT_V_OV && !attn_rot_disable) {\n"
                "            attn_rot_v = (atoi(ROT_V_OV) != 0) &&\n"
                "                n_embd_head_v_all > 0 &&\n"
                "                ggml_is_quantized(type_v) &&\n"
                "                hparams.n_embd_head_v() % 64 == 0;\n"
                "        }\n"
                "    }\n"
            )
        raise ValueError(f"Unexpected conflict index {index} in {path}")

    replacements = (replacement_for, replacement_for, replacement_for)
    target.write_text(_replace_conflicts(text, path, replacements), encoding="utf-8", newline="\n")


def _resolve_metal_turboquant_copy_kernels(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    required_ours = (
        'kernel_cpy_tq3_1s_f32',
        'kernel_cpy_tq3_1s_f16',
        'kernel_cpy_tq4_1s_f32',
        'kernel_cpy_tq4_1s_f16',
    )

    def replacement_for(index: int, ours: str, theirs: str) -> str:
        if index == 0:
            for token in required_ours:
                if token not in ours:
                    raise ValueError(f"Missing TurboQuant Metal copy kernel {token} in {path}")
            if "template<typename T>" not in theirs:
                raise ValueError(f"Missing official Metal kernel template continuation in {path}")
            return _join_blocks(ours, theirs)
        if index == 1:
            if "kernel_set_rows_turbo4" not in ours:
                raise ValueError(f"Missing TurboQuant Metal set_rows instantiation in {path}")
            return _join_blocks(ours, theirs)
        raise ValueError(f"Unexpected conflict index {index} in {path}")

    target.write_text(
        _replace_conflicts(text, path, (replacement_for, replacement_for)),
        encoding="utf-8",
        newline="\n",
    )


def _resolve_mmvq_rdna_tq4_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    start, end_line_end, ours, theirs = _single_conflict_parts(text, path)
    if "case GGML_TYPE_TQ4_1S:" not in ours or "case GGML_TYPE_Q4_K:" not in ours:
        raise ValueError(f"Missing local RDNA TQ4/Q4_K cases in {path}")
    if "return 8;" not in theirs:
        raise ValueError(f"Missing official RDNA return value in {path}")

    replacement = (
        "                case GGML_TYPE_TQ4_1S:\n"
        "                case GGML_TYPE_Q4_K:\n"
        "                    return 8;\n"
    )
    target.write_text(text[:start] + replacement + text[end_line_end:], encoding="utf-8", newline="\n")


def _resolve_vulkan_nvfp4_turbo3_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    start, end_line_end, ours, theirs = _single_conflict_parts(text, path)
    if "#elif defined(DATA_A_TURBO3_0)" not in ours:
        raise ValueError(f"Missing TurboQuant Vulkan DATA_A_TURBO3_0 branch in {path}")
    if "#define dequantFuncA dequantFuncTURBO3_0" not in ours:
        raise ValueError(f"Missing TurboQuant Vulkan dequantFuncA mapping in {path}")
    if "#define dequantFuncA_v dequantFuncNVFP4_v" not in theirs:
        raise ValueError(f"Missing official Vulkan NVFP4 vector mapping in {path}")

    replacement = theirs
    if not replacement.endswith("\n"):
        replacement += "\n"
    replacement += ours
    target.write_text(text[:start] + replacement + text[end_line_end:], encoding="utf-8", newline="\n")


def _resolve_tests_cmake_union(worktree: Path, path: str) -> None:
    target = worktree / Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    start, end_line_end, ours, theirs = _single_conflict_parts(text, path)
    if "test-turboquant" not in ours:
        raise ValueError(f"Missing TurboQuant test block in {path}")
    if "test-col2im-1d.cpp" not in theirs:
        raise ValueError(f"Missing official col2im test in {path}")

    replacement = ours
    if not replacement.endswith("\n"):
        replacement += "\n"
    if "test-col2im-1d.cpp" not in replacement:
        replacement += "    llama_build_and_test(test-col2im-1d.cpp)\n"
    target.write_text(text[:start] + replacement + text[end_line_end:], encoding="utf-8", newline="\n")


def _fetch_refs(repo: Path) -> None:
    for spec in REFS.values():
        _run_git(
            repo,
            [
                "fetch",
                "--no-tags",
                spec["url"],
                f"+{spec['remote_ref']}:{spec['local_ref']}",
            ],
        )


def _current_head(repo: Path) -> str:
    return _run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()


def _rev_parse(repo: Path, ref: str) -> str:
    return _run_git(repo, ["rev-parse", ref]).stdout.strip()


def _rev_counts(repo: Path, left: str, right: str) -> tuple[int, int]:
    output = _run_git(repo, ["rev-list", "--left-right", "--count", f"{left}...{right}"]).stdout.strip()
    left_count, right_count = output.split()
    return int(left_count), int(right_count)


def _recent_commits(repo: Path, ref: str, limit: int) -> list[str]:
    output = _run_git(repo, ["log", "--oneline", f"--max-count={limit}", ref]).stdout
    return [line for line in output.splitlines() if line.strip()]


def _security_commits(repo: Path, base: str, ref: str, limit: int) -> list[str]:
    grep_args: list[str] = []
    for pattern in SECURITY_PATTERNS:
        grep_args.extend(["--grep", pattern])
    output = _run_git(
        repo,
        [
            "log",
            "--oneline",
            "--regexp-ignore-case",
            "--extended-regexp",
            *grep_args,
            f"--max-count={limit}",
            f"{base}..{ref}",
        ],
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def _recommendation_for(name: str, line: MergeLine) -> str:
    if name == "zapabob" and line.behind > 0 and not line.protected_overlaps:
        return "fast_forward_submodule_pointer"
    if name == "zapabob" and line.behind > 0:
        return "merge_zapabob_first_then_run_contract_tests"
    if name == "official":
        return "manual_overlay_required_preserve_so8_triality_then_api_followup"
    if name == "thetom" and line.protected_overlaps:
        return "selective_turboquant_overlay_required_preserve_so8_triality_api"
    if line.behind > 0:
        return "merge_rehearsal_before_apply"
    return "already_contains_line_or_no_action"


def _build_line(repo: Path, name: str, base_ref: str, ref: str, limit: int) -> MergeLine:
    ref_head = _rev_parse(repo, ref)
    merge_base = _run_git(repo, ["merge-base", base_ref, ref]).stdout.strip()
    ahead, behind = _rev_counts(repo, base_ref, ref)
    diff_output = _run_git(repo, ["diff", "--name-status", f"{merge_base}..{ref}"]).stdout
    local_diff_output = _run_git(repo, ["diff", "--name-status", f"{merge_base}..{base_ref}"]).stdout
    changed_files = _parse_name_status(diff_output)
    local_changed_files = _parse_name_status(local_diff_output)
    upstream_paths = {path for changed in changed_files for path in changed.normalized_paths}
    local_paths = {path for changed in local_changed_files for path in changed.normalized_paths}
    overlapping_files = sorted(upstream_paths & local_paths)
    partial = MergeLine(
        name=name,
        ref=ref,
        head=ref_head,
        merge_base=merge_base,
        ahead=ahead,
        behind=behind,
        changed_files=changed_files,
        local_changed_file_count=len(local_changed_files),
        overlapping_files=overlapping_files,
        protected_files=_classify_paths(changed_files, PROTECTED_PATTERNS),
        protected_overlaps=[
            path for path in overlapping_files if _path_matches(path, PROTECTED_PATTERNS)
        ],
        api_files=_classify_paths(changed_files, API_PATTERNS),
        cuda_files=_classify_paths(changed_files, CUDA_PATTERNS),
        security_commits=_security_commits(repo, merge_base, ref, limit),
        recent_commits=_recent_commits(repo, ref, limit),
        recommendation="",
    )
    return dataclasses.replace(partial, recommendation=_recommendation_for(name, partial))


def _build_report(repo: Path, fetch: bool, limit: int) -> dict[str, object]:
    if fetch:
        _fetch_refs(repo)

    head = _current_head(repo)
    lines = [
        _build_line(repo, name, "HEAD", spec["local_ref"], limit)
        for name, spec in REFS.items()
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "head": head,
        "lines": [_line_to_json(line) for line in lines],
        "policy": {
            "base_rule": "Prefer latest official behavior when equivalent, then overlay SO8/Triality advantages.",
            "protected_rule": "Do not delete llama-turboquant, hypura.turboquant metadata, or Triality vector defaults.",
            "api_rule": "Track official server/tool-calling/web-search-compatible API surfaces explicitly.",
            "rtx5060ti_rule": "Verify GPU memory before claiming a CUDA build; current target is a 16 GB RTX 5060 Ti.",
        },
    }


def _line_to_json(line: MergeLine) -> dict[str, object]:
    return {
        "name": line.name,
        "ref": line.ref,
        "head": line.head,
        "merge_base": line.merge_base,
        "ahead": line.ahead,
        "behind": line.behind,
        "changed_file_count": len(line.changed_files),
        "local_changed_file_count": line.local_changed_file_count,
        "overlapping_files": line.overlapping_files,
        "protected_files": line.protected_files,
        "protected_overlaps": line.protected_overlaps,
        "api_files": line.api_files,
        "cuda_files": line.cuda_files,
        "security_commits": line.security_commits,
        "recent_commits": line.recent_commits,
        "recommendation": line.recommendation,
    }


def _write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_clean_submodule(repo: Path) -> None:
    status = _run_git(repo, ["status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    if status:
        raise GitError(f"Refusing to mutate dirty submodule {repo}:\n{status}")


def _apply_zapabob(repo: Path) -> str:
    _ensure_clean_submodule(repo)
    _run_git(repo, ["checkout", "refs/remotes/origin/main"])
    return _current_head(repo)


def _rehearse_merge(
    repo: Path,
    ref: str,
    keep_worktree: bool,
    conflict_artifacts_dir: Path | None,
    auto_resolve_safe_conflicts: bool,
) -> dict[str, object]:
    temp_root = Path(tempfile.mkdtemp(prefix="llama-tq-merge-"))
    worktree = temp_root / "worktree"
    try:
        _run_git(repo, ["worktree", "add", "--detach", str(worktree), "HEAD"])
        merge = _run_git(worktree, ["merge", "--no-commit", "--no-ff", ref], check=False)
        status = _run_git(worktree, ["status", "--porcelain=v1", "--untracked-files=no"], check=False)
        status_lines = status.stdout.splitlines()
        conflicts = _conflicted_paths(status_lines)
        protected_conflicts = _classify_plain_paths(conflicts, PROTECTED_PATTERNS)
        api_conflicts = _classify_plain_paths(conflicts, API_PATTERNS)
        cuda_conflicts = _classify_plain_paths(conflicts, CUDA_PATTERNS)
        classified_conflicts = set(protected_conflicts) | set(api_conflicts) | set(cuda_conflicts)
        conflict_plan = _conflict_stage_plan(worktree, conflicts)
        result: dict[str, object] = {
            "worktree": str(worktree),
            "ref": ref,
            "returncode": merge.returncode,
            "stdout_tail": merge.stdout[-4000:],
            "stderr_tail": merge.stderr[-4000:],
            "conflicts": conflicts,
            "conflict_summary": {
                "count": len(conflicts),
                "protected": protected_conflicts,
                "api": api_conflicts,
                "cuda": cuda_conflicts,
                "noncore": sorted(set(conflicts) - classified_conflicts),
            },
            "conflict_plan": conflict_plan,
            "status_counts": _status_counts(status_lines),
            "status": status_lines,
        }
        if conflict_artifacts_dir is not None:
            result["conflict_artifacts"] = _write_conflict_artifacts(
                worktree,
                conflict_plan,
                conflict_artifacts_dir,
            )
        if auto_resolve_safe_conflicts:
            resolved = _auto_resolve_safe_conflicts(worktree, conflict_plan)
            post_status = _run_git(
                worktree,
                ["status", "--porcelain=v1", "--untracked-files=no"],
                check=False,
            )
            post_status_lines = post_status.stdout.splitlines()
            remaining_conflicts = _conflicted_paths(post_status_lines)
            result["auto_resolve"] = {
                "enabled": True,
                "resolved": resolved,
                "remaining_conflicts": remaining_conflicts,
                "remaining_count": len(remaining_conflicts),
                "status_counts": _status_counts(post_status_lines),
                "status": post_status_lines,
            }
        if keep_worktree:
            result["kept"] = True
            return result
        return result
    finally:
        if not keep_worktree:
            _run_git(repo, ["worktree", "remove", "--force", str(worktree)], check=False)
            shutil.rmtree(temp_root, ignore_errors=True)


def _apply_merge(
    repo: Path,
    ref: str,
    conflict_artifacts_dir: Path | None,
    auto_resolve_safe_conflicts: bool,
) -> dict[str, object]:
    _ensure_clean_submodule(repo)
    merge = _run_git(repo, ["merge", "--no-commit", "--no-ff", ref], check=False)
    status = _run_git(repo, ["status", "--porcelain=v1", "--untracked-files=no"], check=False)
    status_lines = status.stdout.splitlines()
    conflicts = _conflicted_paths(status_lines)
    protected_conflicts = _classify_plain_paths(conflicts, PROTECTED_PATTERNS)
    api_conflicts = _classify_plain_paths(conflicts, API_PATTERNS)
    cuda_conflicts = _classify_plain_paths(conflicts, CUDA_PATTERNS)
    classified_conflicts = set(protected_conflicts) | set(api_conflicts) | set(cuda_conflicts)
    conflict_plan = _conflict_stage_plan(repo, conflicts)
    result: dict[str, object] = {
        "repo": str(repo),
        "ref": ref,
        "returncode": merge.returncode,
        "stdout_tail": merge.stdout[-4000:],
        "stderr_tail": merge.stderr[-4000:],
        "conflicts": conflicts,
        "conflict_summary": {
            "count": len(conflicts),
            "protected": protected_conflicts,
            "api": api_conflicts,
            "cuda": cuda_conflicts,
            "noncore": sorted(set(conflicts) - classified_conflicts),
        },
        "conflict_plan": conflict_plan,
        "status_counts": _status_counts(status_lines),
        "status": status_lines,
        "commit_required": True,
    }
    if conflict_artifacts_dir is not None and conflict_plan:
        result["conflict_artifacts"] = _write_conflict_artifacts(repo, conflict_plan, conflict_artifacts_dir)
    if auto_resolve_safe_conflicts:
        resolved = _auto_resolve_safe_conflicts(repo, conflict_plan)
        post_status = _run_git(repo, ["status", "--porcelain=v1", "--untracked-files=no"], check=False)
        post_status_lines = post_status.stdout.splitlines()
        remaining_conflicts = _conflicted_paths(post_status_lines)
        result["auto_resolve"] = {
            "enabled": True,
            "resolved": resolved,
            "remaining_conflicts": remaining_conflicts,
            "remaining_count": len(remaining_conflicts),
            "status_counts": _status_counts(post_status_lines),
            "status": post_status_lines,
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=LLAMA_CPP_DIR, help="Vendored llama.cpp checkout.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report path.")
    parser.add_argument("--no-fetch", action="store_true", help="Use existing refs without fetching.")
    parser.add_argument("--limit", type=int, default=20, help="Recent/security commit limit per line.")
    parser.add_argument("--apply-zapabob", action="store_true", help="Advance submodule to zapabob/main if clean.")
    parser.add_argument(
        "--rehearse-merge",
        choices=sorted(REFS),
        help="Run merge rehearsal for a line in a temporary worktree.",
    )
    parser.add_argument(
        "--apply-merge",
        choices=sorted(REFS),
        help="Apply a merge to the vendored checkout with --no-commit --no-ff.",
    )
    parser.add_argument("--keep-worktree", action="store_true", help="Keep rehearsal worktree for inspection.")
    parser.add_argument(
        "--conflict-artifacts-dir",
        type=Path,
        help="Write base/ours/theirs conflict artifacts from a merge rehearsal.",
    )
    parser.add_argument(
        "--auto-resolve-safe-conflicts",
        action="store_true",
        help="In rehearsal, resolve policy-marked official-preferred conflicts and report what remains.",
    )
    return parser.parse_args()


def main() -> int:
    """Run merge inventory and optional safe actions."""

    args = _parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        print(f"Missing llama.cpp checkout: {repo}", file=sys.stderr)
        return 2

    report = _build_report(repo, fetch=not args.no_fetch, limit=args.limit)

    if args.apply_zapabob:
        report["applied_zapabob_head"] = _apply_zapabob(repo)

    if args.rehearse_merge:
        ref = REFS[args.rehearse_merge]["local_ref"]
        report["merge_rehearsal"] = _rehearse_merge(
            repo,
            ref,
            keep_worktree=args.keep_worktree,
            conflict_artifacts_dir=args.conflict_artifacts_dir,
            auto_resolve_safe_conflicts=args.auto_resolve_safe_conflicts,
        )

    if args.apply_merge:
        ref = REFS[args.apply_merge]["local_ref"]
        report["applied_merge"] = _apply_merge(
            repo,
            ref,
            conflict_artifacts_dir=args.conflict_artifacts_dir,
            auto_resolve_safe_conflicts=args.auto_resolve_safe_conflicts,
        )

    _write_report(report, args.report)
    print(f"Wrote {args.report}")
    lines = report.get("lines", [])
    if not isinstance(lines, list):
        lines = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        print(
            f"{line['name']}: behind={line['behind']} protected={len(line['protected_files'])} "
            f"api={len(line['api_files'])} cuda={len(line['cuda_files'])} -> {line['recommendation']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
