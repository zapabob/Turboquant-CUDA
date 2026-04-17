"""Pydantic models shared by the TurboQuant Studio API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from turboquant.capture import DEFAULT_PROMPT_PANEL
from turboquant.runtime import DEFAULT_CAPTURE_MODEL_PRESET, REQUIRED_CUDA, model_preset_names
from turboquant.runtime_eval import CHAT_RUNTIME_TASKS, CURRENT_MAIN_RUNTIME_PROFILES, MCQ_RUNTIME_TASKS


class PromptPanelEntry(BaseModel):
    """Prompt metadata surfaced to the Studio shell."""

    label: str
    prompt: str


class ModelPresetInfo(BaseModel):
    """Capture preset metadata."""

    name: str
    model_id: str
    lane_name: str
    model_source: str
    default_weight_load: str
    default_dtype: str


class SetupCheck(BaseModel):
    """Single setup-card status."""

    name: str
    ok: bool
    detail: str


class SetupSnapshot(BaseModel):
    """Environment and workflow defaults for the Studio shell."""

    timestamp_utc: str
    repo_root: str
    artifact_root: str
    active_artifact_root: str
    python_version: str
    uv_version: str | None = None
    node_version: str | None = None
    npm_version: str | None = None
    target_cuda: str = REQUIRED_CUDA
    torch_version: str | None = None
    torch_cuda: str | None = None
    cuda_available: bool = False
    gpu_names: list[str] = Field(default_factory=list)
    repo_contract_ok: bool = False
    repo_contract_errors: list[str] = Field(default_factory=list)
    vendored_runtime_ready: bool = False
    checks: list[SetupCheck] = Field(default_factory=list)
    model_presets: list[ModelPresetInfo] = Field(default_factory=list)
    prompt_panel: list[PromptPanelEntry] = Field(default_factory=list)
    capture_presets: list[str] = Field(default_factory=lambda: list(model_preset_names()))
    runtime_profiles: list[str] = Field(default_factory=lambda: sorted(CURRENT_MAIN_RUNTIME_PROFILES))
    compare_modes: list[str] = Field(
        default_factory=lambda: [
            "exact",
            "key_only_random",
            "full_kv",
            "asym_q8_turbo4",
            "asym_q8_turbo3",
            "multiscreen_relevance",
            "key_only_block_so8_triality_vector",
        ]
    )
    paper_validate_variants: list[str] = Field(default_factory=lambda: ["captured_qwen", "synthetic"])


class ArtifactRecord(BaseModel):
    """Artifact tree node."""

    relative_path: str
    absolute_path: str
    kind: Literal["directory", "file"]
    size_bytes: int | None = None
    modified_at: str | None = None
    children: list["ArtifactRecord"] | None = None


class BaseJobPayload(BaseModel):
    """Common fields for UI-triggered jobs."""

    dry_run: bool = False


class CaptureFormState(BaseJobPayload):
    """Capture workflow request."""

    model_id: str | None = None
    model_preset: str | None = DEFAULT_CAPTURE_MODEL_PRESET
    lane_name: str | None = None
    prompt: str | None = None
    prompt_label: str = "custom"
    output_dir: str = "artifacts/kv"
    weight_load: Literal["4bit", "8bit", "none"] = "4bit"
    dtype: str = "float16"
    trust_remote_code: bool = False
    max_length: int = 96
    seed: int = 0


class PaperValidateSpec(BaseJobPayload):
    """Paper-baseline workflow request."""

    variant: Literal["captured_qwen", "synthetic"] = "captured_qwen"
    kv_dir: str = "artifacts/kv_rtx3060_qwen9b"
    trials: int = 3
    max_layers: int = 0
    bits: str = "2,2.5,3,3.5,4,8"
    output_dir: str | None = None
    write_config: bool = False
    config_out: str | None = None
    synthetic_layers: int = 4
    batch: int = 1
    heads: int = 2
    seq_len: int = 64
    head_dim: int = 128
    dim: int = 128
    num_vectors: int = 1024
    num_pairs: int = 2048


class MatrixRunSpec(BaseJobPayload):
    """Qwen 3060 matrix workflow request."""

    kv_dir: str = "artifacts/kv_rtx3060_qwen9b"
    rotation_dir: str = "artifacts/research_extension/triality_full_train_prod_bf16/rotations"
    bits: str = "3,3.5,4"
    trials: int = 3
    max_layers: int = 2
    eval_device: str = "cuda"
    output_dir: str = "artifacts/qwen_3060_matrix"
    skip_statistics: bool = False
    skip_plots: bool = False
    ms_regular_bits: int = 2
    ms_outlier_bits: int = 4
    ms_outlier_count: int = 64


class ExportReportSpec(BaseJobPayload):
    """Offline report export request."""

    matrix_dir: str | None = "artifacts/qwen_3060_matrix"


class ExportOnlineReportSpec(BaseJobPayload):
    """Online report export request."""

    hf_dir: str = "artifacts/hf_online_eval"
    runtime_dir: str = "artifacts/runtime_eval"
    replay_summary_csv: str = "artifacts/qwen_3060_matrix/metrics/qwen_3060_matrix_summary.csv"
    output_dir: str = "artifacts/online_eval_report"


class RuntimeEvalSpec(BaseJobPayload):
    """Runtime evaluation request."""

    mode: str
    model_path: str
    perplexity_bin: str | None = None
    llama_bench_bin: str | None = None
    server_bin: str | None = None
    corpus_file: str | None = None
    server_base_url: str | None = None
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    server_ready_timeout_sec: float = 120.0
    server_log_prefix: str | None = None
    server_context_size: int = 4096
    server_n_gpu_layers: str = "99"
    server_model_name: str = "qwen-runtime"
    tokenizer_path: str | None = None
    runtime_profile: str | None = None
    runtime_env_json: str | None = None
    threads: int = 4
    repetitions: int = 3
    context_size: int = 512
    batch_size: int = 128
    stride: int = 256
    chunks: int = 0
    n_prompt: int = 256
    n_gen: int = 64
    mcq_tasks: str = ",".join(MCQ_RUNTIME_TASKS)
    chat_tasks: str = ",".join(CHAT_RUNTIME_TASKS)
    lm_eval_limit: int = 0
    allow_mcq_unavailable: bool = False
    output_dir: str = "artifacts/runtime_eval"


class PackageSpec(BaseJobPayload):
    """GGUF packaging request."""

    input_gguf: str
    output_gguf: str
    profiles: str = "paper,so8_triality_vector"
    default_profile: str = "exact"
    hypura_compatible_profile: str = "auto"
    bits: float = 3.5
    rotation_dir: str = "artifacts/research_extension/triality_full_train/rotations"
    paper_rotation_seed: int = 0
    paper_qjl_seed: int = 1
    force: bool = False


class ServeSpec(BaseJobPayload):
    """Local serving request."""

    server_bin: str | None = None
    model_path: str | None = None
    gguf: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 4096
    threads: int = 4
    model_alias: str = "qwen-runtime"
    n_gpu_layers: str = "99"
    runtime_profile: str | None = None
    runtime_env_json: str | None = None
    turboquant_mode: str = "gguf-auto"
    release: bool = False


class JobSpec(BaseModel):
    """Persisted Studio job request."""

    kind: str
    dry_run: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    """Stored/running Studio job metadata."""

    id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed", "canceled", "cancel_requested"]
    dry_run: bool
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    mode: str | None = None
    label: str | None = None
    artifact_root: str | None = None
    log_path: str | None = None
    log_tail: str | None = None
    exit_code: int | None = None
    pid: int | None = None
    error: str | None = None
    spec: JobSpec
    result: dict[str, Any] | None = None


SetupSnapshot.model_rebuild()
ArtifactRecord.model_rebuild()


DEFAULT_SETUP_SNAPSHOT = SetupSnapshot(
    timestamp_utc=datetime.now(timezone.utc).isoformat(),
    repo_root="",
    artifact_root="",
    active_artifact_root="",
    python_version="",
    prompt_panel=[PromptPanelEntry(label=item.label, prompt=item.prompt) for item in DEFAULT_PROMPT_PANEL],
)
