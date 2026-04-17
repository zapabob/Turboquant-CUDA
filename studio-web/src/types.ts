export type JobState = "queued" | "running" | "completed" | "failed" | "canceled" | "cancel_requested";

export interface SetupCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface ModelPresetInfo {
  name: string;
  model_id: string;
  lane_name: string;
  model_source: string;
  default_weight_load: string;
  default_dtype: string;
}

export interface PromptPanelEntry {
  label: string;
  prompt: string;
}

export interface SetupSnapshot {
  timestamp_utc: string;
  repo_root: string;
  artifact_root: string;
  active_artifact_root: string;
  python_version: string;
  uv_version: string | null;
  node_version: string | null;
  npm_version: string | null;
  target_cuda: string;
  torch_version: string | null;
  torch_cuda: string | null;
  cuda_available: boolean;
  gpu_names: string[];
  repo_contract_ok: boolean;
  repo_contract_errors: string[];
  vendored_runtime_ready: boolean;
  checks: SetupCheck[];
  model_presets: ModelPresetInfo[];
  prompt_panel: PromptPanelEntry[];
  capture_presets: string[];
  runtime_profiles: string[];
  compare_modes: string[];
  paper_validate_variants: string[];
}

export interface JobSpec {
  kind: string;
  dry_run: boolean;
  payload: Record<string, unknown>;
}

export interface JobStatus {
  id: string;
  kind: string;
  status: JobState;
  dry_run: boolean;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  mode?: string | null;
  label?: string | null;
  artifact_root?: string | null;
  log_path?: string | null;
  log_tail?: string | null;
  exit_code?: number | null;
  pid?: number | null;
  error?: string | null;
  spec: JobSpec;
  result?: Record<string, unknown> | null;
}

export interface ArtifactSummary {
  artifact_root: string;
  existing_files: number;
  existing_directories: number;
  known_paths: Record<string, { path: string; exists: boolean }>;
}

export interface ArtifactRecord {
  relative_path: string;
  absolute_path: string;
  kind: "directory" | "file";
  size_bytes?: number | null;
  modified_at?: string | null;
  children?: ArtifactRecord[] | null;
}
