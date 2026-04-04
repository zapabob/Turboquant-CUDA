use serde::{Deserialize, Serialize};

// ── Request types ──

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: GenerateOptions,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: GenerateOptions,
    /// Accepted but ignored for MVP.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Ollama/OpenClaw compatibility flag (currently accepted and ignored).
    #[serde(default)]
    pub think: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct GenerateOptions {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_a: Option<f32>,
    pub top_p: Option<f32>,
    pub tfs: Option<f32>,
    pub typical: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub num_predict: Option<u32>,
    pub num_ctx: Option<u32>,
    pub seed: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub sampler_order: Option<Vec<i32>>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    pub smoothing_factor: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tq_so8_off: Option<bool>,
    pub tq_so8_learned: Option<bool>,
    pub tq_triality_off: Option<bool>,
    pub tq_triality_mix: Option<f32>,
    pub tq_rotation_seed: Option<u32>,
    pub tq_artifact: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ShowRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

// ── KoboldCpp-compatible request/response types ──

#[derive(Debug, Deserialize)]
pub struct KoboldGenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_length: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub rep_pen: Option<f32>,
    #[serde(default)]
    pub rep_pen_range: Option<i32>,
    #[serde(default)]
    pub stop_sequence: Option<Vec<String>>,
    #[serde(default)]
    pub sampler_order: Option<Vec<i32>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub top_a: Option<f32>,
    #[serde(default)]
    pub tfs: Option<f32>,
    #[serde(default)]
    pub typical: Option<f32>,
    #[serde(default)]
    pub mirostat: Option<i32>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    #[serde(default)]
    pub dynatemp_range: Option<f32>,
    #[serde(default)]
    pub dynatemp_exponent: Option<f32>,
    #[serde(default)]
    pub smoothing_factor: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub tq_so8_off: Option<bool>,
    #[serde(default)]
    pub tq_so8_learned: Option<bool>,
    #[serde(default)]
    pub tq_triality_off: Option<bool>,
    #[serde(default)]
    pub tq_triality_mix: Option<f32>,
    #[serde(default)]
    pub tq_rotation_seed: Option<u32>,
    #[serde(default)]
    pub tq_artifact: Option<String>,
}

// ── Response types ──

#[derive(Debug, Serialize)]
pub struct GenerateResponseChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseChunk {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct TagsResponse {
    pub models: Vec<ModelTag>,
}

#[derive(Debug, Serialize)]
pub struct ModelTag {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub details: ModelDetails,
}

#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize)]
pub struct ShowResponse {
    pub details: ModelDetails,
    pub model_info: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct KoboldModelResponse {
    pub result: String,
}

#[derive(Debug, Serialize)]
pub struct KoboldGenerateResponse {
    pub results: Vec<KoboldGenerateResult>,
}

#[derive(Debug, Serialize)]
pub struct KoboldGenerateResult {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct KoboldAbortResponse {
    pub success: bool,
}

#[derive(Debug, Serialize)]
pub struct KoboldTrueMaxContextLengthResponse {
    pub value: u32,
}

#[derive(Debug, Serialize)]
pub struct AvailableModelItem {
    pub name: String,
    pub path: String,
    pub selected: bool,
}

#[derive(Debug, Serialize)]
pub struct AvailableModelsResponse {
    pub models: Vec<AvailableModelItem>,
    pub active_model_path: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelSwitchRequest {
    pub path: String,
    #[serde(default)]
    pub context: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ModelSwitchResponse {
    pub success: bool,
    pub model: String,
    pub context: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiPresetItem {
    pub name: String,
    pub payload: serde_json::Value,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct GuiPresetListResponse {
    pub presets: Vec<GuiPresetItem>,
}

#[derive(Debug, Deserialize)]
pub struct GuiPresetSaveRequest {
    pub name: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GuiPresetDeleteRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GuiHistoryItem {
    pub ts: String,
    pub mode: String,
    pub model: String,
    pub prompt_chars: usize,
    pub output_chars: usize,
    pub tok_per_sec_avg: Option<f64>,
    pub total_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct GuiHistoryResponse {
    pub items: Vec<GuiHistoryItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GuiEventItem {
    pub ts: String,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct GuiEventsResponse {
    pub items: Vec<GuiEventItem>,
}

#[derive(Debug, Serialize)]
pub struct UiThemeResponse {
    pub theme: String,
}

#[derive(Debug, Deserialize)]
pub struct UiThemeUpdateRequest {
    pub theme: String,
}

fn default_true() -> bool {
    true
}

/// Produce an RFC 3339 timestamp with nanosecond precision.
pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

/// Lightweight GGUF info kept in AppState (no tensor data).
#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub file_size: u64,
    pub architecture: String,
    pub parameter_count: u64,
    pub quantization: String,
    pub context_length: u32,
}
