use std::collections::HashMap;
use std::env;
use std::ffi::{c_void, CStr};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::cache::kv_codec::{build_kv_codec, KvCodec};
use crate::cache::kv_codec_ffi::KvCodecRuntimeBridge;
use crate::compute::ffi::*;
use crate::compute::nvme_backend::{
    build_override_patterns, eval_callback, HypuraBuftController, LayerStatus, PrefetchState,
};
use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;
use crate::model::tensor_role::TensorRole;
use crate::model::turboquant_sidecar::{
    resolve_turboquant_config, GgufTurboQuantConfig, ResolvedTurboQuantConfig, TurboQuantMode,
};
use crate::profiler;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::placement::{compute_placement_with_context, summarize_placement};
use crate::scheduler::types::*;
use crate::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};

/// A token emitted during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub text: String,
    pub token_id: i32,
    pub tok_per_sec: f64,
    pub is_eog: bool,
}

/// Configuration for an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub sampling: SamplingParams,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: num_performance_cores(),
            sampling: SamplingParams::default(),
        }
    }
}

/// Result returned after generation completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: u32,
    pub prompt_tokens: u32,
    pub tok_per_sec_avg: f64,
    pub prompt_eval_ms: f64,
    pub perf: PerfData,
}

/// A loaded model that can serve multiple generation requests.
///
/// Holds the llama.cpp backend, model, and NVMe scheduling state.
/// Context + sampler are created fresh per request (cheap to create, expensive to keep).
pub struct LoadedModel {
    pub _backend: LlamaBackend,
    pub model: LlamaModel,
    pub config: InferenceConfig,
    pub n_gpu_layers: i32,
    pub model_name: String,
    pub turboquant: ResolvedTurboQuantConfig,
    turboquant_layout: Option<TurboQuantRuntimeLayout>,
    // NVMe scheduling state (None when all tensors fit in GPU+RAM)
    _controller: Option<Box<HypuraBuftController>>,
    prefetch_state: Option<Arc<PrefetchState>>,
    keep_resident: bool,
}

// SAFETY: All access to LoadedModel is serialized under std::sync::Mutex
// and only happens from spawn_blocking threads.
unsafe impl Send for LoadedModel {}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        if let Some(ref state) = self.prefetch_state {
            state.stop_io_pool();
        }
    }
}

/// Parameters for `generate_from_loaded`.
pub struct GenerateFromLoadedParams<'a> {
    pub prompt: &'a str,
    pub sampling: &'a SamplingParams,
    pub sampler_order: Option<&'a [i32]>,
    pub stop_sequences: &'a [String],
    pub cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub token_tx: mpsc::UnboundedSender<GeneratedToken>,
    pub telemetry: Arc<TelemetryEmitter>,
}

#[derive(Debug)]
pub struct RuntimeSetup {
    pub hardware: HardwareProfile,
    pub gguf: GgufFile,
    pub metadata: ModelMetadata,
    pub plan: PlacementPlan,
    pub placement_summary: PlacementSummary,
    pub n_gpu_layers: i32,
    pub turboquant: ResolvedTurboQuantConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct TurboQuantRuntimeLayout {
    tracked_layers: u32,
    tracked_heads: u32,
    tracked_head_dim: usize,
}

pub struct TurboQuantRuntimeSession {
    mode: TurboQuantMode,
    layout: TurboQuantRuntimeLayout,
    runtime_path: TurboQuantRuntimePath,
    codec: RuntimeCodecEngine,
    query_vectors: Mutex<HashMap<(u32, u32, u32), Vec<f32>>>,
    softmax_weights: Mutex<HashMap<(u32, u32, u32), Vec<f32>>>,
    next_token_index: AtomicU32,
    pending_batch: Mutex<Option<PendingBatch>>,
    q_callback_hits: AtomicU32,
    k_callback_hits: AtomicU32,
    v_callback_hits: AtomicU32,
    kq_callback_hits: AtomicU32,
    kq_soft_max_callback_hits: AtomicU32,
    kqv_callback_hits: AtomicU32,
    kq_overwrites: AtomicU32,
    kqv_overwrites: AtomicU32,
}

#[derive(Debug, Clone, Copy)]
enum TurboQuantRuntimePath {
    RustDirectCallback,
    CCodecFfi,
}

impl TurboQuantRuntimePath {
    fn as_str(self) -> &'static str {
        match self {
            TurboQuantRuntimePath::RustDirectCallback => "RustDirectCallback",
            TurboQuantRuntimePath::CCodecFfi => "CCodecFfi",
        }
    }
}

enum RuntimeCodecEngine {
    Rust(Mutex<Box<dyn KvCodec + Send>>),
    Cffi(Mutex<KvCodecRuntimeBridge>),
}

impl RuntimeCodecEngine {
    fn ingest_k(&self, layer: u32, head: u32, token: u32, data: &[f32]) -> anyhow::Result<Vec<f32>> {
        match self {
            RuntimeCodecEngine::Rust(codec) => codec.lock().unwrap().ingest_k(layer, head, token, data),
            RuntimeCodecEngine::Cffi(codec) => codec.lock().unwrap().compress_k(layer, head, token, data),
        }
    }

    fn ingest_v(&self, layer: u32, head: u32, token: u32, data: &[f32]) -> anyhow::Result<Vec<f32>> {
        match self {
            RuntimeCodecEngine::Rust(codec) => codec.lock().unwrap().ingest_v(layer, head, token, data),
            RuntimeCodecEngine::Cffi(codec) => codec.lock().unwrap().compress_v(layer, head, token, data),
        }
    }

    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: std::ops::Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        match self {
            RuntimeCodecEngine::Rust(codec) => codec.lock().unwrap().score_k(layer, head, query, token_range),
            RuntimeCodecEngine::Cffi(codec) => codec.lock().unwrap().score_k(layer, head, query, token_range),
        }
    }

    fn read_v(
        &self,
        layer: u32,
        head: u32,
        token_range: std::ops::Range<u32>,
        head_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        match self {
            RuntimeCodecEngine::Rust(codec) => codec.lock().unwrap().read_v(layer, head, token_range),
            RuntimeCodecEngine::Cffi(codec) => codec.lock().unwrap().read_v(layer, head, token_range, head_dim),
        }
    }
}

#[derive(Debug, Clone)]
struct PendingBatch {
    start_token: u32,
    token_ids: Vec<i32>,
}

struct InferenceCallbackState {
    prefetch_state: Option<Arc<PrefetchState>>,
    turboquant: Option<Arc<TurboQuantRuntimeSession>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeTensorComponent {
    Query,
    Key,
    Value,
    Score,
    ScoreSoftmax,
    ValueOutput,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeTensorShape {
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
}

impl RuntimeTensorShape {
    fn total_elements(self) -> usize {
        self.dim0
            .saturating_mul(self.dim1)
            .saturating_mul(self.dim2)
            .saturating_mul(self.dim3)
    }

    fn width(self) -> usize {
        self.dim0
    }

    fn tokens_per_stream(self) -> usize {
        self.dim1
    }

    fn n_heads(self) -> usize {
        self.dim2.max(1)
    }

    fn n_streams(self) -> usize {
        self.dim3.max(1)
    }

    fn total_tokens(self) -> usize {
        self.tokens_per_stream().saturating_mul(self.n_streams())
    }
}

pub fn resolve_runtime_setup(
    model_path: &Path,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&Path>,
) -> anyhow::Result<RuntimeSetup> {
    anyhow::ensure!(
        model_path.exists(),
        "Model file not found: {}",
        model_path.display()
    );

    let hardware = match profiler::load_cached_profile()? {
        Some(p) if !profiler::is_profile_stale(&p) => p,
        _ => {
            println!("No hardware profile found. Running profiler...");
            let p = profiler::run_full_profile()?;
            profiler::save_profile(&p)?;
            p
        }
    };

    let gguf = GgufFile::open(model_path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;
    let turboquant =
        resolve_turboquant_config(model_path, &metadata, &gguf, turboquant_mode, turboquant_config)?;
    apply_gguf_turboquant_env(turboquant.gguf_metadata.as_ref());
    let plan = compute_placement_with_context(&gguf, &hardware, context)?;
    let placement_summary = summarize_placement(&plan.tier_assignments, &gguf.tensors);
    let gpu_budget = compute_gpu_budget(&hardware, &metadata, context);
    let n_gpu_layers = gpu_layers_from_placement(&plan, &gguf, gpu_budget);

    Ok(RuntimeSetup {
        hardware,
        gguf,
        metadata,
        plan,
        placement_summary,
        n_gpu_layers,
        turboquant,
    })
}

fn apply_gguf_turboquant_env(gguf_turboquant: Option<&GgufTurboQuantConfig>) {
    let Some(cfg) = gguf_turboquant else {
        return;
    };

    std::env::set_var("LLAMA_TURBOQUANT", if cfg.enabled { "1" } else { "0" });
    if let Some(rotation_policy) = cfg.rotation_policy {
        let so8_enabled = !matches!(rotation_policy, crate::model::turboquant_sidecar::RotationPolicy::RandomHaar);
        let so8_learned = matches!(rotation_policy, crate::model::turboquant_sidecar::RotationPolicy::BlockSo8Learned);
        std::env::set_var("LLAMA_TURBOQUANT_SO8", if so8_enabled { "1" } else { "0" });
        std::env::set_var("LLAMA_TURBOQUANT_SO8_LEARNED", if so8_learned { "1" } else { "0" });
        std::env::set_var(
            "LLAMA_TURBOQUANT_TRIALITY",
            if rotation_policy.is_triality() { "1" } else { "0" },
        );
    }
    if let Some(mix) = cfg.triality_mix {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{mix:.3}"));
    }
    std::env::set_var("LLAMA_TURBOQUANT_ROTATION_SEED", cfg.rotation_seed.to_string());
    if let Some(path) = &cfg.artifact_path {
        if !path.trim().is_empty() {
            std::env::set_var("LLAMA_TURBOQUANT_ARTIFACT", path.trim());
        }
    }
}

impl TurboQuantRuntimeSession {
    fn new(
        mode: TurboQuantMode,
        layout: TurboQuantRuntimeLayout,
        runtime_path: TurboQuantRuntimePath,
        codec: RuntimeCodecEngine,
    ) -> Self {
        Self {
            mode,
            layout,
            runtime_path,
            codec,
            query_vectors: Mutex::new(HashMap::new()),
            softmax_weights: Mutex::new(HashMap::new()),
            next_token_index: AtomicU32::new(0),
            pending_batch: Mutex::new(None),
            q_callback_hits: AtomicU32::new(0),
            k_callback_hits: AtomicU32::new(0),
            v_callback_hits: AtomicU32::new(0),
            kq_callback_hits: AtomicU32::new(0),
            kq_soft_max_callback_hits: AtomicU32::new(0),
            kqv_callback_hits: AtomicU32::new(0),
            kq_overwrites: AtomicU32::new(0),
            kqv_overwrites: AtomicU32::new(0),
        }
    }

    fn begin_batch(&self, tokens: &[i32]) {
        let start_token = self
            .next_token_index
            .fetch_add(tokens.len() as u32, Ordering::Relaxed);
        *self.pending_batch.lock().unwrap() = Some(PendingBatch {
            start_token,
            token_ids: tokens.to_vec(),
        });
    }

    fn end_batch(&self) {
        *self.pending_batch.lock().unwrap() = None;
    }

    fn uses_full_kv(&self) -> bool {
        self.mode == TurboQuantMode::PaperFullKv
    }

    fn runtime_path(&self) -> TurboQuantRuntimePath {
        self.runtime_path
    }

    fn observe_eval_callback(&self, tensor: *mut hypura_sys::ggml_tensor, ask: bool) {
        if ask || tensor.is_null() {
            return;
        }

        let Ok(name) = unsafe { CStr::from_ptr((*tensor).name.as_ptr()) }.to_str() else {
            return;
        };

        // Loose `contains` matching to count all node variants (e.g. "Qcur_0", "kqv_rope").
        // apply_runtime_tensor uses strict base-name equality to avoid false positives on
        // future nodes such as "kqv_add" that would match "kqv" via contains.
        if name.contains("Qcur") {
            self.q_callback_hits.fetch_add(1, Ordering::Relaxed);
        } else if name.contains("Kcur") {
            self.k_callback_hits.fetch_add(1, Ordering::Relaxed);
        } else if name.contains("Vcur") {
            self.v_callback_hits.fetch_add(1, Ordering::Relaxed);
        } else if name.contains("kq_soft_max") {
            self.kq_soft_max_callback_hits
                .fetch_add(1, Ordering::Relaxed);
        } else if name.contains("kqv") {
            self.kqv_callback_hits.fetch_add(1, Ordering::Relaxed);
        } else if name.contains("kq") {
            self.kq_callback_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn apply_runtime_tensor(&self, tensor: *mut hypura_sys::ggml_tensor) -> anyhow::Result<()> {
        if tensor.is_null() {
            return Ok(());
        }

        let name = unsafe { CStr::from_ptr((*tensor).name.as_ptr()) }
            .to_string_lossy()
            .to_string();
        let Some(component) = runtime_tensor_component(&name) else {
            return Ok(());
        };
        let Some(layer) = runtime_tensor_layer(&name) else {
            return Ok(());
        };
        if layer >= self.layout.tracked_layers {
            return Ok(());
        }

        let pending = self.pending_batch.lock().unwrap().clone();
        let Some(pending) = pending else {
            return Ok(());
        };

        let shape = tensor_shape(tensor);
        if shape.width() == 0 || shape.n_heads() == 0 || shape.total_tokens() == 0 {
            return Ok(());
        }

        let tracked_heads = shape.n_heads().min(self.layout.tracked_heads as usize);
        let tracked_dim = shape.width().min(self.layout.tracked_head_dim);
        let tracked_tokens = shape.total_tokens().min(pending.token_ids.len());

        if component == RuntimeTensorComponent::Query {
            let values = read_tensor_f32(tensor, shape.total_elements())?;
            let mut queries = self.query_vectors.lock().unwrap();
            for token_offset in 0..tracked_tokens {
                let token = pending.start_token + token_offset as u32;
                let (stream, token_in_stream) = stream_position(token_offset, shape);
                for head in 0..tracked_heads {
                    let range = tensor_vector_range(shape, token_in_stream, head, stream);
                    let mut vector = values[range].to_vec();
                    vector.truncate(tracked_dim);
                    queries.insert((layer, head as u32, token), vector);
                }
            }
            return Ok(());
        }

        if component == RuntimeTensorComponent::Score {
            let mut values = read_tensor_f32(tensor, shape.total_elements())?;
            let mut queries = self.query_vectors.lock().unwrap();
            let tracked_scores = shape.width();

            for token_offset in 0..tracked_tokens {
                let token = pending.start_token + token_offset as u32;
                let (stream, token_in_stream) = stream_position(token_offset, shape);
                for head in 0..tracked_heads {
                    let Some(query) = queries.remove(&(layer, head as u32, token)) else {
                        continue;
                    };
                    let scores =
                        self.codec
                            .score_k(layer, head as u32, &query, 0..tracked_scores as u32)?;
                    let range = tensor_vector_range(shape, token_in_stream, head, stream);
                    for (dst, src) in values[range].iter_mut().zip(scores.iter()) {
                        *dst = *src;
                    }
                }
            }

            write_tensor_f32(tensor, &values)?;
            self.kq_overwrites.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        if component == RuntimeTensorComponent::ScoreSoftmax {
            if !self.uses_full_kv() {
                return Ok(());
            }

            let values = read_tensor_f32(tensor, shape.total_elements())?;
            let mut weights = self.softmax_weights.lock().unwrap();
            for token_offset in 0..tracked_tokens {
                let token = pending.start_token + token_offset as u32;
                let (stream, token_in_stream) = stream_position(token_offset, shape);
                for head in 0..tracked_heads {
                    let range = tensor_vector_range(shape, token_in_stream, head, stream);
                    weights.insert((layer, head as u32, token), values[range].to_vec());
                }
            }
            return Ok(());
        }

        if component == RuntimeTensorComponent::ValueOutput {
            if !self.uses_full_kv() {
                return Ok(());
            }

            let mut values = read_tensor_f32(tensor, shape.total_elements())?;
            let mut weights = self.softmax_weights.lock().unwrap();

            for token_offset in 0..tracked_tokens {
                let token = pending.start_token + token_offset as u32;
                let (stream, token_in_stream) = stream_position(token_offset, shape);
                for head in 0..tracked_heads {
                    let Some(attn) = weights.remove(&(layer, head as u32, token)) else {
                        continue;
                    };
                    let read_back = self.codec.read_v(
                        layer,
                        head as u32,
                        0..attn.len() as u32,
                        tracked_dim,
                    )?;
                    if read_back.is_empty() {
                        continue;
                    }
                    let mut output = vec![0.0f32; tracked_dim];
                    for (token_idx, weight) in attn.iter().enumerate() {
                        let base = token_idx.saturating_mul(tracked_dim);
                        if base + tracked_dim > read_back.len() {
                            break;
                        }
                        for dim in 0..tracked_dim {
                            output[dim] += *weight * read_back[base + dim];
                        }
                    }
                    let range = tensor_vector_range(shape, token_in_stream, head, stream);
                    for (dst, src) in values[range].iter_mut().zip(output.iter()) {
                        *dst = *src;
                    }
                }
            }

            write_tensor_f32(tensor, &values)?;
            self.kqv_overwrites.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let mut values = read_tensor_f32(tensor, shape.total_elements())?;

        for token_offset in 0..tracked_tokens {
            let token = pending.start_token + token_offset as u32;
            let (stream, token_in_stream) = stream_position(token_offset, shape);
            for head in 0..tracked_heads {
                let range = tensor_vector_range(shape, token_in_stream, head, stream);
                let mut vector = values[range.clone()].to_vec();
                vector.truncate(tracked_dim);

                let transformed = match component {
                    RuntimeTensorComponent::Key => {
                        self.codec.ingest_k(layer, head as u32, token, &vector)?
                    }
                    RuntimeTensorComponent::Value => {
                        self.codec.ingest_v(layer, head as u32, token, &vector)?
                    }
                    RuntimeTensorComponent::Query
                    | RuntimeTensorComponent::Score
                    | RuntimeTensorComponent::ScoreSoftmax
                    | RuntimeTensorComponent::ValueOutput => unreachable!(),
                };

                for (dst, src) in values[range].iter_mut().zip(transformed.iter()) {
                    *dst = *src;
                }
            }
        }

        write_tensor_f32(tensor, &values)?;

        Ok(())
    }

    fn callback_hits(&self) -> RuntimeCallbackStats {
        RuntimeCallbackStats {
            q_hits: self.q_callback_hits.load(Ordering::Relaxed),
            k_hits: self.k_callback_hits.load(Ordering::Relaxed),
            v_hits: self.v_callback_hits.load(Ordering::Relaxed),
            kq_hits: self.kq_callback_hits.load(Ordering::Relaxed),
            kq_soft_max_hits: self.kq_soft_max_callback_hits.load(Ordering::Relaxed),
            kqv_hits: self.kqv_callback_hits.load(Ordering::Relaxed),
            kq_overwrites: self.kq_overwrites.load(Ordering::Relaxed),
            kqv_overwrites: self.kqv_overwrites.load(Ordering::Relaxed),
        }
    }

    fn mode(&self) -> TurboQuantMode {
        self.mode
    }
}

fn turboquant_runtime_layout(
    metadata: &ModelMetadata,
    mode: TurboQuantMode,
) -> Option<TurboQuantRuntimeLayout> {
    if mode == TurboQuantMode::Exact {
        return None;
    }

    let tracked_layers = metadata.num_layers.max(1);
    let tracked_heads = metadata.num_kv_heads.max(1);
    let head_dim = if metadata.num_heads > 0 {
        (metadata.embedding_dim / metadata.num_heads).max(1) as usize
    } else {
        1
    };

    Some(TurboQuantRuntimeLayout {
        tracked_layers,
        tracked_heads,
        tracked_head_dim: head_dim.max(1),
    })
}

fn build_turboquant_runtime_session(
    turboquant: &ResolvedTurboQuantConfig,
    layout: Option<TurboQuantRuntimeLayout>,
) -> anyhow::Result<Option<Arc<TurboQuantRuntimeSession>>> {
    let Some(layout) = layout else {
        return Ok(None);
    };

    let runtime_path = match env::var("HYPURA_TURBOQUANT_RUNTIME")
        .unwrap_or_else(|_| "rust".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "cffi" | "ffi" | "c" => TurboQuantRuntimePath::CCodecFfi,
        _ => TurboQuantRuntimePath::RustDirectCallback,
    };

    let codec = match runtime_path {
        TurboQuantRuntimePath::RustDirectCallback => {
            RuntimeCodecEngine::Rust(Mutex::new(build_kv_codec(turboquant)?))
        }
        TurboQuantRuntimePath::CCodecFfi => {
            RuntimeCodecEngine::Cffi(Mutex::new(KvCodecRuntimeBridge::new(turboquant)?))
        }
    };

    tracing::info!(
        "TurboQuant runtime path selected: {} (mode={})",
        runtime_path.as_str(),
        turboquant.mode
    );

    Ok(Some(Arc::new(TurboQuantRuntimeSession::new(
        turboquant.mode,
        layout,
        runtime_path,
        codec,
    ))))
}

fn validate_turboquant_runtime_mode(turboquant: &ResolvedTurboQuantConfig) -> anyhow::Result<()> {
    match turboquant.mode {
        TurboQuantMode::Exact
        | TurboQuantMode::PaperKeyOnly
        | TurboQuantMode::PaperFullKv
        | TurboQuantMode::ResearchKvSplit => {
            Ok(())
        }
    }
}

extern "C" fn eval_callback_with_runtime(
    tensor: *mut hypura_sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if tensor.is_null() || user_data.is_null() {
        return true;
    }

    let state = unsafe { &*(user_data as *const InferenceCallbackState) };
    if let Some(ref turboquant) = state.turboquant {
        turboquant.observe_eval_callback(tensor, ask);
        if !ask {
            if let Err(err) = turboquant.apply_runtime_tensor(tensor) {
                tracing::warn!("TurboQuant runtime callback failed: {err}");
            }
        }
    }

    if let Some(ref prefetch_state) = state.prefetch_state {
        eval_callback(tensor, ask, Arc::as_ptr(prefetch_state) as *mut c_void)
    } else {
        true
    }
}

fn runtime_tensor_component(name: &str) -> Option<RuntimeTensorComponent> {
    let base = name.split('-').next().unwrap_or(name);
    if base.starts_with("Qcur") {
        Some(RuntimeTensorComponent::Query)
    } else if base.starts_with("Kcur") {
        Some(RuntimeTensorComponent::Key)
    } else if base.starts_with("Vcur") {
        Some(RuntimeTensorComponent::Value)
    } else if base == "kq_soft_max" {
        Some(RuntimeTensorComponent::ScoreSoftmax)
    } else if base == "kqv" {
        Some(RuntimeTensorComponent::ValueOutput)
    } else if base == "kq" {
        Some(RuntimeTensorComponent::Score)
    } else {
        None
    }
}

fn runtime_tensor_layer(name: &str) -> Option<u32> {
    name.rsplit_once('-')
        .and_then(|(_, raw)| raw.parse::<u32>().ok())
}

fn tensor_shape(tensor: *mut hypura_sys::ggml_tensor) -> RuntimeTensorShape {
    RuntimeTensorShape {
        dim0: unsafe { (*tensor).ne[0].max(0) as usize },
        dim1: unsafe { (*tensor).ne[1].max(1) as usize },
        dim2: unsafe { (*tensor).ne[2].max(1) as usize },
        dim3: unsafe { (*tensor).ne[3].max(1) as usize },
    }
}

fn tensor_vector_range(
    shape: RuntimeTensorShape,
    token_in_stream: usize,
    head: usize,
    stream: usize,
) -> std::ops::Range<usize> {
    let base = ((((stream * shape.n_heads()) + head) * shape.tokens_per_stream())
        + token_in_stream)
        * shape.width();
    base..(base + shape.width())
}

fn stream_position(token_offset: usize, shape: RuntimeTensorShape) -> (usize, usize) {
    let tokens_per_stream = shape.tokens_per_stream().max(1);
    let stream = token_offset / tokens_per_stream;
    let token_in_stream = token_offset % tokens_per_stream;
    (
        stream.min(shape.n_streams().saturating_sub(1)),
        token_in_stream,
    )
}

#[derive(Debug, Clone, Copy)]
struct RuntimeCallbackStats {
    q_hits: u32,
    k_hits: u32,
    v_hits: u32,
    kq_hits: u32,
    kq_soft_max_hits: u32,
    kqv_hits: u32,
    kq_overwrites: u32,
    kqv_overwrites: u32,
}

fn read_tensor_f32(
    tensor: *mut hypura_sys::ggml_tensor,
    elements: usize,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        unsafe { (*tensor).type_ } == hypura_sys::ggml_type_GGML_TYPE_F32,
        "TurboQuant runtime currently expects F32 callback tensors"
    );
    let mut out = vec![0.0f32; elements];
    unsafe {
        hypura_sys::ggml_backend_tensor_get(
            tensor,
            out.as_mut_ptr() as *mut c_void,
            0,
            elements * std::mem::size_of::<f32>(),
        );
    }
    Ok(out)
}

fn write_tensor_f32(tensor: *mut hypura_sys::ggml_tensor, values: &[f32]) -> anyhow::Result<()> {
    anyhow::ensure!(
        unsafe { (*tensor).type_ } == hypura_sys::ggml_type_GGML_TYPE_F32,
        "TurboQuant runtime currently expects F32 callback tensors"
    );
    unsafe {
        hypura_sys::ggml_backend_tensor_set(
            tensor,
            values.as_ptr() as *const c_void,
            0,
            std::mem::size_of_val(values),
        );
    }
    Ok(())
}

/// Load a model once for repeated generation (server use case).
///
/// Extracts the model loading logic from `generate_with_nvme_scheduling` so the
/// heavy work (GGUF parse, placement, buffer setup, model load, prefetch init)
/// happens once at startup.
pub fn load_model(
    model_path: &Path,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &PlacementPlan,
    gguf: &GgufFile,
    turboquant: &ResolvedTurboQuantConfig,
) -> anyhow::Result<LoadedModel> {
    let backend = LlamaBackend::init();
    let metadata = ModelMetadata::from_gguf(gguf)?;
    let turboquant_layout = turboquant_runtime_layout(&metadata, turboquant.mode);
    validate_turboquant_runtime_mode(turboquant)?;

    let has_nvme = plan
        .tier_assignments
        .values()
        .any(|t| *t == StorageTier::Nvme);

    // Derive model name from GGUF metadata or filename
    let model_name = gguf
        .get_string("general.name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".into())
        });

    if !has_nvme {
        let model = LlamaModel::load(model_path, n_gpu_layers, true)?;
        return Ok(LoadedModel {
            _backend: backend,
            model,
            config: config.clone(),
            n_gpu_layers,
            model_name,
            turboquant: turboquant.clone(),
            turboquant_layout,
            _controller: None,
            prefetch_state: None,
            keep_resident: false,
        });
    }

    // NVMe path: create custom buffer type
    let controller = HypuraBuftController::new(model_path, gguf);
    let (_patterns, overrides) =
        build_override_patterns(plan, gguf, controller.buft_ptr(), n_gpu_layers);

    let nvme_count = plan
        .tier_assignments
        .values()
        .filter(|t| **t == StorageTier::Nvme)
        .count();
    tracing::info!("NVMe scheduling: {nvme_count} tensors on custom buffer type");

    let model =
        LlamaModel::load_with_overrides(model_path, n_gpu_layers, true, overrides.as_ptr())?;
        
    let num_layers = model.n_layers() as u32;

    let nvme_layers: std::collections::HashSet<u32> = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .filter_map(|t| t.layer_index)
        .collect();

    let prefetch_state = controller.build_prefetch_state(gguf, num_layers, nvme_layers);

    // Determine keep-resident mode
    let total_ram = total_physical_memory();
    let nvme_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .map(|t| t.size_bytes)
        .sum();

    let model_total_bytes = gguf.total_tensor_bytes();
    let buffer_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| {
            let tier = plan.tier_assignments.get(&t.name);
            tier == Some(&StorageTier::Nvme) || tier == Some(&StorageTier::Ram)
        })
        .map(|t| t.size_bytes)
        .sum();
    let gpu_bytes = model_total_bytes.saturating_sub(buffer_bytes);
    let gpu_committed_estimate = gpu_bytes * 60 / 100;
    let runtime_overhead: u64 = 5 * (1 << 29);
    let estimated_committed = gpu_committed_estimate + buffer_bytes + runtime_overhead;
    let headroom: u64 = 4 * (1 << 30);

    let keep_resident =
        nvme_bytes > 0 && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(headroom);

    let should_preload = keep_resident
        && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(6 * (1 << 30));

    if keep_resident {
        tracing::info!(
            "NVMe keep-resident mode: {:.2} GB NVMe spill, est. committed {:.1}/{:.1} GB",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            (estimated_committed + nvme_bytes) as f64 / (1u64 << 30) as f64,
            total_ram as f64 / (1u64 << 30) as f64,
        );
    } else if nvme_bytes > 0 {
        tracing::info!(
            "NVMe streaming mode: {:.2} GB NVMe spill",
            nvme_bytes as f64 / (1u64 << 30) as f64,
        );
    }
    prefetch_state
        .keep_nvme_resident
        .store(keep_resident, std::sync::atomic::Ordering::Relaxed);

    // Start multi-threaded I/O pool
    let num_io_workers = (num_performance_cores() as usize / 2).clamp(2, 4);
    prefetch_state.start_io_pool(num_io_workers)?;

    if should_preload {
        prefetch_state.preload_ram_layers();
    } else if keep_resident {
        tracing::info!(
            "Skipping preload: model {:.1} GB exceeds preload threshold (will load lazily)",
            model_total_bytes as f64 / (1u64 << 30) as f64,
        );
    }

    // Initial NVMe prefetch
    prefetch_state.prefetch_all_nvme();

    if keep_resident {
        prefetch_state
            .prefetch_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    Ok(LoadedModel {
        _backend: backend,
        model,
        config: config.clone(),
        n_gpu_layers,
        model_name,
        turboquant: turboquant.clone(),
        turboquant_layout,
        _controller: Some(controller),
        prefetch_state: Some(prefetch_state),
        keep_resident,
    })
}

/// Generate text from a pre-loaded model.
///
/// Creates a fresh context + sampler per request. The model itself is reused.
pub fn generate_from_loaded(
    loaded: &mut LoadedModel,
    params: GenerateFromLoadedParams<'_>,
) -> anyhow::Result<GenerationResult> {
    let GenerateFromLoadedParams {
        prompt,
        sampling,
        sampler_order,
        stop_sequences,
        cancel_flag,
        token_tx,
        telemetry,
    } = params;

    // Build context — with or without NVMe callback
    let turboquant_session =
        build_turboquant_runtime_session(&loaded.turboquant, loaded.turboquant_layout)?;
    let config = &loaded.config;
    let callback_state = if loaded.prefetch_state.is_some() || turboquant_session.is_some() {
        Some(Box::new(InferenceCallbackState {
            prefetch_state: loaded.prefetch_state.clone(),
            turboquant: turboquant_session.clone(),
        }))
    } else {
        None
    };
    let callback_ptr = callback_state
        .as_ref()
        .map(|state| state.as_ref() as *const InferenceCallbackState as *mut c_void);
    let mut ctx = if let Some(state_ptr) = callback_ptr {
        LlamaContext::new_with_callback_and_options(
            &loaded.model,
            config.n_ctx,
            config.n_batch,
            config.n_threads,
            Some(eval_callback_with_runtime),
            state_ptr,
            turboquant_session.is_some(),
        )?
        // Immediately convert back to avoid leak — the PrefetchState is kept alive
        // by the Arc in LoadedModel, not by this raw pointer.
    } else {
        LlamaContext::new(
            &loaded.model,
            config.n_ctx,
            config.n_batch,
            config.n_threads,
        )?
    };

    let mut sampler = LlamaSampler::new_with_order(sampling, sampler_order);

    let tokens = loaded.model.tokenize(prompt, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Prefetch NVMe layers before prompt eval
    if let Some(ref state) = loaded.prefetch_state {
        state.prefetch_all_nvme();
    }

    // Process prompt
    let prompt_start = Instant::now();
    let batch_size = config.n_batch as usize;
    for chunk in tokens.chunks(batch_size) {
        if let Some(ref session) = turboquant_session {
            session.begin_batch(chunk);
        }
        ctx.decode(chunk)?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Generation loop
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..sampling.max_tokens {
        if let Some(flag) = &cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
        }
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = loaded.model.is_eog(token_id);
        let piece = loaded.model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        if token_tx
            .send(GeneratedToken {
                text: piece,
                token_id,
                tok_per_sec,
                is_eog,
            })
            .is_err()
        {
            break;
        }

        if is_eog {
            break;
        }

        if let Some(matched) = stop_sequences
            .iter()
            .find(|seq| !seq.is_empty() && generated_text.ends_with(seq.as_str()))
        {
            let trim_len = generated_text.len().saturating_sub(matched.len());
            generated_text.truncate(trim_len);
            break;
        }

        if !loaded.keep_resident {
            if let Some(ref state) = loaded.prefetch_state {
                state.prefetch_all_nvme();
            }
        }

        if let Some(ref session) = turboquant_session {
            session.begin_batch(&[token_id]);
        }
        ctx.decode(&[token_id])?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
    }

    if let Some(ref session) = turboquant_session {
        let stats = session.callback_hits();
        tracing::info!(
            "TurboQuant request session complete: path={}, mode={}, q_hits={}, k_hits={}, v_hits={}, kq_hits={}, kq_soft_max_hits={}, kqv_hits={}, kq_overwrites={}, kqv_overwrites={}",
            session.runtime_path().as_str(),
            session.mode(),
            stats.q_hits,
            stats.k_hits,
            stats.v_hits,
            stats.kq_hits,
            stats.kq_soft_max_hits,
            stats.kqv_hits,
            stats.kq_overwrites,
            stats.kqv_overwrites,
        );
    }

    let perf = ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Compute GPU budget for model weights (bytes) after reserving space for
/// KV cache and compute buffers within the Metal working set.
pub fn compute_gpu_budget(
    hw: &HardwareProfile,
    metadata: &ModelMetadata,
    context_length: u32,
) -> u64 {
    let gpu_working_set = hw.gpu.as_ref().map_or(0, |g| g.vram_bytes);
    // KV cache on GPU: 2 * layers * kv_heads * head_dim * 2 bytes * context
    let head_dim = if metadata.num_heads > 0 {
        metadata.embedding_dim as u64 / metadata.num_heads as u64
    } else {
        0
    };
    let kv_on_gpu = 2
        * metadata.num_layers as u64
        * metadata.num_kv_heads as u64
        * head_dim
        * 2
        * context_length as u64;
    // Reserve 2 GiB for compute buffers + Metal framework overhead
    let runtime_overhead: u64 = 2 * (1 << 30);
    gpu_working_set
        .saturating_sub(kv_on_gpu)
        .saturating_sub(runtime_overhead)
}

/// Derive `n_gpu_layers` from a PlacementPlan.
///
/// In expert-streaming mode, expert tensors are on the Hypura NVMe buffer (not Metal),
/// so they don't count against GPU working set or the NVMe-layer cutoff. All layers
/// can be offloaded to Metal — the eval_callback loads expert data on demand.
///
/// In other modes, caps at the minimum of:
/// 1. The first NVMe layer (layers with NVMe tensors must stay on CPU)
/// 2. The GPU working set capacity
pub fn gpu_layers_from_placement(
    plan: &PlacementPlan,
    gguf: &GgufFile,
    gpu_budget_bytes: u64,
) -> i32 {
    // SparseMoeMmap: if model fits in GPU, offload all layers. If not, use CPU-only
    // (ngl=0) and rely on mmap + OS page cache for the sparse active working set.
    if plan.inference_mode == InferenceMode::SparseMoeMmap {
        let total_bytes = gguf.total_tensor_bytes();
        if total_bytes <= gpu_budget_bytes {
            let max_layer = gguf
                .tensors
                .iter()
                .filter_map(|t| t.layer_index)
                .max()
                .unwrap_or(0);
            return max_layer as i32 + 1 + 1; // all layers + output
        } else {
            tracing::info!(
                "Sparse MoE mmap: model ({:.1} GB) exceeds GPU budget ({:.1} GB), using CPU-only (ngl=0)",
                total_bytes as f64 / (1u64 << 30) as f64,
                gpu_budget_bytes as f64 / (1u64 << 30) as f64,
            );
            return 0;
        }
    }

    let expert_streaming = plan.inference_mode == InferenceMode::ExpertStreaming;
    let dense_ffn_streaming = plan.inference_mode == InferenceMode::DenseFfnStreaming;
    let mut max_layer: i32 = -1;
    let mut first_nvme_layer: Option<u32> = None;

    // Compute per-layer sizes (excluding streamed tensors from GPU budget)
    let mut layer_sizes: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();

    for t in &gguf.tensors {
        if let Some(layer_idx) = t.layer_index {
            max_layer = max_layer.max(layer_idx as i32);

            // In streaming modes, NVMe-streamed tensors go to the Hypura pool buffer,
            // not Metal shared buffers — don't count them for GPU budget/NVMe cutoff.
            if expert_streaming {
                let role = TensorRole::from_name(&t.name);
                if matches!(role, TensorRole::MoeFusedExperts) {
                    continue;
                }
            }
            if dense_ffn_streaming {
                let role = TensorRole::from_name(&t.name);
                if matches!(
                    role,
                    TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                ) {
                    continue;
                }
            }

            *layer_sizes.entry(layer_idx).or_default() += t.size_bytes;
            if plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme) {
                first_nvme_layer = Some(match first_nvme_layer {
                    Some(existing) => existing.min(layer_idx),
                    None => layer_idx,
                });
            }
        }
    }

    if max_layer < 0 {
        return 0;
    }

    // Cap by NVMe cutoff (in expert-streaming, first_nvme_layer is None → no cap)
    let from_nvme = match first_nvme_layer {
        Some(nvme_start) => nvme_start as i32 + 1,
        None => max_layer + 1 + 1,
    };

    // Cap by GPU working set: sum layers until budget exhausted.
    let non_layer_gpu_size: u64 = gguf
        .tensors
        .iter()
        .filter(|t| t.layer_index.is_none())
        .map(|t| t.size_bytes)
        .sum();
    let mut cumulative: u64 = non_layer_gpu_size;
    let mut max_fitting: i32 = 0;
    for (&layer_idx, &size) in &layer_sizes {
        cumulative += size;
        if cumulative <= gpu_budget_bytes {
            max_fitting = layer_idx as i32 + 1;
        } else {
            break;
        }
    }
    // +1 for the output layer llama.cpp counts separately
    let from_capacity = max_fitting + 1;

    from_nvme.min(from_capacity)
}

/// Run inference on a blocking thread. Streams tokens via `token_tx`.
pub fn generate_blocking(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    token_tx: mpsc::UnboundedSender<GeneratedToken>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<GenerationResult> {
    generate_blocking_internal(
        model_path,
        prompt,
        config,
        n_gpu_layers,
        token_tx,
        telemetry,
        None,
    )
}

fn generate_blocking_internal(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    token_tx: mpsc::UnboundedSender<GeneratedToken>,
    telemetry: Arc<TelemetryEmitter>,
    turboquant_session: Option<Arc<TurboQuantRuntimeSession>>,
) -> anyhow::Result<GenerationResult> {
    let _backend = LlamaBackend::init();

    let model = LlamaModel::load(model_path, n_gpu_layers, true)?;
    let callback_state = turboquant_session.as_ref().map(|session| {
        Box::new(InferenceCallbackState {
            prefetch_state: None,
            turboquant: Some(session.clone()),
        })
    });
    let callback_ptr = callback_state
        .as_ref()
        .map(|state| state.as_ref() as *const InferenceCallbackState as *mut c_void);
    let mut ctx = if let Some(state_ptr) = callback_ptr {
        LlamaContext::new_with_callback_and_options(
            &model,
            config.n_ctx,
            config.n_batch,
            config.n_threads,
            Some(eval_callback_with_runtime),
            state_ptr,
            turboquant_session.is_some(),
        )?
    } else {
        LlamaContext::new(&model, config.n_ctx, config.n_batch, config.n_threads)?
    };
    let mut sampler = LlamaSampler::new(&config.sampling);

    // Tokenize prompt
    let tokens = model.tokenize(prompt, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Process prompt
    let prompt_start = Instant::now();
    // Decode in batches if prompt is longer than n_batch
    let batch_size = config.n_batch as usize;
    for chunk in tokens.chunks(batch_size) {
        if let Some(ref session) = turboquant_session {
            session.begin_batch(chunk);
        }
        ctx.decode(chunk)?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Generation loop
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..config.sampling.max_tokens {
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = model.is_eog(token_id);
        let piece = model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        let gen_token = GeneratedToken {
            text: piece,
            token_id,
            tok_per_sec,
            is_eog,
        };

        if token_tx.send(gen_token).is_err() {
            break;
        }

        if is_eog {
            break;
        }

        if let Some(ref session) = turboquant_session {
            session.begin_batch(&[token_id]);
        }
        ctx.decode(&[token_id])?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
    }

    if let Some(ref session) = turboquant_session {
        let stats = session.callback_hits();
        tracing::info!(
            "TurboQuant blocking session complete: path={}, mode={}, q_hits={}, k_hits={}, v_hits={}, kq_hits={}, kq_soft_max_hits={}, kqv_hits={}, kq_overwrites={}, kqv_overwrites={}",
            session.runtime_path().as_str(),
            session.mode(),
            stats.q_hits,
            stats.k_hits,
            stats.v_hits,
            stats.kq_hits,
            stats.kq_soft_max_hits,
            stats.kqv_hits,
            stats.kq_overwrites,
            stats.kqv_overwrites,
        );
    }

    let perf = ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Run inference with NVMe-aware tensor scheduling.
/// Uses custom buffer type for NVMe-tier tensors + cb_eval for layer tracking.
pub fn generate_with_nvme_scheduling(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &PlacementPlan,
    gguf: &GgufFile,
    turboquant: &ResolvedTurboQuantConfig,
    token_tx: mpsc::UnboundedSender<GeneratedToken>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<GenerationResult> {
    let _backend = LlamaBackend::init();
    let metadata = ModelMetadata::from_gguf(gguf)?;
    let turboquant_layout = turboquant_runtime_layout(&metadata, turboquant.mode);
    let turboquant_session = build_turboquant_runtime_session(turboquant, turboquant_layout)?;

    // Check if there are any NVMe tensors
    let has_nvme = plan
        .tier_assignments
        .values()
        .any(|t| *t == StorageTier::Nvme);

    if !has_nvme {
        return generate_blocking_internal(
            model_path,
            prompt,
            config,
            n_gpu_layers,
            token_tx,
            telemetry,
            turboquant_session,
        );
    }

    // Create custom buffer type for NVMe-tier tensors
    let controller = HypuraBuftController::new(model_path, gguf);
    let (_patterns, overrides) =
        build_override_patterns(plan, gguf, controller.buft_ptr(), n_gpu_layers);

    let nvme_count = plan
        .tier_assignments
        .values()
        .filter(|t| **t == StorageTier::Nvme)
        .count();
    tracing::info!("NVMe scheduling: {nvme_count} tensors on custom buffer type");

    // Expert/dense-FFN streaming: use_mmap=false so Metal creates individual GPU buffers
    // for resident tensors instead of one giant MTLBuffer for the entire model file.
    let use_mmap = !matches!(
        plan.inference_mode,
        InferenceMode::ExpertStreaming | InferenceMode::DenseFfnStreaming
    );

    // Dense FFN streaming: redirect FFN tensor fread to a small scratch buffer during
    // model loading. Without this, fread commits ~22 GB of anonymous mmap pages for
    // FFN tensors, causing OOM on 32 GB machines.
    if plan.inference_mode == InferenceMode::DenseFfnStreaming {
        controller.enable_dense_ffn_scratch(gguf);
    }

    let model =
        LlamaModel::load_with_overrides(model_path, n_gpu_layers, use_mmap, overrides.as_ptr())?;

    // Build prefetch state with layer groupings and file offsets
    let num_layers = model.n_layers() as u32;

    // Determine which layers are NVMe (released after use) vs RAM (loaded once, kept)
    let nvme_layers: std::collections::HashSet<u32> = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .filter_map(|t| t.layer_index)
        .collect();

    let prefetch_state = controller.build_prefetch_state(gguf, num_layers, nvme_layers);

    // Determine if NVMe layers can stay resident in physical memory.
    // When the NVMe spill is modest relative to total RAM, we keep NVMe-tier data
    // loaded after the first forward pass — eliminating all NVMe I/O for subsequent
    // tokens. This is the key optimization for "barely overflows" models like
    // Mixtral 30.9GB on 32GB: the 2GB NVMe spill stays resident, so only the first
    // forward pass incurs NVMe I/O.
    let total_ram = total_physical_memory();
    let nvme_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .map(|t| t.size_bytes)
        .sum();
    // Keep-resident mode: keep NVMe data loaded after the first forward pass,
    // eliminating all NVMe I/O for subsequent tokens.
    //
    // Estimate actual committed memory: GPU layers via mmap commit ~60% of their
    // size on Apple Silicon (demand paging), our buffer commits its full allocation,
    // plus ~2.5 GB for KV cache + Metal overhead.
    //
    // Keep-resident when: estimated_committed + nvme_bytes fits in RAM with 4 GB
    // headroom for page cache. This lets small-spill models (Mixtral 2 GB on 32 GB)
    // stay resident while large-spill models (Llama 70B, 16 GB NVMe) use streaming.
    let model_total_bytes = gguf.total_tensor_bytes();
    let buffer_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| {
            let tier = plan.tier_assignments.get(&t.name);
            tier == Some(&StorageTier::Nvme) || tier == Some(&StorageTier::Ram)
        })
        .map(|t| t.size_bytes)
        .sum();
    let gpu_bytes = model_total_bytes.saturating_sub(buffer_bytes);
    let gpu_committed_estimate = gpu_bytes * 60 / 100; // ~60% committed via mmap
    let runtime_overhead: u64 = 5 * (1 << 29); // ~2.5 GB for KV cache + Metal
    let estimated_committed = gpu_committed_estimate + buffer_bytes + runtime_overhead;
    let headroom: u64 = 4 * (1 << 30); // 4 GB for page cache + system

    let keep_resident =
        nvme_bytes > 0 && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(headroom);

    // Preloading is separate: only preload when estimated committed memory
    // (including all buffer layers pre-loaded) fits with 6 GB headroom.
    let should_preload = keep_resident
        && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(6 * (1 << 30));

    if keep_resident {
        tracing::info!(
            "NVMe keep-resident mode: {:.2} GB NVMe spill, est. committed {:.1}/{:.1} GB",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            (estimated_committed + nvme_bytes) as f64 / (1u64 << 30) as f64,
            total_ram as f64 / (1u64 << 30) as f64,
        );
    } else if nvme_bytes > 0 {
        tracing::info!(
            "NVMe streaming mode: {:.2} GB NVMe spill, est. committed {:.1} GB + {:.1} GB NVMe > {:.1} GB limit",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            estimated_committed as f64 / (1u64 << 30) as f64,
            nvme_bytes as f64 / (1u64 << 30) as f64,
            total_ram.saturating_sub(headroom) as f64 / (1u64 << 30) as f64,
        );
    }
    let expert_streaming = plan.inference_mode == InferenceMode::ExpertStreaming;
    let dense_ffn_streaming = plan.inference_mode == InferenceMode::DenseFfnStreaming;
    let any_streaming = expert_streaming || dense_ffn_streaming;

    // Unified memory budget for pool sizing and residency decisions.
    // Streaming modes use use_mmap=false → GPU tensors 100% committed.
    let metadata_for_budget = ModelMetadata::from_gguf(gguf).ok();
    let head_dim = metadata_for_budget
        .as_ref()
        .map(|m| {
            if m.num_heads > 0 {
                m.embedding_dim as u64 / m.num_heads as u64
            } else {
                0
            }
        })
        .unwrap_or(0);
    let num_kv_heads = metadata_for_budget
        .as_ref()
        .map(|m| m.num_kv_heads)
        .unwrap_or(0);

    let memory_budget = MemoryBudget::compute(
        total_ram,
        gpu_bytes,
        !any_streaming, // use_mmap=false for streaming modes
        num_layers,
        num_kv_heads,
        head_dim,
        config.n_ctx,
        plan.kv_cache_plan.kv_quantization,
    );

    tracing::info!(
        "Memory budget: {:.1} GB committed ({:.1} GPU + {:.1} KV + {:.1} Metal + {:.1} OS), {:.1} GB available",
        memory_budget.total_committed as f64 / 1e9,
        memory_budget.gpu_committed as f64 / 1e9,
        memory_budget.kv_cache_bytes as f64 / 1e9,
        memory_budget.metal_overhead as f64 / 1e9,
        memory_budget.os_overhead as f64 / 1e9,
        memory_budget.available as f64 / 1e9,
    );

    if expert_streaming {
        // Expert-streaming: non-expert tensors on GPU/Metal, experts on NVMe buffer.
        let num_experts = gguf.get_u32("expert_count").unwrap_or(8);
        // Dynamic pool slots: each slot = largest fused expert tensor (~385 MB for Mixtral).
        // Min 6 (2 layers × 3 tensors), max 18 (6 layers).
        let expert_slot_size = gguf
            .tensors
            .iter()
            .filter(|t| matches!(TensorRole::from_name(&t.name), TensorRole::MoeFusedExperts))
            .map(|t| t.size_bytes)
            .max()
            .unwrap_or(1);
        let num_slots = memory_budget.pool_slots(expert_slot_size, 6, 18);
        let pool = controller.activate_expert_pool(gguf, num_experts, num_slots)?;
        let memory_budget = memory_budget.with_pool(pool.pool_size as u64);
        tracing::info!(
            "Expert-streaming mode: {:.2} GB expert tensors on NVMe, {:.0} MB pool ({} slots), {:.1} GB available",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            pool.pool_size as f64 / 1e6,
            pool.num_slots,
            memory_budget.available as f64 / 1e9,
        );

        *prefetch_state.expert_pool.lock().unwrap() = Some(pool);
        let tensor_ptrs = controller.take_tensor_ptrs();
        let state_mut = unsafe { &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState) };
        state_mut.fused_tensor_ptrs = tensor_ptrs;

        prefetch_state
            .expert_streaming
            .store(true, std::sync::atomic::Ordering::Relaxed);
    } else if dense_ffn_streaming {
        // Dense FFN-streaming: attention+norms on GPU, FFN on NVMe pool buffer.
        // Dynamic pool slots: each slot = largest FFN tensor (~193 MB for Llama 70B).
        // Min 6 (2 layers × 3 tensors), max 24 (8 layers).
        let ffn_slot_size = gguf
            .tensors
            .iter()
            .filter(|t| {
                matches!(
                    TensorRole::from_name(&t.name),
                    TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                )
            })
            .map(|t| t.size_bytes)
            .max()
            .unwrap_or(1);
        let num_slots = memory_budget.pool_slots(ffn_slot_size, 6, 24);
        let pool = controller.activate_dense_ffn_pool(gguf, num_slots)?;
        let _memory_budget = memory_budget.with_pool(pool.pool_size as u64);
        tracing::info!(
            "Dense FFN-streaming mode: {:.2} GB FFN tensors on NVMe, {:.0} MB pool ({} slots), {:.1} GB available",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            pool.pool_size as f64 / 1e6,
            pool.num_slots,
            _memory_budget.available as f64 / 1e9,
        );

        // Prefetch lookahead = (pool_slots / 3) - 1 (reserve current layer)
        let lookahead = ((num_slots / 3).saturating_sub(1)).max(1) as u32;
        *prefetch_state.expert_pool.lock().unwrap() = Some(pool);
        let tensor_ptrs = controller.take_tensor_ptrs();
        let state_mut = unsafe { &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState) };
        state_mut.fused_tensor_ptrs = tensor_ptrs;
        state_mut.dense_ffn_lookahead = lookahead;

        prefetch_state
            .dense_ffn_streaming
            .store(true, std::sync::atomic::Ordering::Relaxed);
    } else {
        prefetch_state
            .keep_nvme_resident
            .store(keep_resident, std::sync::atomic::Ordering::Relaxed);
    }

    // Start multi-threaded I/O pool (opens per-worker F_NOCACHE fds)
    let num_io_workers = (num_performance_cores() as usize / 2).clamp(2, 4);
    prefetch_state.start_io_pool(num_io_workers)?;

    // Only preload when the full model fits with 6 GB headroom.
    // For "barely overflows" models (Mixtral on 32GB), skip preloading —
    // layers will be loaded lazily via ensure_layer_loaded on first use,
    // then kept resident (not released) for subsequent tokens.
    // Expert-streaming never preloads — experts are loaded on demand.
    let any_streaming = expert_streaming || dense_ffn_streaming;
    if !any_streaming && should_preload {
        prefetch_state.preload_ram_layers();
    } else if keep_resident && !any_streaming {
        tracing::info!(
            "Skipping preload: model {:.1} GB exceeds preload threshold {:.1} GB (will load lazily)",
            model_total_bytes as f64 / (1u64 << 30) as f64,
            (total_ram.saturating_sub(6 * (1 << 30))) as f64 / (1u64 << 30) as f64,
        );
    }

    let nvme_tensor_count = prefetch_state.tensor_map.len();
    let nvme_layer_count = prefetch_state.layer_regions.len();
    tracing::info!("Prefetch state: {nvme_tensor_count} tensors across {nvme_layer_count} layers");

    let callback_state = Box::new(InferenceCallbackState {
        prefetch_state: Some(prefetch_state.clone()),
        turboquant: turboquant_session.clone(),
    });
    let state_ptr = callback_state.as_ref() as *const InferenceCallbackState as *mut c_void;

    let kv_quant = plan.kv_cache_plan.kv_quantization;
    // Expert-streaming with use_mmap=false + pool buffer keeps Metal working set small
    // enough for full batch size. No n_batch reduction needed.
    let effective_batch = config.n_batch;
    let mut ctx = LlamaContext::new_with_callback_and_kv_and_options(
        &model,
        config.n_ctx,
        effective_batch,
        config.n_threads,
        Some(eval_callback_with_runtime),
        state_ptr,
        kv_quant,
        turboquant_session.is_some(),
    )?;

    let mut sampler = LlamaSampler::new(&config.sampling);

    // KV cache manager for windowed compaction (Phase 3b)
    let mut kv_manager = if plan.kv_cache_plan.warm_window_tokens > 0 {
        Some(crate::cache::kv_cache::KvCacheManager::new(
            plan.kv_cache_plan.hot_window_tokens,
        ))
    } else {
        None
    };

    // Tokenize prompt
    let tokens = model.tokenize(prompt, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Enable I/O tracing for streaming/expert-streaming diagnostics.
    if !keep_resident || any_streaming {
        prefetch_state.enable_trace();
    }

    // Hybrid residency: keep first N layers' FFN permanently in RAM.
    // Uses the unified MemoryBudget (already accounts for GPU, KV, Metal, OS, pool).
    if dense_ffn_streaming {
        let pool_size = prefetch_state
            .expert_pool
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |p| p.pool_size as u64);
        let budget = MemoryBudget::compute(
            total_ram,
            gpu_bytes,
            false, // use_mmap=false for dense FFN streaming
            num_layers,
            num_kv_heads,
            head_dim,
            config.n_ctx,
            plan.kv_cache_plan.kv_quantization,
        )
        .with_pool(pool_size);
        let resident_budget = budget.available;

        // Compute per-layer FFN size from first layer with layouts
        let per_layer_ffn: u64 = prefetch_state
            .dense_ffn_layouts
            .values()
            .next()
            .map(|layouts| layouts.iter().map(|l| l.size as u64).sum())
            .unwrap_or(0);

        if per_layer_ffn > 0 {
            let max_resident = (resident_budget / per_layer_ffn) as u32;
            // Cap conservatively: Metal needs significant headroom for compute
            // buffers, KV cache, and page cache. On 32 GB M1 Max, 10+ resident
            // layers causes memory pressure that slows Metal compute more than
            // the I/O savings. Scale with available memory.
            let num_resident = max_resident.min(num_layers / 4);
            // Only activate residency when available memory is >50% of total RAM.
            // On 32 GB M1 Max, residency causes memory pressure that slows Metal
            // compute more than the I/O savings. Needs 64 GB+ to be beneficial.
            // Residency needs abundant headroom — Metal compute buffers, KV cache,
            // and page cache all compete with resident data. On 32 GB, even 7 GB
            // of resident data causes measurable compute slowdown.
            let min_available_for_residency = total_ram * 75 / 100;
            if num_resident >= 4 && budget.available > min_available_for_residency {
                if let Some((base, size, layers, offsets)) =
                    prefetch_state.activate_resident_ffn(num_resident)
                {
                    let state_mut =
                        unsafe { &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState) };
                    state_mut.resident_ffn_base = base;
                    state_mut.resident_ffn_size = size;
                    state_mut.resident_ffn_layers = layers;
                    state_mut.resident_ffn_offsets = offsets;
                }
            }
        }
    }

    // Eagerly prefetch NVMe layers before the first forward pass.
    if expert_streaming {
        prefetch_state.warm_cache_from_coactivation();
    } else if dense_ffn_streaming {
        // Pre-load the first streaming layers' FFN data so the initial eval_callback
        // doesn't stall. Skip resident layers (already loaded).
        let first_streaming = prefetch_state.resident_ffn_layers.len() as u32;
        for layer in first_streaming..(first_streaming + 4).min(num_layers) {
            if prefetch_state.dense_ffn_layouts.contains_key(&layer) {
                prefetch_state.prefetch_dense_ffn(layer);
            }
        }
    } else {
        prefetch_state.prefetch_all_nvme();
    }

    if keep_resident && !any_streaming {
        // Keep-resident mode: disable the eval callback immediately. Layer data
        // is provided by llama.cpp's own mmap mechanism (use_mmap=true) — our
        // buffer's posix_memalign pages are overlaid with mmap file pages during
        // model loading. Calling pread would REPLACE these efficient file-backed
        // pages with anonymous pages, increasing memory pressure.
        //
        // The prefetch_all_nvme above still runs to ensure NVMe layers are loaded
        // via pread (those might not have mmap pages committed). The callback is
        // disabled so it won't block the forward pass.
        tracing::info!("Keep-resident mode; disabling eval callback");
        prefetch_state
            .prefetch_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // Process prompt
    let prompt_start = Instant::now();
    let prompt_batch = effective_batch as usize;
    for chunk in tokens.chunks(prompt_batch) {
        if let Some(ref session) = turboquant_session {
            session.begin_batch(chunk);
        }
        ctx.decode(chunk)?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Set KV cache manager position after prompt processing
    if let Some(ref mut kv_mgr) = kv_manager {
        kv_mgr.set_position(prompt_len);
    }

    // Generation loop with active prefetch/release
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..config.sampling.max_tokens {
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = model.is_eog(token_id);
        let piece = model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        if token_tx
            .send(GeneratedToken {
                text: piece,
                token_id,
                tok_per_sec,
                is_eog,
            })
            .is_err()
        {
            break;
        }

        if is_eog {
            break;
        }

        // KV cache compaction for long-context inference
        if let Some(ref mut kv_mgr) = kv_manager {
            kv_mgr.advance(&ctx);
        }

        // In standard streaming mode, request NVMe layers before next forward pass.
        // Keep-resident and expert-streaming modes don't need this — keep-resident
        // has all data loaded, expert-streaming loads experts via eval_callback.
        if !keep_resident && !any_streaming {
            prefetch_state.prefetch_all_nvme();
        }

        // Dense FFN-streaming: pre-submit early STREAMING layer prefetches before the
        // next forward pass. At the token boundary, the I/O pipe goes idle. Priming
        // the first streaming layers keeps the pipe busy through the sampling gap.
        // Resident layers are skipped (always ready).
        if dense_ffn_streaming {
            let first_streaming = prefetch_state.resident_ffn_layers.len() as u32;
            let status = prefetch_state.layer_status.lock().unwrap();
            let mut to_prefetch = Vec::new();
            for layer in first_streaming..(first_streaming + 4).min(num_layers) {
                if !prefetch_state.dense_ffn_layouts.contains_key(&layer) {
                    continue;
                }
                let s = status.get(&layer).copied();
                if s != Some(LayerStatus::Loaded) && s != Some(LayerStatus::Loading) {
                    to_prefetch.push(layer);
                }
            }
            drop(status);
            for layer in to_prefetch {
                prefetch_state.prefetch_dense_ffn(layer);
            }
        }

        let decode_start = std::time::Instant::now();
        if let Some(ref session) = turboquant_session {
            session.begin_batch(&[token_id]);
        }
        ctx.decode(&[token_id])?;
        if let Some(ref session) = turboquant_session {
            session.end_batch();
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        prefetch_state.record_decode(decode_ms);
    }

    // Print I/O trace summary (streaming mode only)
    prefetch_state.print_trace_summary();

    // Stop I/O pool and clean up
    prefetch_state.stop_io_pool();

    if let Some(ref session) = turboquant_session {
        let stats = session.callback_hits();
        tracing::info!(
            "TurboQuant NVMe session complete: path={}, mode={}, q_hits={}, k_hits={}, v_hits={}, kq_hits={}, kq_soft_max_hits={}, kqv_hits={}, kq_overwrites={}, kqv_overwrites={}",
            session.runtime_path().as_str(),
            session.mode(),
            stats.q_hits,
            stats.k_hits,
            stats.v_hits,
            stats.kq_hits,
            stats.kq_soft_max_hits,
            stats.kqv_hits,
            stats.kq_overwrites,
            stats.kqv_overwrites,
        );
    }

    let perf = ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Query total physical RAM.
///
/// macOS: `hw.memsize` sysctl.
/// Linux/Windows: `sysinfo` crate (no privileged API needed).
fn total_physical_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let total = unsafe {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let name = b"hw.memsize\0";
            libc::sysctlbyname(
                name.as_ptr() as *const i8,
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len as *mut usize,
                std::ptr::null_mut(),
                0,
            );
            size
        };
        if total == 0 {
            32 * (1 << 30)
        } else {
            total
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let total = sys.total_memory();
        if total == 0 {
            16 * (1 << 30)
        } else {
            total
        }
    }
}

fn num_performance_cores() -> i32 {
    // macOS: use hw.perflevel0 (P-cores only)
    #[cfg(target_os = "macos")]
    {
        crate::profiler::cpu::sysctl_u32("hw.perflevel0.logicalcpu")
            .map(|n| n as i32)
            .unwrap_or_else(|_| {
                std::thread::available_parallelism()
                    .map(|n| (n.get() / 2).max(1) as i32)
                    .unwrap_or(4)
            })
    }
    // Non-macOS: use half of logical CPUs as a conservative estimate for I/O threads
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1) as i32)
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::model::gguf::{GgmlType, TensorInfo};

    fn make_gguf(layers: u32, tensors_per_layer: u32) -> GgufFile {
        let mut tensors = Vec::new();
        for l in 0..layers {
            for i in 0..tensors_per_layer {
                let name = format!("blk.{l}.tensor_{i}.weight");
                tensors.push(TensorInfo {
                    name,
                    dimensions: vec![4096, 4096],
                    dtype: GgmlType::Q4K,
                    offset: 0,
                    size_bytes: 1 << 26,
                    layer_index: Some(l),
                });
            }
        }
        GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors,
            data_offset: 0,
        }
    }

    fn make_plan(assignments: HashMap<String, StorageTier>) -> PlacementPlan {
        PlacementPlan {
            model_id: "test".into(),
            hardware_profile_hash: "".into(),
            tier_assignments: assignments,
            prefetch_schedule: PrefetchSchedule {
                layer_prefetches: vec![],
            },
            estimated_tok_per_sec: 0.0,
            estimated_time_to_first_token: 0.0,
            kv_cache_plan: KvCachePlan {
                hot_window_tokens: 0,
                warm_window_tokens: 0,
                hot_tier: StorageTier::Gpu,
                warm_tier: StorageTier::Ram,
                hot_bytes: 0,
                warm_bytes: 0,
                kv_quantization: None,
            },
            experience_tier: ExperienceTier::Fast,
            inference_mode: InferenceMode::FullStreaming,
        }
    }

    #[test]
    fn test_gpu_layers_offloads_all() {
        // On unified memory, all layers should be GPU-offloaded
        // regardless of individual tensor tier assignments
        let gguf = make_gguf(10, 3);
        let mut assignments = HashMap::new();
        for t in &gguf.tensors {
            assignments.insert(t.name.clone(), StorageTier::Gpu);
        }
        let plan = make_plan(assignments);
        // 10 layers (0-9) + 1 output = 11
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 11);
    }

    #[test]
    fn test_gpu_layers_mixed_tiers_stops_at_nvme() {
        // Only offload layers before the first NVMe layer
        // Layers 0-5 on GPU, layers 6-9 on NVMe
        let gguf = make_gguf(10, 3);
        let mut assignments = HashMap::new();
        for t in &gguf.tensors {
            let tier = if t.layer_index.unwrap() < 6 {
                StorageTier::Gpu
            } else {
                StorageTier::Nvme
            };
            assignments.insert(t.name.clone(), tier);
        }
        let plan = make_plan(assignments);
        // Layers 0-5 on GPU (6 layers) + 1 output = 7
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 7);
    }

    #[test]
    fn test_gpu_layers_empty() {
        let gguf = GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors: vec![],
            data_offset: 0,
        };
        let plan = make_plan(HashMap::new());
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 0);
    }

    #[test]
    fn runtime_tensor_component_maps_attention_nodes() {
        assert_eq!(
            runtime_tensor_component("Qcur-3"),
            Some(RuntimeTensorComponent::Query)
        );
        assert_eq!(
            runtime_tensor_component("Kcur-3"),
            Some(RuntimeTensorComponent::Key)
        );
        assert_eq!(
            runtime_tensor_component("Vcur-3"),
            Some(RuntimeTensorComponent::Value)
        );
        assert_eq!(
            runtime_tensor_component("kq-3"),
            Some(RuntimeTensorComponent::Score)
        );
        assert_eq!(
            runtime_tensor_component("kq_soft_max-3"),
            Some(RuntimeTensorComponent::ScoreSoftmax)
        );
        assert_eq!(
            runtime_tensor_component("kqv-3"),
            Some(RuntimeTensorComponent::ValueOutput)
        );
    }

    #[test]
    fn tensor_vector_range_accounts_for_streams() {
        let shape = RuntimeTensorShape {
            dim0: 4,
            dim1: 2,
            dim2: 3,
            dim3: 2,
        };

        assert_eq!(tensor_vector_range(shape, 0, 0, 0), 0..4);
        assert_eq!(tensor_vector_range(shape, 1, 0, 0), 4..8);
        assert_eq!(tensor_vector_range(shape, 0, 1, 0), 8..12);
        assert_eq!(tensor_vector_range(shape, 0, 0, 1), 24..28);
    }
}
