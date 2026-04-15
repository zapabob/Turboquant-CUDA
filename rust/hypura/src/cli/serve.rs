use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use hypura::compute::inference::{self, LlamaTurboquantCliBridge};
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use hypura::server::ollama_types::GgufInfo;
use hypura::server::routes::{self, AppState};
use hypura::telemetry::metrics::TelemetryEmitter;
use indicatif::{ProgressBar, ProgressStyle};

use super::fmt_util::cli_progress_enabled;

fn set_process_env_var<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) {
    unsafe {
        std::env::set_var(key, value);
    }
}

pub fn run(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    _rotation_seed: u32,
    tq_so8_off: bool,
    tq_so8_learned: bool,
    tq_triality_off: bool,
    tq_triality_mix: f32,
    tq_rotation_seed: u32,
    tq_artifact: Option<&str>,
    model_dir: Option<&str>,
    ui_theme: &str,
) -> anyhow::Result<()> {
    let llama_bridge = LlamaTurboquantCliBridge {
        rotation_policy,
        llama_rotation_seed: tq_rotation_seed,
        tq_so8_off,
        tq_triality_off,
        tq_so8_learned,
        tq_triality_mix,
        tq_artifact: tq_artifact.map(|s| s.to_string()),
    };
    if !ui_theme.trim().is_empty() {
        set_process_env_var("HYPURA_UI_THEME", ui_theme.trim());
    }
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(
        model_path,
        host,
        port,
        context,
        turboquant_mode,
        turboquant_config,
        model_dir,
        llama_bridge,
    ))
}

async fn run_async(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    model_dir: Option<&str>,
    llama_bridge: LlamaTurboquantCliBridge,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);

    let pb_setup = if cli_progress_enabled() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
                .tick_chars("|/-\\ "),
        );
        pb.set_message("Analyzing GGUF, hardware profile, and placement...");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        pb
    } else {
        ProgressBar::hidden()
    };

    let runtime = inference::resolve_runtime_setup(
        path,
        context,
        turboquant_mode,
        turboquant_config.map(Path::new),
        llama_bridge.clone(),
    )?;
    if cli_progress_enabled() {
        pb_setup.finish_and_clear();
    }
    let file_size = std::fs::metadata(path)?.len();

    let gguf_info = GgufInfo {
        file_size,
        architecture: runtime.metadata.architecture.clone(),
        parameter_count: runtime.metadata.parameter_count,
        quantization: runtime
            .metadata
            .quantization
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        context_length: runtime.metadata.context_length,
    };

    let config = inference::InferenceConfig {
        n_ctx: context,
        ..inference::InferenceConfig::default()
    };

    let pb = if cli_progress_enabled() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
                .tick_chars("|/-\\ "),
        );
        pb.set_message("Loading model weights into memory (no ETA - size-dependent)...");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        pb
    } else {
        ProgressBar::hidden()
    };
    let load_start = Instant::now();

    let path_owned = path.to_path_buf();
    let plan_arc = Arc::new(runtime.plan.clone());
    let gguf_arc = Arc::new(runtime.gguf.clone());
    let turboquant = runtime.turboquant.clone();
    let plan_for_load = plan_arc.clone();
    let gguf_for_load = gguf_arc.clone();
    let n_gpu_layers = runtime.n_gpu_layers;

    let loaded = tokio::task::spawn_blocking(move || {
        inference::load_model(
            &path_owned,
            &config,
            n_gpu_layers,
            &plan_for_load,
            &gguf_for_load,
            &turboquant,
        )
    })
    .await??;

    if cli_progress_enabled() {
        pb.finish_and_clear();
    }
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("Model loaded in {load_secs:.1}s");

    let load_duration_ns = load_start.elapsed().as_nanos() as u64;
    let model_name = std::env::var("HYPURA_MODEL_NAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| loaded.model_name.clone());

    let telemetry = Arc::new(TelemetryEmitter::new(256));
    let init_theme = std::env::var("HYPURA_UI_THEME")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| matches!(s.as_str(), "light" | "dark" | "classic"))
        .unwrap_or_else(|| "classic".to_string());

    let serve_turboquant_config_path = turboquant_config.map(std::path::PathBuf::from);

    let state = Arc::new(AppState {
        loaded_model: Arc::new(std::sync::Mutex::new(loaded)),
        model_name: Arc::new(std::sync::Mutex::new(model_name.clone())),
        model_path: Arc::new(std::sync::Mutex::new(path.to_path_buf())),
        gguf_info: Arc::new(std::sync::Mutex::new(gguf_info)),
        model_dir: model_dir
            .map(std::path::PathBuf::from)
            .or_else(|| path.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
        default_context: context,
        load_duration_ns,
        telemetry,
        turboquant: runtime.turboquant,
        serve_turboquant_mode: turboquant_mode,
        serve_turboquant_config_path,
        serve_llama_bridge: llama_bridge,
        active_cancel: Arc::new(std::sync::Mutex::new(None)),
        generation_in_progress: Arc::new(AtomicBool::new(false)),
        gui_presets: Arc::new(std::sync::Mutex::new(HashMap::new())),
        gui_history: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        gui_events: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        ui_theme: Arc::new(std::sync::Mutex::new(init_theme)),
    });

    let app = routes::router(state.clone());
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura serving {model_name}");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/generate, /api/chat, /api/tags");
    if std::env::var("HYPURA_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some()
    {
        println!(
            "  API key: required (HYPURA_API_KEY); use Authorization: Bearer <key> or X-API-Key"
        );
    }
    println!(
        "  TurboQuant: mode={}, schema={}, config={}, runtime_status={}",
        state.turboquant.mode,
        state.turboquant.schema_label(),
        state.turboquant.source_label(),
        if state.turboquant.mode == hypura::model::turboquant_sidecar::TurboQuantMode::Exact {
            "inactive"
        } else if state.turboquant.mode
            == hypura::model::turboquant_sidecar::TurboQuantMode::PaperFullKv
        {
            "experimental-full-kv"
        } else {
            "faithful-attached"
        }
    );
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
