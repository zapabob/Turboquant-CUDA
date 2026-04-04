mod cli;

use clap::{Parser, Subcommand};
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};

#[derive(Parser)]
#[command(
    name = "hypura",
    version,
    about = "Storage-tier-aware LLM inference scheduler"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hardware profiler and save results
    Profile {
        /// Force re-profiling even if a recent profile exists
        #[arg(long)]
        force: bool,
    },
    /// Show performance estimate for a model without loading it
    Estimate {
        /// Path to model file or HuggingFace model ID
        model: String,
    },
    /// Load model with tiered scheduling and run inference
    Run {
        /// Path to model file
        model: String,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
        /// Single prompt (non-interactive mode)
        #[arg(long)]
        prompt: Option<String>,
        /// Interactive chat mode
        #[arg(long)]
        interactive: bool,
        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,
        /// TurboQuant runtime mode
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant
        #[arg(long, value_enum)]
        rotation_policy: Option<RotationPolicy>,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
    },
    /// Start Ollama-compatible API server
    Serve {
        /// Path to model file
        model: String,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
        /// TurboQuant runtime mode
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant
        #[arg(long, value_enum)]
        rotation_policy: Option<RotationPolicy>,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
        /// Disable SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_off: bool,
        /// Enable learned SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_learned: bool,
        /// Disable Triality runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_triality_off: bool,
        /// Triality mix coefficient for TurboQuant env bridge [0,1]
        #[arg(long, default_value = "0.5")]
        tq_triality_mix: f32,
        /// Rotation seed for TurboQuant runtime env bridge
        #[arg(long, default_value = "0")]
        tq_rotation_seed: u32,
        /// Optional TurboQuant artifact path for runtime env bridge
        #[arg(long)]
        tq_artifact: Option<String>,
        /// Optional model directory used by Kobold-lite model selector
        #[arg(long)]
        model_dir: Option<String>,
        /// Optional Kobold-lite theme hint (stored in env for UI)
        #[arg(long, default_value = "classic")]
        ui_theme: String,
    },
    /// Benchmark tok/s: Hypura scheduling vs naive mmap
    Bench {
        /// Path to model file
        model: String,
        /// Also benchmark with naive mmap for comparison
        #[arg(long)]
        baseline: bool,
        /// Maximum context length
        #[arg(long, default_value = "2048")]
        context: u32,
        /// Tokens to generate per run
        #[arg(long, default_value = "128")]
        max_tokens: u32,
        /// Prompt text
        #[arg(long)]
        prompt: Option<String>,
        /// Force unsafe operations (e.g. baseline with model larger than RAM)
        #[arg(long)]
        force: bool,
        /// TurboQuant runtime mode for the Hypura run
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant
        #[arg(long, value_enum)]
        rotation_policy: Option<RotationPolicy>,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
    },
    /// Print model metadata, tensor list, and placement plan
    Inspect {
        /// Path to model file
        model: String,
        /// Show individual tensor details
        #[arg(long)]
        tensors: bool,
    },
    /// Low-level NVMe I/O microbenchmark (diagnostic)
    Iobench {
        /// Path to a GGUF model file
        model: String,
        /// Amount of data to read in each test (GiB)
        #[arg(long, default_value = "1.0")]
        read_gb: f64,
    },
    /// (MoE only) Reorganize expert layout on disk for sequential access
    Optimize {
        /// Path to model file
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("HYPURA_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Profile { force } => cli::profile::run(force),
        Commands::Estimate { model } => cli::estimate::run(&model),
        Commands::Run {
            model,
            context,
            prompt,
            interactive,
            max_tokens,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
        } => cli::run::run(
            &model,
            context,
            prompt.as_deref(),
            interactive,
            max_tokens,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy.map(|p| p.as_str().to_string()).as_deref(),
            rotation_seed,
        ),
        Commands::Serve {
            model,
            host,
            port,
            context,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact,
            model_dir,
            ui_theme,
        } => cli::serve::run(
            &model,
            &host,
            port,
            context,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy.map(|p| p.as_str().to_string()).as_deref(),
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact.as_deref(),
            model_dir.as_deref(),
            &ui_theme,
        ),
        Commands::Bench {
            model,
            baseline,
            context,
            max_tokens,
            prompt,
            force,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
        } => cli::bench::run(
            &model,
            baseline,
            context,
            max_tokens,
            prompt.as_deref(),
            force,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy.map(|p| p.as_str().to_string()).as_deref(),
            rotation_seed,
        ),
        Commands::Inspect { model, tensors } => cli::inspect::run(&model, tensors),
        Commands::Iobench { model, read_gb } => cli::iobench::run(&model, read_gb),
        Commands::Optimize { model } => cli::optimize::run(&model),
    }
}
