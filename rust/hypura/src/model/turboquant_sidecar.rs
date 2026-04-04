use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TurboQuantMode {
    Exact,
    PaperKeyOnly,
    PaperFullKv,
    ResearchKvSplit,
}

impl TurboQuantMode {
    pub fn requires_sidecar(self) -> bool {
        !matches!(self, Self::Exact)
    }

    pub fn expected_schema_kind(self) -> Option<TurboQuantSchemaKind> {
        match self {
            Self::Exact => None,
            Self::PaperKeyOnly | Self::PaperFullKv => Some(TurboQuantSchemaKind::Paper),
            Self::ResearchKvSplit => Some(TurboQuantSchemaKind::Research),
        }
    }

    pub fn config_suffix(self) -> Option<&'static str> {
        match self.expected_schema_kind()? {
            TurboQuantSchemaKind::Paper => Some("turboquant_config.paper.json"),
            TurboQuantSchemaKind::Research => Some("turboquant_config.research.json"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::PaperKeyOnly => "paper-key-only",
            Self::PaperFullKv => "paper-full-kv",
            Self::ResearchKvSplit => "research-kv-split",
        }
    }
}

impl fmt::Display for TurboQuantMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurboQuantSchemaKind {
    Paper,
    Research,
}

impl TurboQuantSchemaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paper => "paper",
            Self::Research => "research",
        }
    }
}

impl fmt::Display for TurboQuantSchemaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Rotation policies for TurboQuant KV compression
/// Based on: random_haar, block_so8_static, block_so8_learned (zapabob/Turboquant-CUDA)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RotationPolicy {
    /// Random Haar orthogonal rotation (paper baseline)
    RandomHaar,
    /// Block-diagonal SO(8) static rotation
    BlockSo8Static,
    /// Block-diagonal SO(8) with learned rotations
    BlockSo8Learned,
    /// Triality vector view (standard basis)
    TrialityVector,
    /// Triality spinor+ view
    TrialitySpinorPlus,
    /// Triality spinor- view
    TrialitySpinorMinus,
}

impl RotationPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RandomHaar => "random_haar",
            Self::BlockSo8Static => "block_so8_static",
            Self::BlockSo8Learned => "block_so8_learned",
            Self::TrialityVector => "triality_vector",
            Self::TrialitySpinorPlus => "triality_spinor_plus",
            Self::TrialitySpinorMinus => "triality_spinor_minus",
        }
    }

    pub fn is_triality(&self) -> bool {
        matches!(
            self,
            Self::TrialityVector | Self::TrialitySpinorPlus | Self::TrialitySpinorMinus
        )
    }

    pub fn triality_view(&self) -> Option<&'static str> {
        match self {
            Self::TrialityVector => Some("vector"),
            Self::TrialitySpinorPlus => Some("spinor_plus_proxy"),
            Self::TrialitySpinorMinus => Some("spinor_minus_proxy"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationArtifact {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    pub matrix: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizerArtifact {
    pub centroids: Vec<f32>,
    #[serde(default)]
    pub decision_boundaries: Vec<f32>,
    #[serde(default)]
    pub protected_centroids: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualQjlArtifact {
    pub sketch_dim: u32,
    pub scale: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub projections: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTurboQuantConfig {
    pub schema_kind: TurboQuantSchemaKind,
    pub codec: String,
    pub num_layers: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub key_bits: f32,
    #[serde(default)]
    pub value_bits: Option<f32>,
    #[serde(default)]
    pub mixed_bits: Option<f32>,
    #[serde(default)]
    pub value_exact: bool,
    pub rotation: RotationArtifact,
    pub scalar_quantizer: ScalarQuantizerArtifact,
    #[serde(default)]
    pub residual_qjl: Option<ResidualQjlArtifact>,
    #[serde(default)]
    pub protected_channels: Vec<u32>,
    #[serde(default)]
    pub outlier_channels: Vec<u32>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResearchTurboQuantConfig {
    #[serde(default)]
    pub schema_kind: Option<TurboQuantSchemaKind>,
    #[serde(default)]
    pub codec: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurboQuantSidecarConfig {
    Paper(PaperTurboQuantConfig),
    Research(ResearchTurboQuantConfig),
}

impl TurboQuantSidecarConfig {
    pub fn schema_kind(&self) -> TurboQuantSchemaKind {
        match self {
            Self::Paper(_) => TurboQuantSchemaKind::Paper,
            Self::Research(_) => TurboQuantSchemaKind::Research,
        }
    }

    pub fn paper(&self) -> Option<&PaperTurboQuantConfig> {
        match self {
            Self::Paper(cfg) => Some(cfg),
            Self::Research(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTurboQuantConfig {
    pub mode: TurboQuantMode,
    pub schema_kind: Option<TurboQuantSchemaKind>,
    pub source_path: Option<PathBuf>,
    pub config: Option<TurboQuantSidecarConfig>,
    pub gguf_metadata: Option<GgufTurboQuantConfig>,
}

impl ResolvedTurboQuantConfig {
    pub fn exact() -> Self {
        Self {
            mode: TurboQuantMode::Exact,
            schema_kind: None,
            source_path: None,
            config: None,
            gguf_metadata: None,
        }
    }

    pub fn schema_label(&self) -> &'static str {
        self.schema_kind
            .map(TurboQuantSchemaKind::as_str)
            .unwrap_or("none")
    }

    pub fn source_label(&self) -> String {
        self.source_path
            .as_ref()
            .map(|p| p.display().to_string())
            .or_else(|| self.gguf_metadata.as_ref().map(|cfg| cfg.source_label().to_string()))
            .unwrap_or_else(|| "none".to_string())
    }

    pub fn paper_config(&self) -> Option<&PaperTurboQuantConfig> {
        self.config.as_ref()?.paper()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTurboQuantConfig {
    pub enabled: bool,
    pub mode: TurboQuantMode,
    pub rotation_policy: Option<RotationPolicy>,
    pub triality_view: Option<String>,
    pub triality_mix: Option<f32>,
    pub rotation_seed: u32,
    pub artifact_path: Option<String>,
    pub head_dim: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
}

impl GgufTurboQuantConfig {
    pub fn source_label(&self) -> &'static str {
        "gguf-metadata"
    }
}

pub fn resolve_turboquant_config(
    model_path: &Path,
    metadata: &ModelMetadata,
    gguf: &GgufFile,
    mode: TurboQuantMode,
    explicit_config: Option<&Path>,
) -> anyhow::Result<ResolvedTurboQuantConfig> {
    if mode == TurboQuantMode::Exact {
        let mut resolved = ResolvedTurboQuantConfig::exact();
        resolved.gguf_metadata = read_gguf_turboquant_config(gguf, metadata);
        return Ok(resolved);
    }

    let config_path = match explicit_config {
        Some(path) => path.to_path_buf(),
        None => match auto_discover_sidecar_path(model_path, mode) {
            Some(path) => path,
            None => {
                if let Some(gguf_metadata) = read_gguf_turboquant_config(gguf, metadata) {
                    anyhow::ensure!(
                        gguf_metadata.mode == mode,
                        "GGUF TurboQuant metadata requests mode `{}`, but CLI requested `{mode}`",
                        gguf_metadata.mode
                    );
                    return Ok(ResolvedTurboQuantConfig {
                        mode,
                        schema_kind: None,
                        source_path: None,
                        config: None,
                        gguf_metadata: Some(gguf_metadata),
                    });
                }
                if mode == TurboQuantMode::ResearchKvSplit {
                    tracing::warn!(
                        "No TurboQuant research sidecar or GGUF metadata found next to {}. Falling back to exact runtime.",
                        model_path.display()
                    );
                    return Ok(ResolvedTurboQuantConfig::exact());
                }
                return Err(anyhow::anyhow!(
                    "No TurboQuant sidecar found for mode `{mode}` next to {}. \
                     Pass `--turboquant-config <path>` or place a matching sidecar beside the model.",
                    model_path.display()
                ));
            }
        },
    };

    anyhow::ensure!(
        config_path.exists(),
        "TurboQuant sidecar for mode `{mode}` not found: {}",
        config_path.display()
    );

    let json = std::fs::read_to_string(&config_path).with_context(|| {
        format!(
            "Failed to read TurboQuant sidecar {} for mode `{mode}`",
            config_path.display()
        )
    })?;
    let config = parse_sidecar_config(&config_path, &json, mode).with_context(|| {
        format!(
            "Invalid TurboQuant sidecar {} for mode `{mode}`",
            config_path.display()
        )
    })?;
    let schema_kind = config.schema_kind();

    validate_mode_schema(mode, schema_kind, &config_path)?;
    validate_config_against_metadata(&config, metadata, &config_path)?;

    Ok(ResolvedTurboQuantConfig {
        mode,
        schema_kind: Some(schema_kind),
        source_path: Some(config_path),
        config: Some(config),
        gguf_metadata: read_gguf_turboquant_config(gguf, metadata),
    })
}

pub fn read_gguf_turboquant_config(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> Option<GgufTurboQuantConfig> {
    let enabled = gguf.get_bool("hypura.turboquant.enabled")?;
    if !enabled {
        return None;
    }

    let mode = match gguf.get_string("hypura.turboquant.mode")? {
        "exact" => TurboQuantMode::Exact,
        "paper-key-only" => TurboQuantMode::PaperKeyOnly,
        "paper-full-kv" => TurboQuantMode::PaperFullKv,
        "research-kv-split" => TurboQuantMode::ResearchKvSplit,
        _ => return None,
    };

    let rotation_policy = gguf
        .get_string("hypura.turboquant.rotation_policy")
        .and_then(|value| match value {
            "random_haar" => Some(RotationPolicy::RandomHaar),
            "block_so8_static" => Some(RotationPolicy::BlockSo8Static),
            "block_so8_learned" => Some(RotationPolicy::BlockSo8Learned),
            "triality_vector" => Some(RotationPolicy::TrialityVector),
            "triality_spinor_plus" => Some(RotationPolicy::TrialitySpinorPlus),
            "triality_spinor_minus" => Some(RotationPolicy::TrialitySpinorMinus),
            _ => None,
        });
    let triality_view = gguf
        .get_string("hypura.turboquant.triality_view")
        .map(ToOwned::to_owned)
        .or_else(|| rotation_policy.and_then(|policy| policy.triality_view().map(str::to_string)));
    let head_dim = if metadata.num_heads == 0 {
        0
    } else {
        metadata.embedding_dim / metadata.num_heads
    };

    Some(GgufTurboQuantConfig {
        enabled,
        mode,
        rotation_policy,
        triality_view,
        triality_mix: gguf.get_f32("hypura.turboquant.triality_mix"),
        rotation_seed: gguf.get_u32("hypura.turboquant.rotation_seed").unwrap_or(0),
        artifact_path: gguf
            .get_string("hypura.turboquant.artifact")
            .map(ToOwned::to_owned),
        head_dim,
        num_layers: metadata.num_layers,
        num_kv_heads: metadata.num_kv_heads,
    })
}

pub fn auto_discover_sidecar_path(model_path: &Path, mode: TurboQuantMode) -> Option<PathBuf> {
    let suffix = mode.config_suffix()?;
    let candidate = model_path.with_extension(suffix);
    candidate.exists().then_some(candidate)
}

pub fn parse_sidecar_config(
    config_path: &Path,
    json: &str,
    _mode: TurboQuantMode,
) -> anyhow::Result<TurboQuantSidecarConfig> {
    let value: Value = serde_json::from_str(json).with_context(|| {
        format!(
            "TurboQuant sidecar {} must contain valid JSON",
            config_path.display()
        )
    })?;
    let schema_kind = detect_schema_kind(config_path, &value)?;

    Ok(match schema_kind {
        TurboQuantSchemaKind::Paper => {
            let config: PaperTurboQuantConfig =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "TurboQuant sidecar {} could not be parsed as a paper config",
                        config_path.display()
                    )
                })?;
            validate_paper_artifacts(&config, config_path)?;
            TurboQuantSidecarConfig::Paper(config)
        }
        TurboQuantSchemaKind::Research => {
            let mut config: ResearchTurboQuantConfig =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "TurboQuant sidecar {} could not be parsed as a research config",
                        config_path.display()
                    )
                })?;
            config.schema_kind = Some(TurboQuantSchemaKind::Research);
            TurboQuantSidecarConfig::Research(config)
        }
    })
}

fn detect_schema_kind(
    config_path: &Path,
    value: &serde_json::Value,
) -> anyhow::Result<TurboQuantSchemaKind> {
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!(
            "TurboQuant sidecar {} must be a JSON object",
            config_path.display()
        )
    })?;

    let (schema_key, raw_kind) = ["schema_kind", "kind", "schema"]
        .into_iter()
        .find_map(|key| object.get(key).map(|value| (key, value)))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "TurboQuant sidecar {} is missing `schema_kind` (expected `paper` or `research`)",
                config_path.display()
            )
        })?;

    let raw_kind = raw_kind.as_str().ok_or_else(|| {
        anyhow::anyhow!(
            "TurboQuant sidecar {} field `{schema_key}` must be a string",
            config_path.display()
        )
    })?;

    match raw_kind {
        "paper" => Ok(TurboQuantSchemaKind::Paper),
        "research" => Ok(TurboQuantSchemaKind::Research),
        other => Err(anyhow::anyhow!(
            "TurboQuant sidecar {} has unsupported schema kind `{other}` (expected `paper` or `research`)",
            config_path.display()
        )),
    }
}

fn validate_paper_artifacts(cfg: &PaperTurboQuantConfig, config_path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cfg.codec.trim().is_empty(),
        "TurboQuant sidecar {} field `codec` must not be empty",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.num_layers > 0 && cfg.num_kv_heads > 0 && cfg.head_dim > 0,
        "TurboQuant sidecar {} layer/head dimensions must be greater than zero",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.key_bits > 0.0,
        "TurboQuant sidecar {} field `key_bits` must be greater than zero",
        config_path.display()
    );
    anyhow::ensure!(
        !cfg.rotation.matrix.is_empty(),
        "TurboQuant sidecar {} must include a non-empty `rotation.matrix` artifact",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.rotation.matrix.len() == (cfg.head_dim * cfg.head_dim) as usize,
        "TurboQuant sidecar {} rotation.matrix length={} does not match head_dim^2={}",
        config_path.display(),
        cfg.rotation.matrix.len(),
        cfg.head_dim * cfg.head_dim
    );
    anyhow::ensure!(
        !cfg.scalar_quantizer.centroids.is_empty(),
        "TurboQuant sidecar {} must include scalar quantizer centroids",
        config_path.display()
    );
    if let Some(residual_qjl) = &cfg.residual_qjl {
        anyhow::ensure!(
            residual_qjl.sketch_dim > 0,
            "TurboQuant sidecar {} residual_qjl.sketch_dim must be greater than zero",
            config_path.display()
        );
        if !residual_qjl.projections.is_empty() {
            anyhow::ensure!(
                residual_qjl.projections.len()
                    == residual_qjl.sketch_dim as usize * cfg.head_dim as usize,
                "TurboQuant sidecar {} residual_qjl.projections length={} does not match sketch_dim*head_dim={}",
                config_path.display(),
                residual_qjl.projections.len(),
                residual_qjl.sketch_dim as usize * cfg.head_dim as usize
            );
        }
    }

    Ok(())
}

fn validate_mode_schema(
    mode: TurboQuantMode,
    schema_kind: TurboQuantSchemaKind,
    config_path: &Path,
) -> anyhow::Result<()> {
    let expected = mode.expected_schema_kind().ok_or_else(|| {
        anyhow::anyhow!("TurboQuant validation requested for exact mode unexpectedly")
    })?;

    anyhow::ensure!(
        expected == schema_kind,
        "TurboQuant sidecar {} declares `{schema_kind}` schema, but mode `{mode}` requires `{expected}`",
        config_path.display()
    );

    Ok(())
}

fn validate_config_against_metadata(
    config: &TurboQuantSidecarConfig,
    metadata: &ModelMetadata,
    config_path: &Path,
) -> anyhow::Result<()> {
    match config {
        TurboQuantSidecarConfig::Paper(cfg) => {
            anyhow::ensure!(
                cfg.num_layers == metadata.num_layers as u64,
                "TurboQuant sidecar {} num_layers={} does not match model num_layers={}",
                config_path.display(),
                cfg.num_layers,
                metadata.num_layers
            );

            anyhow::ensure!(
                cfg.num_kv_heads == metadata.num_kv_heads as u64,
                "TurboQuant sidecar {} num_kv_heads={} does not match model num_kv_heads={}",
                config_path.display(),
                cfg.num_kv_heads,
                metadata.num_kv_heads
            );

            let model_head_dim = if metadata.num_heads > 0 {
                metadata.embedding_dim as u64 / metadata.num_heads as u64
            } else {
                0
            };
            anyhow::ensure!(
                cfg.head_dim == model_head_dim,
                "TurboQuant sidecar {} head_dim={} does not match model head_dim={}",
                config_path.display(),
                cfg.head_dim,
                model_head_dim
            );
        }
        TurboQuantSidecarConfig::Research(cfg) => {
            if let Some(schema_kind) = cfg.schema_kind {
                anyhow::ensure!(
                    schema_kind == TurboQuantSchemaKind::Research,
                    "TurboQuant sidecar {} research config must declare `research` schema",
                    config_path.display()
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::gguf::GgufValue;

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "qwen3".into(),
            parameter_count: 9_000_000_000,
            context_length: 4096,
            quantization: Some("q4".into()),
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            embedding_dim: 16,
            vocab_size: 32_000,
            is_moe: false,
            num_experts: None,
            num_experts_used: None,
        }
    }

    fn sample_paper_json() -> String {
        serde_json::json!({
            "schema_kind": "paper",
            "codec": "turboquant-prod",
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "key_bits": 3.5,
            "value_bits": 3.5,
            "mixed_bits": 3.5,
            "value_exact": false,
            "rotation": {
                "kind": "matrix",
                "seed": 7,
                "matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]
            },
            "scalar_quantizer": {
                "centroids": [-1.0, -0.25, 0.25, 1.0],
                "decision_boundaries": [-0.5, 0.0, 0.5]
            },
            "residual_qjl": {
                "sketch_dim": 2,
                "scale": 1.0,
                "seed": 123,
                "projections": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0
                ]
            },
            "protected_channels": [1],
            "outlier_channels": [1]
        })
        .to_string()
    }

    fn sample_gguf() -> GgufFile {
        GgufFile {
            version: 3,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    #[test]
    fn paper_config_parse_and_validate() {
        let path = Path::new("model.turboquant_config.paper.json");
        let parsed =
            parse_sidecar_config(path, &sample_paper_json(), TurboQuantMode::PaperKeyOnly).unwrap();
        let metadata = sample_metadata();
        validate_config_against_metadata(&parsed, &metadata, path).unwrap();
        let paper = parsed.paper().unwrap();
        assert_eq!(paper.codec, "turboquant-prod");
        assert_eq!(paper.rotation.matrix.len(), 16);
        assert_eq!(paper.scalar_quantizer.centroids.len(), 4);
    }

    #[test]
    fn schema_mode_mismatch_fails() {
        let path = Path::new("model.turboquant_config.paper.json");
        let parsed =
            parse_sidecar_config(path, &sample_paper_json(), TurboQuantMode::PaperKeyOnly).unwrap();
        let err = validate_mode_schema(TurboQuantMode::ResearchKvSplit, parsed.schema_kind(), path)
            .unwrap_err();
        assert!(err.to_string().contains("requires `research`"));
    }

    #[test]
    fn exact_mode_ignores_sidecar() {
        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &sample_gguf(),
            TurboQuantMode::Exact,
            None,
        )
        .unwrap();
        assert_eq!(resolved.mode, TurboQuantMode::Exact);
        assert!(resolved.config.is_none());
    }

    #[test]
    fn gguf_triality_metadata_resolves_without_sidecar() {
        let mut gguf = sample_gguf();
        gguf.metadata.insert(
            "hypura.turboquant.enabled".into(),
            GgufValue::Bool(true),
        );
        gguf.metadata.insert(
            "hypura.turboquant.mode".into(),
            GgufValue::String("research-kv-split".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.rotation_policy".into(),
            GgufValue::String("triality_spinor_plus".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.rotation_seed".into(),
            GgufValue::Uint32(17),
        );
        gguf.metadata.insert(
            "hypura.turboquant.triality_mix".into(),
            GgufValue::Float32(0.75),
        );

        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
        )
        .unwrap();

        let gguf_cfg = resolved.gguf_metadata.expect("gguf metadata should be attached");
        assert_eq!(gguf_cfg.mode, TurboQuantMode::ResearchKvSplit);
        assert_eq!(gguf_cfg.rotation_policy, Some(RotationPolicy::TrialitySpinorPlus));
        assert_eq!(gguf_cfg.triality_view.as_deref(), Some("spinor_plus_proxy"));
        assert_eq!(gguf_cfg.rotation_seed, 17);
    }
}
