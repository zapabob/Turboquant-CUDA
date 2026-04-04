use std::collections::HashMap;
use std::ops::Range;

use crate::cache::kv_codec_python::TurboQuantCodec;
use crate::model::turboquant_sidecar::{
    PaperTurboQuantConfig, ResolvedTurboQuantConfig, TurboQuantMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvComponent {
    Key,
    Value,
}

pub trait KvCodec: Send {
    fn name(&self) -> &'static str;
    fn fork_session(&self) -> Box<dyn KvCodec + Send>;
    fn ingest_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>>;
    fn ingest_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>>;
    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>>;
    fn read_v(&self, layer: u32, head: u32, token_range: Range<u32>) -> anyhow::Result<Vec<f32>>;
}

pub fn build_kv_codec(
    resolved: &ResolvedTurboQuantConfig,
) -> anyhow::Result<Box<dyn KvCodec + Send>> {
    match resolved.mode {
        TurboQuantMode::Exact => Ok(Box::new(ExactKvCodec::default())),
        TurboQuantMode::PaperKeyOnly => Ok(Box::new(PaperKeyOnlyCodec::new(
            resolved.paper_config().ok_or_else(|| {
                anyhow::anyhow!("paper-key-only requires a parsed paper TurboQuant config")
            })?,
        )?)),
        TurboQuantMode::PaperFullKv => Ok(Box::new(PaperFullKvCodec::new(
            resolved.paper_config().ok_or_else(|| {
                anyhow::anyhow!("paper-full-kv requires a parsed paper TurboQuant config")
            })?,
        )?)),
        TurboQuantMode::ResearchKvSplit => Ok(Box::new(ResearchKvSplitCodec::new(resolved)?)),
    }
}

type KvStoreKey = (u32, u32, u32);

#[derive(Clone, Default)]
pub struct ExactKvCodec {
    key_vectors: HashMap<KvStoreKey, Vec<f32>>,
    value_vectors: HashMap<KvStoreKey, Vec<f32>>,
}

impl KvCodec for ExactKvCodec {
    fn name(&self) -> &'static str {
        "exact"
    }

    fn fork_session(&self) -> Box<dyn KvCodec + Send> {
        Box::new(self.clone())
    }

    fn ingest_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        self.key_vectors.insert((layer, head, token), data.to_vec());
        Ok(data.to_vec())
    }

    fn ingest_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        self.value_vectors
            .insert((layer, head, token), data.to_vec());
        Ok(data.to_vec())
    }

    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(token_range
            .map(|token| {
                self.key_vectors
                    .get(&(layer, head, token))
                    .map(|key| dot_product(query, key))
                    .unwrap_or(0.0)
            })
            .collect())
    }

    fn read_v(&self, layer: u32, head: u32, token_range: Range<u32>) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        for token in token_range {
            if let Some(value) = self.value_vectors.get(&(layer, head, token)) {
                out.extend_from_slice(value);
            }
        }
        Ok(out)
    }
}

#[derive(Clone)]
struct EncodedTurboVector {
    mse_reconstruction: Vec<f32>,
    residual_signs: Vec<i8>,
    protected_exact: Vec<(usize, f32)>,
}

#[derive(Clone)]
struct PaperArtifacts {
    head_dim: usize,
    rotation: Vec<f32>,
    centroids: Vec<f32>,
    protected_centroids: Vec<f32>,
    qjl_scale: Option<f32>,
    qjl_sketch_dim: usize,
    qjl_projections: Vec<f32>,
    protected_channels: Vec<usize>,
}

impl PaperArtifacts {
    fn from_config(config: &PaperTurboQuantConfig) -> anyhow::Result<Self> {
        let head_dim = config.head_dim as usize;
        anyhow::ensure!(
            head_dim > 0,
            "TurboQuant head_dim must be greater than zero"
        );
        anyhow::ensure!(
            config.rotation.matrix.len() == head_dim * head_dim,
            "TurboQuant rotation artifact length {} does not match head_dim^2 {}",
            config.rotation.matrix.len(),
            head_dim * head_dim
        );
        anyhow::ensure!(
            !config.scalar_quantizer.centroids.is_empty(),
            "TurboQuant scalar quantizer must define centroids"
        );

        let qjl_sketch_dim = config
            .residual_qjl
            .as_ref()
            .map(|qjl| qjl.sketch_dim as usize)
            .unwrap_or(0);
        let qjl_scale = config.residual_qjl.as_ref().map(|qjl| qjl.scale);
        let qjl_projections = if let Some(qjl) = &config.residual_qjl {
            if qjl.projections.is_empty() {
                seeded_projection_matrix(
                    qjl.seed.unwrap_or(0x9E37_79B9_7F4A_7C15),
                    qjl_sketch_dim,
                    head_dim,
                )
            } else {
                qjl.projections.clone()
            }
        } else {
            Vec::new()
        };

        anyhow::ensure!(
            qjl_projections.is_empty() || qjl_projections.len() == qjl_sketch_dim * head_dim,
            "TurboQuant residual QJL projection artifact does not match sketch_dim*head_dim"
        );

        Ok(Self {
            head_dim,
            rotation: config.rotation.matrix.clone(),
            centroids: config.scalar_quantizer.centroids.clone(),
            protected_centroids: if config.scalar_quantizer.protected_centroids.is_empty() {
                config.scalar_quantizer.centroids.clone()
            } else {
                config.scalar_quantizer.protected_centroids.clone()
            },
            qjl_scale,
            qjl_sketch_dim,
            qjl_projections,
            protected_channels: config
                .protected_channels
                .iter()
                .copied()
                .map(|idx| idx as usize)
                .filter(|idx| *idx < head_dim)
                .collect(),
        })
    }

    fn encode(&self, vector: &[f32]) -> EncodedTurboVector {
        let input = fit_dim(vector, self.head_dim);
        let norm = l2_norm(&input);
        if norm == 0.0 {
            return EncodedTurboVector {
                mse_reconstruction: vec![0.0; self.head_dim],
                residual_signs: vec![1; self.qjl_sketch_dim],
                protected_exact: Vec::new(),
            };
        }

        let unit: Vec<f32> = input.iter().map(|value| *value / norm).collect();
        let rotated = apply_rotation(&self.rotation, self.head_dim, &unit);

        let mut quantized = vec![0.0; self.head_dim];
        for (idx, value) in rotated.iter().enumerate() {
            let centroids = if self.protected_channels.contains(&idx) {
                &self.protected_centroids
            } else {
                &self.centroids
            };
            quantized[idx] = nearest_centroid(*value, centroids);
        }

        let mut mse_reconstruction =
            apply_rotation_transpose(&self.rotation, self.head_dim, &quantized);
        for value in &mut mse_reconstruction {
            *value *= norm;
        }

        let protected_exact = self
            .protected_channels
            .iter()
            .map(|&idx| (idx, input[idx]))
            .collect::<Vec<_>>();
        for &(idx, value) in &protected_exact {
            mse_reconstruction[idx] = value;
        }

        let residual = subtract_vectors(&input, &mse_reconstruction);
        let residual_signs = if self.qjl_sketch_dim == 0 {
            Vec::new()
        } else {
            (0..self.qjl_sketch_dim)
                .map(|row| {
                    let projection =
                        &self.qjl_projections[row * self.head_dim..(row + 1) * self.head_dim];
                    if dot_product(projection, &residual) >= 0.0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect()
        };

        EncodedTurboVector {
            mse_reconstruction,
            residual_signs,
            protected_exact,
        }
    }

    fn reconstruct(&self, encoded: &EncodedTurboVector) -> Vec<f32> {
        let mut out = encoded.mse_reconstruction.clone();
        if self.qjl_sketch_dim > 0 && !encoded.residual_signs.is_empty() {
            let scale = self.qjl_scale.unwrap_or(1.0) / self.qjl_sketch_dim as f32;
            for (row, sign) in encoded.residual_signs.iter().enumerate() {
                let projection =
                    &self.qjl_projections[row * self.head_dim..(row + 1) * self.head_dim];
                for (idx, value) in projection.iter().enumerate() {
                    out[idx] += scale * *sign as f32 * value;
                }
            }
        }
        for &(idx, value) in &encoded.protected_exact {
            out[idx] = value;
        }
        out
    }

    fn score(&self, encoded: &EncodedTurboVector, query: &[f32]) -> f32 {
        let query = fit_dim(query, self.head_dim);
        let mut score = dot_product(&query, &encoded.mse_reconstruction);
        if self.qjl_sketch_dim > 0 && !encoded.residual_signs.is_empty() {
            let scale = self.qjl_scale.unwrap_or(1.0) / self.qjl_sketch_dim as f32;
            for (row, sign) in encoded.residual_signs.iter().enumerate() {
                let projection =
                    &self.qjl_projections[row * self.head_dim..(row + 1) * self.head_dim];
                score += scale * *sign as f32 * dot_product(projection, &query);
            }
        }
        score
    }
}

#[derive(Clone)]
pub struct PaperKeyOnlyCodec {
    artifacts: PaperArtifacts,
    key_vectors: HashMap<KvStoreKey, EncodedTurboVector>,
    value_vectors: HashMap<KvStoreKey, Vec<f32>>,
}

impl PaperKeyOnlyCodec {
    pub fn new(config: &PaperTurboQuantConfig) -> anyhow::Result<Self> {
        Ok(Self {
            artifacts: PaperArtifacts::from_config(config)?,
            key_vectors: HashMap::new(),
            value_vectors: HashMap::new(),
        })
    }
}

impl KvCodec for PaperKeyOnlyCodec {
    fn name(&self) -> &'static str {
        "paper-key-only"
    }

    fn fork_session(&self) -> Box<dyn KvCodec + Send> {
        Box::new(self.clone())
    }

    fn ingest_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let encoded = self.artifacts.encode(data);
        let reconstructed = self.artifacts.reconstruct(&encoded);
        self.key_vectors.insert((layer, head, token), encoded);
        Ok(reconstructed)
    }

    fn ingest_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        self.value_vectors
            .insert((layer, head, token), data.to_vec());
        Ok(data.to_vec())
    }

    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(token_range
            .map(|token| {
                self.key_vectors
                    .get(&(layer, head, token))
                    .map(|encoded| self.artifacts.score(encoded, query))
                    .unwrap_or(0.0)
            })
            .collect())
    }

    fn read_v(&self, layer: u32, head: u32, token_range: Range<u32>) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        for token in token_range {
            if let Some(value) = self.value_vectors.get(&(layer, head, token)) {
                out.extend_from_slice(value);
            }
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub struct PaperFullKvCodec {
    artifacts: PaperArtifacts,
    key_vectors: HashMap<KvStoreKey, EncodedTurboVector>,
    value_vectors: HashMap<KvStoreKey, EncodedTurboVector>,
}

#[derive(Clone)]
pub struct ResearchKvSplitCodec {
    python: TurboQuantCodec,
    value_vectors: HashMap<KvStoreKey, Vec<f32>>,
}

impl ResearchKvSplitCodec {
    pub fn new(resolved: &ResolvedTurboQuantConfig) -> anyhow::Result<Self> {
        let gguf_cfg = resolved.gguf_metadata.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "research-kv-split currently requires GGUF TurboQuant metadata or an explicit runtime bridge"
            )
        })?;

        let rotation_policy = gguf_cfg
            .rotation_policy
            .map(|policy| policy.as_str().to_string());

        Ok(Self {
            python: TurboQuantCodec::new(
                "research-kv-split".to_string(),
                resolved
                    .source_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().to_string())
                    .as_deref(),
                gguf_cfg.num_layers,
                gguf_cfg.num_kv_heads,
                gguf_cfg.head_dim,
                rotation_policy.as_deref(),
                Some(gguf_cfg.rotation_seed),
            )?,
            value_vectors: HashMap::new(),
        })
    }
}

impl KvCodec for ResearchKvSplitCodec {
    fn name(&self) -> &'static str {
        "research-kv-split"
    }

    fn fork_session(&self) -> Box<dyn KvCodec + Send> {
        Box::new(self.clone())
    }

    fn ingest_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let _ = token;
        self.python.compress_k(layer, head, data)
    }

    fn ingest_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        self.value_vectors
            .insert((layer, head, token), data.to_vec());
        Ok(data.to_vec())
    }

    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        self.python
            .score_k(layer, head, query, token_range.start, token_range.end)
    }

    fn read_v(&self, layer: u32, head: u32, token_range: Range<u32>) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        for token in token_range {
            if let Some(value) = self.value_vectors.get(&(layer, head, token)) {
                out.extend_from_slice(value);
            }
        }
        Ok(out)
    }
}

impl PaperFullKvCodec {
    pub fn new(config: &PaperTurboQuantConfig) -> anyhow::Result<Self> {
        Ok(Self {
            artifacts: PaperArtifacts::from_config(config)?,
            key_vectors: HashMap::new(),
            value_vectors: HashMap::new(),
        })
    }
}

impl KvCodec for PaperFullKvCodec {
    fn name(&self) -> &'static str {
        "paper-full-kv"
    }

    fn fork_session(&self) -> Box<dyn KvCodec + Send> {
        Box::new(self.clone())
    }

    fn ingest_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let encoded = self.artifacts.encode(data);
        let reconstructed = self.artifacts.reconstruct(&encoded);
        self.key_vectors.insert((layer, head, token), encoded);
        Ok(reconstructed)
    }

    fn ingest_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        data: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let encoded = self.artifacts.encode(data);
        let reconstructed = self.artifacts.reconstruct(&encoded);
        self.value_vectors.insert((layer, head, token), encoded);
        Ok(reconstructed)
    }

    fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(token_range
            .map(|token| {
                self.key_vectors
                    .get(&(layer, head, token))
                    .map(|encoded| self.artifacts.score(encoded, query))
                    .unwrap_or(0.0)
            })
            .collect())
    }

    fn read_v(&self, layer: u32, head: u32, token_range: Range<u32>) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        for token in token_range {
            if let Some(encoded) = self.value_vectors.get(&(layer, head, token)) {
                out.extend_from_slice(&self.artifacts.reconstruct(encoded));
            }
        }
        Ok(out)
    }
}

fn fit_dim(vector: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    let copy = vector.len().min(dim);
    out[..copy].copy_from_slice(&vector[..copy]);
    out
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn apply_rotation(matrix: &[f32], dim: usize, vector: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    for row in 0..dim {
        let row_values = &matrix[row * dim..(row + 1) * dim];
        out[row] = dot_product(row_values, vector);
    }
    out
}

fn apply_rotation_transpose(matrix: &[f32], dim: usize, vector: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    for row in 0..dim {
        let value = vector[row];
        for col in 0..dim {
            out[col] += matrix[row * dim + col] * value;
        }
    }
    out
}

fn nearest_centroid(value: f32, centroids: &[f32]) -> f32 {
    centroids
        .iter()
        .copied()
        .min_by(|a, b| {
            (value - *a)
                .abs()
                .partial_cmp(&(value - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(value)
}

fn subtract_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn seeded_projection_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut state = if seed == 0 {
        0xA5A5_1F1F_DEAD_BEEF
    } else {
        seed
    };
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        let value = (unit as f32 * 2.0) - 1.0;
        out.push(value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::turboquant_sidecar::{
        PaperTurboQuantConfig, ResidualQjlArtifact, RotationArtifact, ScalarQuantizerArtifact,
        TurboQuantSchemaKind,
    };

    fn paper_config(value_exact: bool) -> PaperTurboQuantConfig {
        PaperTurboQuantConfig {
            schema_kind: TurboQuantSchemaKind::Paper,
            codec: "turboquant-prod".into(),
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            key_bits: 3.5,
            value_bits: Some(3.5),
            mixed_bits: Some(3.5),
            value_exact,
            rotation: RotationArtifact {
                kind: Some("matrix".into()),
                seed: Some(7),
                matrix: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            },
            scalar_quantizer: ScalarQuantizerArtifact {
                centroids: vec![-1.0, -0.25, 0.25, 1.0],
                decision_boundaries: vec![-0.5, 0.0, 0.5],
                protected_centroids: vec![-1.0, 0.0, 1.0],
            },
            residual_qjl: Some(ResidualQjlArtifact {
                sketch_dim: 2,
                scale: 1.0,
                seed: Some(123),
                projections: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }),
            protected_channels: vec![1],
            outlier_channels: vec![1],
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn exact_codec_round_trips_values() {
        let mut codec = ExactKvCodec::default();
        codec.ingest_k(0, 0, 0, &[1.0, 2.0]).unwrap();
        codec.ingest_v(0, 0, 0, &[3.0, 4.0]).unwrap();
        let forked = codec.fork_session();

        assert_eq!(codec.score_k(0, 0, &[1.0, 1.0], 0..1).unwrap(), vec![3.0]);
        assert_eq!(codec.read_v(0, 0, 0..1).unwrap(), vec![3.0, 4.0]);
        assert_eq!(forked.score_k(0, 0, &[1.0, 1.0], 0..1).unwrap(), vec![3.0]);
    }

    #[test]
    fn paper_key_only_keeps_values_exact_and_scores_from_state() {
        let mut codec = PaperKeyOnlyCodec::new(&paper_config(true)).unwrap();
        let reconstructed = codec.ingest_k(0, 0, 0, &[0.8, 0.2, -0.1, 0.4]).unwrap();
        codec.ingest_v(0, 0, 0, &[3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(codec.read_v(0, 0, 0..1).unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(reconstructed[1], 0.2);
        assert_ne!(
            codec.score_k(0, 0, &[1.0, 0.0, 0.0, 0.0], 0..1).unwrap()[0],
            0.0
        );
    }

    #[test]
    fn paper_full_kv_reconstructs_values() {
        let mut codec = PaperFullKvCodec::new(&paper_config(false)).unwrap();
        let reconstructed = codec.ingest_v(0, 0, 0, &[0.6, -0.2, 0.4, 0.1]).unwrap();
        let read_back = codec.read_v(0, 0, 0..1).unwrap();
        assert_eq!(reconstructed.len(), 4);
        assert_eq!(read_back.len(), 4);
        assert!((read_back[1] + 0.2).abs() < 1e-6);
    }
}
