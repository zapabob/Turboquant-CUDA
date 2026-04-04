use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::cache::kv_codec::{build_kv_codec, KvCodec};
use crate::model::turboquant_sidecar::{ResolvedTurboQuantConfig, TurboQuantMode};

/// FFI bridge that connects Rust KvCodec to C callbacks for llama.cpp integration.
///
/// This struct holds the codec state and provides C-compatible callback functions
/// that can be registered with the hypura_kv_codec_config.
pub struct KvCodecFfi {
    codec: Arc<Mutex<Box<dyn KvCodec + Send>>>,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    compress_values: i32,
}

/// Owned C runtime wrapper for hypura_kv_codec.
///
/// Keeps the Rust callback state alive (`KvCodecFfi`) and exposes safe-ish
/// methods for invoking the C shim entrypoints from Rust inference code.
pub struct KvCodecRuntimeBridge {
    ffi: KvCodecFfi,
    runtime: *mut hypura_sys::hypura_kv_codec_runtime_t,
}

impl KvCodecFfi {
    /// Create a new FFI bridge from a resolved TurboQuant config.
    pub fn new(resolved: &ResolvedTurboQuantConfig) -> anyhow::Result<Self> {
        let codec = build_kv_codec(resolved)?;

        let paper_config = resolved.paper_config();
        let gguf_cfg = resolved.gguf_metadata.as_ref();
        let num_layers = paper_config
            .map(|c| c.num_layers as u32)
            .or_else(|| gguf_cfg.map(|cfg| cfg.num_layers))
            .unwrap_or(0);
        let num_kv_heads = paper_config
            .map(|c| c.num_kv_heads as u32)
            .or_else(|| gguf_cfg.map(|cfg| cfg.num_kv_heads))
            .unwrap_or(0);
        let head_dim = paper_config
            .map(|c| c.head_dim as u32)
            .or_else(|| gguf_cfg.map(|cfg| cfg.head_dim))
            .unwrap_or(0);

        Ok(Self {
            codec: Arc::new(Mutex::new(codec)),
            num_layers,
            num_kv_heads,
            head_dim,
            compress_values: if resolved.mode == TurboQuantMode::PaperFullKv {
                1
            } else {
                0
            },
        })
    }

    /// Create an exact-mode FFI bridge (no compression).
    pub fn exact(num_layers: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        use crate::cache::kv_codec::ExactKvCodec;

        Self {
            codec: Arc::new(Mutex::new(Box::new(ExactKvCodec::default()))),
            num_layers,
            num_kv_heads,
            head_dim,
            compress_values: 0,
        }
    }

    /// Get the codec name.
    pub fn codec_name(&self) -> &'static str {
        self.codec.lock().unwrap().name()
    }

    /// Fork the codec for a new sequence.
    pub fn fork(&self) -> Self {
        let codec = self.codec.lock().unwrap();
        Self {
            codec: Arc::new(Mutex::new(codec.fork_session())),
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            compress_values: self.compress_values,
        }
    }

    /// Get raw pointer for C callbacks.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        Arc::as_ptr(&self.codec) as *mut c_void
    }

    /// Create C callback config.
    /// SAFETY: The returned config contains function pointers that are only valid
    /// while this KvCodecFfi instance (and its contained Arc) are alive.
    pub unsafe fn to_c_config(&self) -> hypura_sys::hypura_kv_codec_config {
        hypura_sys::hypura_kv_codec_config {
            compress_k: Some(compress_k_callback),
            compress_v: Some(compress_v_callback),
            score_k: Some(score_k_callback),
            read_v: Some(read_v_callback),
            rust_ctx: Arc::as_ptr(&self.codec) as *mut c_void,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            compress_keys: 1,
            compress_values: self.compress_values,
            use_exact_score: 0,
        }
    }
}

impl KvCodecRuntimeBridge {
    pub fn new(resolved: &ResolvedTurboQuantConfig) -> anyhow::Result<Self> {
        let ffi = KvCodecFfi::new(resolved)?;
        let cfg = unsafe { ffi.to_c_config() };
        let runtime = unsafe { hypura_sys::hypura_kv_codec_runtime_create(&cfg) };
        anyhow::ensure!(!runtime.is_null(), "failed to create hypura_kv_codec runtime");
        Ok(Self { ffi, runtime })
    }

    pub fn mode_name(&self) -> &'static str {
        self.ffi.codec_name()
    }

    pub fn compress_k(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        vector: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let mut out = vec![0.0f32; vector.len()];
        let rc = unsafe {
            hypura_sys::hypura_kv_codec_compress_k_vec(
                self.runtime,
                layer,
                head,
                token,
                vector.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        anyhow::ensure!(rc >= 0, "hypura_kv_codec_compress_k_vec failed");
        Ok(out)
    }

    pub fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_range: Range<u32>,
    ) -> anyhow::Result<Vec<f32>> {
        let n_tokens = token_range.end.saturating_sub(token_range.start) as usize;
        let mut scores = vec![0.0f32; n_tokens];
        let rc = unsafe {
            hypura_sys::hypura_kv_codec_score_k_vec(
                self.runtime,
                layer,
                head,
                query.as_ptr(),
                token_range.start,
                token_range.end,
                scores.as_mut_ptr(),
            )
        };
        anyhow::ensure!(rc >= 0, "hypura_kv_codec_score_k_vec failed");
        Ok(scores)
    }

    pub fn compress_v(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
        vector: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let mut out = vec![0.0f32; vector.len()];
        let rc = unsafe {
            hypura_sys::hypura_kv_codec_compress_v_vec(
                self.runtime,
                layer,
                head,
                token,
                vector.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        anyhow::ensure!(rc >= 0, "hypura_kv_codec_compress_v_vec failed");
        Ok(out)
    }

    pub fn read_v(
        &self,
        layer: u32,
        head: u32,
        token_range: Range<u32>,
        head_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let n_tokens = token_range.end.saturating_sub(token_range.start) as usize;
        let mut v_buffer = vec![0.0f32; n_tokens.saturating_mul(head_dim)];
        let rc = unsafe {
            hypura_sys::hypura_kv_codec_read_v_vec(
                self.runtime,
                layer,
                head,
                token_range.start,
                token_range.end,
                v_buffer.as_mut_ptr(),
            )
        };
        anyhow::ensure!(rc >= 0, "hypura_kv_codec_read_v_vec failed");
        Ok(v_buffer)
    }
}

impl Drop for KvCodecRuntimeBridge {
    fn drop(&mut self) {
        if !self.runtime.is_null() {
            unsafe { hypura_sys::hypura_kv_codec_runtime_free(self.runtime) };
            self.runtime = std::ptr::null_mut();
        }
    }
}

// ── C Callback Functions ───────────────────────────────────────────────────

/// C callback for K compression.
/// SAFETY: rust_ctx must be a valid Arc<Mutex<Box<dyn KvCodec>>> pointer.
unsafe extern "C" fn compress_k_callback(
    rust_ctx: *mut c_void,
    layer: u32,
    head: u32,
    token: u32,
    k_data: *const f32,
    head_dim: u32,
    output: *mut f32,
) -> i32 {
    if rust_ctx.is_null() || k_data.is_null() || output.is_null() {
        return -1;
    }

    let codec_ptr = rust_ctx as *const Mutex<Box<dyn KvCodec + Send>>;
    let codec = match codec_ptr.as_ref() {
        Some(c) => c,
        None => return -1,
    };

    let data = std::slice::from_raw_parts(k_data, head_dim as usize);

    let result = codec.lock().unwrap().ingest_k(layer, head, token, data);

    match result {
        Ok(reconstructed) => {
            let out_slice = std::slice::from_raw_parts_mut(output, head_dim as usize);
            out_slice.copy_from_slice(&reconstructed);
            reconstructed.len() as i32
        }
        Err(_) => -1,
    }
}

/// C callback for V compression.
unsafe extern "C" fn compress_v_callback(
    rust_ctx: *mut c_void,
    layer: u32,
    head: u32,
    token: u32,
    v_data: *const f32,
    head_dim: u32,
    output: *mut f32,
) -> i32 {
    if rust_ctx.is_null() || v_data.is_null() || output.is_null() {
        return -1;
    }

    let codec_ptr = rust_ctx as *const Mutex<Box<dyn KvCodec + Send>>;
    let codec = match codec_ptr.as_ref() {
        Some(c) => c,
        None => return -1,
    };

    let data = std::slice::from_raw_parts(v_data, head_dim as usize);

    let result = codec.lock().unwrap().ingest_v(layer, head, token, data);

    match result {
        Ok(reconstructed) => {
            let out_slice = std::slice::from_raw_parts_mut(output, head_dim as usize);
            out_slice.copy_from_slice(&reconstructed);
            reconstructed.len() as i32
        }
        Err(_) => -1,
    }
}

/// C callback for K scoring.
unsafe extern "C" fn score_k_callback(
    rust_ctx: *mut c_void,
    layer: u32,
    head: u32,
    query: *const f32,
    head_dim: u32,
    token_start: u32,
    token_end: u32,
    scores: *mut f32,
) -> i32 {
    if rust_ctx.is_null() || query.is_null() || scores.is_null() {
        return -1;
    }

    let codec_ptr = rust_ctx as *const Mutex<Box<dyn KvCodec + Send>>;
    let codec = match codec_ptr.as_ref() {
        Some(c) => c,
        None => return -1,
    };

    let query_slice = std::slice::from_raw_parts(query, head_dim as usize);
    let token_range: Range<u32> = token_start..token_end;
    let n_tokens = (token_end - token_start) as usize;

    let result = codec
        .lock()
        .unwrap()
        .score_k(layer, head, query_slice, token_range);

    match result {
        Ok(scored) => {
            let out_slice = std::slice::from_raw_parts_mut(scores, n_tokens);
            out_slice.copy_from_slice(&scored);
            0
        }
        Err(_) => -1,
    }
}

/// C callback for V reading.
unsafe extern "C" fn read_v_callback(
    rust_ctx: *mut c_void,
    layer: u32,
    head: u32,
    head_dim: u32,
    token_start: u32,
    token_end: u32,
    v_buffer: *mut f32,
) -> i32 {
    if rust_ctx.is_null() || v_buffer.is_null() {
        return -1;
    }

    let codec_ptr = rust_ctx as *const Mutex<Box<dyn KvCodec + Send>>;
    let codec = match codec_ptr.as_ref() {
        Some(c) => c,
        None => return -1,
    };

    let token_range: Range<u32> = token_start..token_end;
    let n_tokens = (token_end - token_start) as usize;
    let total_elements = n_tokens * head_dim as usize;

    let result = codec.lock().unwrap().read_v(layer, head, token_range);

    match result {
        Ok(values) => {
            let out_slice = std::slice::from_raw_parts_mut(v_buffer, total_elements);
            out_slice.copy_from_slice(&values[..total_elements.min(values.len())]);
            0
        }
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::turboquant_sidecar::{
        PaperTurboQuantConfig, ResolvedTurboQuantConfig, RotationArtifact, ScalarQuantizerArtifact,
        TurboQuantMode, TurboQuantSchemaKind,
    };

    fn test_config() -> ResolvedTurboQuantConfig {
        use crate::model::turboquant_sidecar::TurboQuantSidecarConfig;

        let paper = PaperTurboQuantConfig {
            schema_kind: TurboQuantSchemaKind::Paper,
            codec: "turboquant-prod".into(),
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            key_bits: 3.5,
            value_bits: None,
            mixed_bits: None,
            value_exact: true,
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
                protected_centroids: vec![],
            },
            residual_qjl: None,
            protected_channels: vec![],
            outlier_channels: vec![],
            extra: serde_json::Map::new(),
        };

        ResolvedTurboQuantConfig {
            mode: TurboQuantMode::PaperKeyOnly,
            schema_kind: Some(TurboQuantSchemaKind::Paper),
            source_path: None,
            config: Some(TurboQuantSidecarConfig::Paper(paper)),
        }
    }

    #[test]
    fn ffi_bridge_creation() {
        let config = test_config();
        let ffi = KvCodecFfi::new(&config).unwrap();
        assert_eq!(ffi.codec_name(), "paper-key-only");
        assert_eq!(ffi.num_layers, 2);
        assert_eq!(ffi.num_kv_heads, 2);
        assert_eq!(ffi.head_dim, 4);
    }

    #[test]
    fn ffi_exact_mode() {
        let ffi = KvCodecFfi::exact(4, 8, 64);
        assert_eq!(ffi.codec_name(), "exact");
    }
}
