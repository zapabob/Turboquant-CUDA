//! KoboldCpp `/api/v1/generate` request/response shapes and mapping to llama.cpp `/v1/completions`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Subset of fields accepted by KoboldCpp-style clients (SillyTavern, etc.).
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct KoboldGenerateRequest {
    pub prompt: String,
    pub max_length: u32,
    pub max_context_length: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub top_a: Option<f32>,
    pub tfs: Option<f32>,
    pub typical: Option<f32>,
    pub rep_pen: Option<f32>,
    pub rep_pen_range: Option<u32>,
    pub rep_pen_slope: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub trim_stop: Option<bool>,
    pub stop_sequence: Option<Vec<String>>,
    pub banned_tokens: Option<Vec<i64>>,
    pub bias_tokens: Option<Vec<Value>>,
    pub quiet: Option<bool>,
}

impl Default for KoboldGenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_length: 128,
            max_context_length: None,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(0),
            top_a: None,
            tfs: None,
            typical: None,
            rep_pen: Some(1.0),
            rep_pen_range: None,
            rep_pen_slope: None,
            presence_penalty: None,
            frequency_penalty: None,
            trim_stop: None,
            stop_sequence: None,
            banned_tokens: None,
            bias_tokens: None,
            quiet: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct KoboldGenerateResponse {
    pub results: Vec<KoboldResult>,
}

#[derive(Debug, Serialize)]
pub struct KoboldResult {
    pub text: String,
}

/// Map Kobold request to OpenAI-compatible `/v1/completions` body for llama.cpp server.
pub fn kobold_to_openai_completions(req: &KoboldGenerateRequest) -> Value {
    let mut body = json!({
        "prompt": req.prompt,
        "max_tokens": req.max_length,
        "temperature": req.temperature.unwrap_or(0.7),
        "top_p": req.top_p.unwrap_or(0.9),
    });

    let top_k = req.top_k.unwrap_or(0);
    if top_k > 0 {
        body["top_k"] = json!(top_k);
    }

    if let Some(rp) = req.rep_pen {
        body["repeat_penalty"] = json!(rp);
    }
    if let Some(stops) = &req.stop_sequence {
        if !stops.is_empty() {
            body["stop"] = json!(stops);
        }
    }
    if let Some(pp) = req.presence_penalty {
        body["presence_penalty"] = json!(pp);
    }
    if let Some(fp) = req.frequency_penalty {
        body["frequency_penalty"] = json!(fp);
    }

    body
}

/// Same as [`kobold_to_openai_completions`] with `"stream": true` for SSE upstream.
pub fn kobold_to_openai_completions_stream(req: &KoboldGenerateRequest) -> Value {
    let mut v = kobold_to_openai_completions(req);
    v["stream"] = json!(true);
    v
}

/// Parse OpenAI-style completion response from llama-server.
pub fn openai_completion_to_kobold(v: &Value) -> anyhow::Result<KoboldGenerateResponse> {
    let text = v
        .pointer("/choices/0/text")
        .and_then(|x| x.as_str())
        .map(str::to_owned)
        .or_else(|| {
            v.pointer("/choices/0/message/content")
                .and_then(|x| x.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_default();

    Ok(KoboldGenerateResponse {
        results: vec![KoboldResult { text }],
    })
}
