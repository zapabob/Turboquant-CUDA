//! KoboldCpp-compatible HTTP surface; forwards LLM/token work to llama.cpp `llama-server`.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Json;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::sync::watch;
use tower_http::cors::{Any, CorsLayer};

use crate::kobold::{
    kobold_to_openai_completions, kobold_to_openai_completions_stream, openai_completion_to_kobold,
    KoboldGenerateRequest,
};
use crate::stream::openai_sse_to_kai_sse;
use axum::extract::State;
use futures_util::StreamExt;

/// Static metadata advertised to Kobold clients (SillyTavern, etc.).
#[derive(Clone)]
pub struct ProxyMeta {
    pub advertised_model: String,
    pub max_length: u32,
    pub max_context: u32,
}

#[derive(Clone)]
pub struct ProxyState {
    pub client: reqwest::Client,
    pub backend_base: String,
    pub meta: ProxyMeta,
}

fn reqwest_to_axum(res: reqwest::Response) -> Response {
    let status = res.status();
    let mut headers = HeaderMap::new();
    for (k, v) in res.headers() {
        if k == "transfer-encoding" || k == "connection" {
            continue;
        }
        if let Ok(name) = k.as_str().parse::<HeaderName>() {
            headers.insert(name, v.clone());
        }
    }
    let stream = res
        .bytes_stream()
        .map(|r| r.map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "upstream stream")));
    let body = Body::from_stream(stream);
    let mut res = Response::new(body);
    *res.status_mut() = status;
    *res.headers_mut() = headers;
    res
}

async fn forward_to_llama(State(state): State<Arc<ProxyState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match body.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => return (StatusCode::BAD_REQUEST, "body").into_response(),
    };
    let pq = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", state.backend_base, pq);
    let mut rb = state.client.request(parts.method.clone(), &url);
    if let Some(ct) = parts.headers.get("content-type") {
        rb = rb.header("content-type", ct);
    }
    if let Some(ac) = parts.headers.get("accept") {
        rb = rb.header("accept", ac);
    }
    if let Some(auth) = parts.headers.get("authorization") {
        rb = rb.header("authorization", auth);
    }
    let res = if bytes.is_empty() {
        rb.send().await
    } else {
        rb.body(bytes).send().await
    };
    match res {
        Ok(r) => reqwest_to_axum(r),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

fn extra_capabilities() -> Value {
    json!({
        "result": "KoboldCpp",
        "version": "kobold_gguf_gui-0.2",
        "protected": false,
        "llm": true,
        "txt2img": false,
        "vision": false,
        "audio": false,
        "transcribe": false,
        "multiplayer": false,
        "websearch": false,
        "tts": false,
        "embeddings": true,
        "music": false,
        "savedata": false,
        "admin": 0,
        "router": false,
        "guidance": false,
        "jinja": false,
        "mcp": false
    })
}

async fn kobold_generate(
    State(state): State<Arc<ProxyState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> impl IntoResponse {
    let url = format!("{}/v1/completions", state.backend_base);
    let body = kobold_to_openai_completions(&req);
    let res = match state.client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": {"msg": format!("backend request failed: {e}"), "type": "service_unavailable"}})),
            )
                .into_response();
        }
    };
    if !res.status().is_success() {
        let status = res.status();
        let t = res.text().await.unwrap_or_default();
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": {"msg": format!("llama-server {status}: {t}"), "type": "bad_input"}})),
        )
            .into_response();
    }
    let v: Value = match res.json().await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": {"msg": format!("bad json from backend: {e}"), "type": "bad_input"}})),
            )
                .into_response();
        }
    };
    match openai_completion_to_kobold(&v) {
        Ok(k) => (StatusCode::OK, Json(k)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": {"msg": e.to_string(), "type": "server_error"}})),
        )
            .into_response(),
    }
}

async fn kobold_generate_stream(
    State(state): State<Arc<ProxyState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> impl IntoResponse {
    let url = format!("{}/v1/completions", state.backend_base);
    let body = kobold_to_openai_completions_stream(&req);
    let res = match state
        .client
        .post(&url)
        .header("accept", "text/event-stream")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": {"msg": format!("backend stream failed: {e}"), "type": "service_unavailable"}})),
            )
                .into_response();
        }
    };
    if !res.status().is_success() {
        let status = res.status();
        let t = res.text().await.unwrap_or_default();
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": {"msg": format!("llama-server {status}: {t}"), "type": "bad_input"}})),
        )
            .into_response();
    }
    let stream = openai_sse_to_kai_sse(res);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header("X-Accel-Buffering", "no")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

async fn tokencount(State(state): State<Arc<ProxyState>>, body: Bytes) -> impl IntoResponse {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"value": -1}))).into_response(),
    };
    let prompt = v
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    let url = format!("{}/tokenize", state.backend_base);
    let payload = json!({ "content": prompt });
    let res = match state.client.post(&url).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"value": -1, "error": e.to_string()})),
            )
                .into_response();
        }
    };
    let out: Value = match res.json().await {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_GATEWAY, Json(json!({"value": -1}))).into_response(),
    };
    let ids = out.get("tokens").cloned().unwrap_or(json!([]));
    let n = ids.as_array().map(|a| a.len()).unwrap_or(0);
    (StatusCode::OK, Json(json!({"value": n, "ids": ids}))).into_response()
}

async fn detokenize(State(state): State<Arc<ProxyState>>, body: Bytes) -> impl IntoResponse {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"result": "", "success": false})),
            )
                .into_response();
        }
    };
    let tokens = v.get("ids").cloned().unwrap_or_else(|| json!([]));
    let url = format!("{}/detokenize", state.backend_base);
    let payload = json!({ "tokens": tokens });
    let res = match state.client.post(&url).json(&payload).send().await {
        Ok(r) => r,
        Err(_) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"result": "", "success": false})),
            )
                .into_response();
        }
    };
    let out: Value = match res.json().await {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"result": "", "success": false})),
            )
                .into_response();
        }
    };
    let text = out
        .get("content")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    (
        StatusCode::OK,
        Json(json!({"result": text, "success": true})),
    )
        .into_response()
}

fn service_unavailable(msg: &'static str) -> impl IntoResponse {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"detail": {"msg": msg, "type": "service_unavailable"}})),
    )
}

async fn get_api_model(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({"result": state.meta.advertised_model}))
}

async fn get_max_length(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({"value": state.meta.max_length}))
}

async fn get_max_context(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({"value": state.meta.max_context}))
}

async fn get_true_max_context(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({"value": state.meta.max_context}))
}

async fn get_extra_version() -> Json<Value> {
    Json(extra_capabilities())
}

async fn proxy_extra_embeddings(State(state): State<Arc<ProxyState>>, body: Bytes) -> Response {
    let url = format!("{}/v1/embeddings", state.backend_base);
    let res = state
        .client
        .post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await;
    match res {
        Ok(r) => reqwest_to_axum(r),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

async fn wait_shutdown(mut shutdown: watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

pub async fn run_proxy_server(
    bind: SocketAddr,
    backend_port: u16,
    shutdown: watch::Receiver<bool>,
    meta: ProxyMeta,
) -> anyhow::Result<()> {
    let backend_base = format!("http://127.0.0.1:{backend_port}");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("reqwest client")?;

    let state = Arc::new(ProxyState {
        client,
        backend_base,
        meta,
    });

    let app = axum::Router::new()
        .route("/api/v1/generate", post(kobold_generate))
        .route("/api/latest/generate", post(kobold_generate))
        .route("/api/extra/generate/stream", post(kobold_generate_stream))
        .route("/api/extra/tokencount", post(tokencount))
        .route("/api/extra/tokenize", post(tokencount))
        .route("/api/extra/detokenize", post(detokenize))
        .route(
            "/api/extra/json_to_grammar",
            post(|Json(_body): Json<Value>| async move {
                Json(json!({"result": "", "success": false, "note": "grammar conversion not implemented in proxy"}))
            }),
        )
        .route(
            "/api/extra/abort",
            post(|| async {
                Json(json!({"success": "false", "done": "true"}))
            }),
        )
        .route(
            "/api/extra/generate/check",
            post(|| async {
                Json(json!({"results": [{"text": ""}]}))
            }),
        )
        .route(
            "/api/extra/last_logprobs",
            post(|| async { Json(json!({"logprobs": null})) }),
        )
        .route(
            "/api/extra/multiplayer/status",
            post(|| async {
                Json(json!({"error": "Multiplayer not enabled!"}))
            }),
        )
        .route(
            "/api/extra/data/list",
            post(|| async { Json(json!([])) }),
        )
        .route(
            "/api/extra/data/load",
            post(|| async {
                Json(json!({"success": false, "data": null}))
            }),
        )
        .route(
            "/api/extra/data/save",
            post(|| async {
                Json(json!({"success": false, "error": "SaveDataFile not enabled!"}))
            }),
        )
        .route(
            "/api/extra/multiplayer/getstory",
            post(|| async {
                ([(header::CONTENT_TYPE, "text/plain")], Bytes::new())
            }),
        )
        .route(
            "/api/extra/multiplayer/setstory",
            post(|| async {
                Json(json!({"success": false, "error": "Multiplayer not enabled!"}))
            }),
        )
        .route(
            "/api/extra/websearch",
            post(|| async { Json(json!([])) }),
        )
        .route(
            "/api/extra/websearch/{*rest}",
            post(|| async { Json(json!([])) }),
        )
        .route(
            "/api/admin/reload_config",
            post(|| async {
                (StatusCode::FORBIDDEN, Json(json!({"success": false})))
            }),
        )
        .route(
            "/api/admin/list_options",
            get(|| async { Json(json!([])) }),
        )
        .route(
            "/api/admin/check_state",
            post(|| async {
                Json(json!({"success": false, "old_states": [], "new_state_size": 0, "new_tokens": 0}))
            }),
        )
        .route(
            "/api/admin/load_state",
            post(|| async {
                Json(json!({"success": false, "new_state_size": 0, "new_tokens": 0}))
            }),
        )
        .route(
            "/api/admin/clear_state",
            post(|| async { Json(json!({"success": false})) }),
        )
        .route(
            "/api/extra/shutdown",
            post(|| async { Json(json!({"success": false})) }),
        )
        .route(
            "/api/extra/transcribe",
            post(|| async { service_unavailable("STT not available in llama-only proxy") }),
        )
        .route(
            "/api/extra/tts",
            post(|| async { service_unavailable("TTS not available in llama-only proxy") }),
        )
        .route(
            "/api/extra/music/prepare",
            post(|| async { service_unavailable("Music not available") }),
        )
        .route(
            "/api/extra/music/generate",
            post(|| async { service_unavailable("Music not available") }),
        )
        .route("/api/extra/embeddings", post(proxy_extra_embeddings))
        .route(
            "/api/extra/perf",
            get(|| async {
                Json(json!({
                    "last_process": 0,
                    "last_eval": 0,
                    "last_token_count": 0,
                    "last_input_count": 0,
                    "total_gens": 0,
                    "queue": 0,
                    "idle": 1,
                    "uptime": 0,
                    "idletime": 0,
                    "quiet": false
                }))
            }),
        )
        .route("/api/extra/preloadstory", post(|| async { Json(json!({})) }))
        .route("/api/extra/version", get(get_extra_version))
        .route("/api/extra/true_max_context_length", get(get_true_max_context))
        .route("/api/v1/model", get(get_api_model))
        .route("/api/latest/model", get(get_api_model))
        .route("/api/v1/config/max_length", get(get_max_length))
        .route("/api/latest/config/max_length", get(get_max_length))
        .route("/api/v1/config/max_context_length", get(get_max_context))
        .route("/api/latest/config/max_context_length", get(get_max_context))
        .route(
            "/api/v1/config/soft_prompt",
            get(|| async { Json(json!({"value": ""})) }),
        )
        .route(
            "/api/latest/config/soft_prompt",
            get(|| async { Json(json!({"value": ""})) }),
        )
        .route(
            "/api/v1/config/soft_prompts_list",
            get(|| async { Json(json!({"values": []})) }),
        )
        .route(
            "/api/latest/config/soft_prompts_list",
            get(|| async { Json(json!({"values": []})) }),
        )
        .route(
            "/api/v1/info/version",
            get(|| async { Json(json!({"result": "1.87.1"})) }),
        )
        .route(
            "/api/latest/info/version",
            get(|| async { Json(json!({"result": "1.87.1"})) }),
        )
        .route(
            "/api/show",
            get(|| async {
                Json(json!({
                    "parameters": "temperature 0.7",
                    "license": "kobold_gguf_gui proxy (llama.cpp)",
                    "modelfile": "kobold_gguf_gui",
                    "capabilities": ["completion"],
                    "modified_at": "2026-04-04T00:00:00.0000000+00:00",
                    "details": {},
                    "model_info": {}
                }))
            }),
        )
        .route(
            "/api/tags",
            get(|| async { Json(json!({"models": []})) }),
        )
        .route(
            "/api/ps",
            get(|| async { Json(json!({"models": []})) }),
        )
        .route(
            "/api/version",
            get(|| async { Json(json!({"version": "0.0.0"})) }),
        )
        .route("/health", get(|| async { "ok" }))
        .route("/v1/{*rest}", any(|s: State<Arc<ProxyState>>, r: Request<Body>| async move {
            forward_to_llama(s, r).await
        }))
        .route("/tokenize", post(|s: State<Arc<ProxyState>>, r: Request<Body>| async move {
            forward_to_llama(s, r).await
        }))
        .route("/detokenize", post(|s: State<Arc<ProxyState>>, r: Request<Body>| async move {
            forward_to_llama(s, r).await
        }))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(wait_shutdown(shutdown))
        .await?;
    Ok(())
}
