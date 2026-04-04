//! Convert llama-server OpenAI-style `text/event-stream` into KoboldCpp Kai SSE (`event: message`).

use std::convert::Infallible;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

/// Map one OpenAI completion SSE JSON chunk to Kobold `event: message` lines.
fn oai_chunk_to_kai_lines(v: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let finish = v
        .pointer("/choices/0/finish_reason")
        .and_then(|x| {
            if x.is_null() {
                None
            } else {
                x.as_str().map(str::to_owned)
            }
        });

    let mut piece = v
        .pointer("/choices/0/text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if piece.is_empty() {
        piece = v
            .pointer("/choices/0/delta/content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
    }

    if !piece.is_empty() {
        let kai = json!({"token": piece, "finish_reason": Value::Null});
        out.push(format!("event: message\ndata: {}\n\n", kai));
    }

    if let Some(fr) = finish {
        let kai = json!({"token": "", "finish_reason": fr});
        out.push(format!("event: message\ndata: {}\n\n", kai));
    }

    out
}

/// Stream: upstream OpenAI SSE → downstream Kobold Kai SSE bytes.
pub fn openai_sse_to_kai_sse(
    upstream: reqwest::Response,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    async_stream::stream! {
        let mut pending = String::new();
        let mut stream = upstream.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            pending.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = pending.find("\n\n") {
                let block = pending[..pos].to_string();
                pending = pending[pos + 2..].to_string();
                for line in block.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let rest = if let Some(r) = line.strip_prefix("data:") {
                        r.trim()
                    } else {
                        continue;
                    };
                    if rest == "[DONE]" {
                        continue;
                    }
                    let v: Value = match serde_json::from_str(rest) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if v.get("error").is_some() {
                        let kai = json!({"token": "", "finish_reason": "error"});
                        yield Ok(Bytes::from(format!(
                            "event: message\ndata: {}\n\n",
                            kai
                        )));
                        continue;
                    }
                    for l in oai_chunk_to_kai_lines(&v) {
                        yield Ok(Bytes::from(l));
                    }
                }
            }
        }
    }
}
