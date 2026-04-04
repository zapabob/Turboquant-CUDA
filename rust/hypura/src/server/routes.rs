use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{fs, path::PathBuf};
use std::collections::{HashMap, VecDeque};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum::response::Html;
use tokio::sync::{mpsc, oneshot};

use crate::compute::inference::{GenerateFromLoadedParams, GenerationResult, LoadedModel};
use crate::model::turboquant_sidecar::{ResolvedTurboQuantConfig, TurboQuantMode};
use crate::server::chat::format_chat_prompt;
use crate::server::ollama_types::*;
use crate::server::streaming;
use crate::telemetry::metrics::TelemetryEmitter;

pub struct AppState {
    pub loaded_model: Arc<std::sync::Mutex<LoadedModel>>,
    pub model_name: Arc<Mutex<String>>,
    pub model_path: Arc<Mutex<PathBuf>>,
    pub gguf_info: Arc<Mutex<GgufInfo>>,
    pub model_dir: PathBuf,
    pub default_context: u32,
    pub load_duration_ns: u64,
    pub telemetry: Arc<TelemetryEmitter>,
    pub turboquant: ResolvedTurboQuantConfig,
    pub active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub generation_in_progress: Arc<AtomicBool>,
    pub gui_presets: Arc<Mutex<HashMap<String, GuiPresetItem>>>,
    pub gui_history: Arc<Mutex<VecDeque<GuiHistoryItem>>>,
    pub gui_events: Arc<Mutex<VecDeque<GuiEventItem>>>,
    pub ui_theme: Arc<Mutex<String>>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health_handler))
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/kobold-lite", get(kobold_lite_gui_handler))
        .route("/api/extra/models", get(models_handler))
        .route("/api/extra/model/switch", post(model_switch_handler))
        .route("/api/extra/presets/list", get(gui_presets_list_handler))
        .route("/api/extra/presets/save", post(gui_presets_save_handler))
        .route("/api/extra/presets/delete", post(gui_presets_delete_handler))
        .route("/api/extra/history", get(gui_history_handler))
        .route("/api/extra/events", get(gui_events_handler))
        .route("/api/extra/ui-theme", get(ui_theme_get_handler))
        .route("/api/extra/ui-theme", post(ui_theme_set_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/v1/model", get(kobold_model_handler))
        .route("/api/v1/generate", post(kobold_generate_handler))
        .route("/api/extra/generate/stream", post(kobold_generate_stream_handler))
        .route("/api/extra/generate/check", get(kobold_generate_check_handler))
        .route("/api/extra/abort", post(kobold_abort_handler))
        .route(
            "/api/extra/true_max_context_length",
            get(kobold_true_max_context_length_handler),
        )
        .with_state(state)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn version_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")}))
}

async fn kobold_lite_gui_handler() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hypura Kobold GUI</title>
  <style>
    :root {
      --bg: #f1f2f5;
      --panel: #ffffff;
      --panel-soft: #fafafa;
      --text: #1c1f23;
      --muted: #5f6670;
      --border: #d4d9e1;
      --accent: #3254ff;
      --ok: #1b8f3b;
      --err: #b3261e;
      --kbd: #eceff7;
    }
    body[data-theme="dark"] {
      --bg: #101319;
      --panel: #181d26;
      --panel-soft: #202736;
      --text: #e8edf6;
      --muted: #a9b4c6;
      --border: #2d3648;
      --accent: #7ea2ff;
      --ok: #61d889;
      --err: #ff8f87;
      --kbd: #252d3d;
    }
    body[data-theme="classic"] {
      --bg: #efe7d2;
      --panel: #f6f0e1;
      --panel-soft: #ece2c8;
      --text: #3f2f1f;
      --muted: #685741;
      --border: #b7a37f;
      --accent: #7e4f2b;
      --ok: #3d7a42;
      --err: #9d3a2d;
      --kbd: #e6d7b5;
    }
    body {
      font-family: "Segoe UI", "Noto Sans JP", sans-serif;
      max-width: 1380px;
      margin: 1rem auto;
      padding: 0 1rem;
      color: var(--text);
      background: var(--bg);
    }
    textarea, input, select {
      width: 100%;
      box-sizing: border-box;
      margin-top: .3rem;
      margin-bottom: .7rem;
      background: var(--panel-soft);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: .45rem .55rem;
    }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap: .7rem; }
    .toolbar { display:flex; gap:.6rem; margin:.8rem 0; flex-wrap: wrap; }
    .panel { border:1px solid var(--border); border-radius:8px; padding:.8rem; margin:.8rem 0; background:var(--panel); }
    .layout { display:grid; grid-template-columns: 1.1fr 0.9fr; gap: .9rem; }
    .metrics-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.5rem; }
    .metric { border:1px solid var(--border); border-radius:8px; padding:.6rem; background:var(--panel-soft); }
    .metric .k { font-size:.78rem; color:var(--muted); }
    .metric .v { font-size:1.05rem; font-weight:600; }
    .status { border-left:4px solid var(--muted); padding:.5rem .7rem; background:var(--panel-soft); margin:.6rem 0; }
    .status.ok { border-left-color:var(--ok); }
    .status.err { border-left-color:var(--err); }
    button {
      padding:.55rem .95rem;
      cursor:pointer;
      border-radius:8px;
      border:1px solid var(--border);
      background: var(--panel-soft);
      color: var(--text);
    }
    button:hover { border-color: var(--accent); }
    button[disabled] { opacity:.55; cursor:not-allowed; }
    pre { white-space: pre-wrap; border:1px solid var(--border); padding:1rem; border-radius:6px; min-height:220px; background:var(--panel); }
  </style>
</head>
<body data-theme="classic">
  <h2>Hypura Kobold GUI (Parity++)</h2>
  <div class="panel">
    <div class="toolbar">
      <label style="min-width:220px;">Theme
        <select id="themeSelect" onchange="changeThemeFromSelect()">
          <option value="light">light</option>
          <option value="dark">dark</option>
          <option value="classic">classic</option>
        </select>
      </label>
      <span id="themeStatus">Theme source: initializing</span>
    </div>
  </div>
  <div id="connStatus" class="status">Connection: checking...</div>
  <div class="panel">
    <strong>GGUF Model Switcher</strong>
    <div class="toolbar">
      <select id="modelList"></select>
      <button onclick="refreshModelList()">Refresh Models</button>
      <button onclick="switchModel()">Switch Model</button>
    </div>
    <div id="modelStatus">Model: -</div>
  </div>
  <div class="toolbar">
    <button onclick="applyStage('short')">Stage: Short (4096/64)</button>
    <button onclick="applyStage('medium')">Stage: Medium (4096/256)</button>
    <button onclick="applyStage('long')">Stage: Long (8192/512)</button>
    <button onclick="checkConnection()">Recheck Connection</button>
  </div>
  <div class="panel">
    <strong>Runtime Metrics</strong>
    <div class="metrics-grid">
      <div class="metric"><div class="k">tok/s</div><div class="v" id="mTok">-</div></div>
      <div class="metric"><div class="k">prompt ms</div><div class="v" id="mPrompt">-</div></div>
      <div class="metric"><div class="k">eval count</div><div class="v" id="mEval">-</div></div>
      <div class="metric"><div class="k">total ms</div><div class="v" id="mTotal">-</div></div>
    </div>
    <div class="toolbar">
      <span id="busyState">State: idle</span>
      <span id="lastSuccess">Last success: -</span>
      <span id="tokenProgress">Progress: 0 token / 0 ms</span>
    </div>
  </div>
  <div class="layout">
  <div>
  <div class="panel">
    <label>Preset Name</label>
    <input id="presetName" type="text" placeholder="my-preset">
    <div class="toolbar">
      <button onclick="savePreset()">Preset Save</button>
      <button onclick="loadPreset()">Preset Load</button>
      <button onclick="deletePreset()">Preset Delete</button>
      <button onclick="diffPreset()">Preset Diff</button>
      <button onclick="exportPresetCli()">Export CLI</button>
      <button onclick="importPresetCli()">Import CLI</button>
      <select id="presetList"></select>
    </div>
    <label>CLI Bridge (paste / generated)</label>
    <textarea id="cliBridge" rows="2" placeholder="hypura serve --model ./model.gguf --temperature 0.8 ..."></textarea>
  </div>
  <div class="panel">
    <label>Prompt</label>
    <textarea id="prompt" rows="7"></textarea>
    <div class="row">
      <label>max_length<input id="maxlen" type="number" value="256"></label>
      <label>temperature<input id="temp" type="number" step="0.01" value="0.8"></label>
      <label>top_k<input id="topk" type="number" value="40"></label>
      <label>top_p<input id="topp" type="number" step="0.01" value="0.9"></label>
    </div>
    <div class="row">
      <label>min_p<input id="minp" type="number" step="0.01" value="0.05"></label>
      <label>rep_pen<input id="repen" type="number" step="0.01" value="1.0"></label>
      <label>rep_pen_range<input id="repenRange" type="number" value="64"></label>
      <label>seed<input id="seed" type="number" value="42"></label>
    </div>
    <div class="row">
      <label>top_a<input id="topa" type="number" step="0.01" value="0"></label>
      <label>tfs<input id="tfs" type="number" step="0.01" value="1"></label>
      <label>typical<input id="typical" type="number" step="0.01" value="1"></label>
      <label>sampler_order<input id="samplerOrder" type="text" value="6,0,1,3,4,2,5"></label>
    </div>
    <div class="row">
      <label>mirostat<input id="mirostat" type="number" value="0"></label>
      <label>mirostat_tau<input id="mirostatTau" type="number" step="0.01" value="5"></label>
      <label>mirostat_eta<input id="mirostatEta" type="number" step="0.01" value="0.1"></label>
      <label>dynatemp_range<input id="dynatempRange" type="number" step="0.01" value="0"></label>
    </div>
    <div class="row">
      <label>dynatemp_exponent<input id="dynatempExponent" type="number" step="0.01" value="1"></label>
      <label>smoothing_factor<input id="smoothingFactor" type="number" step="0.01" value="0"></label>
      <label>presence_penalty<input id="presencePenalty" type="number" step="0.01" value="0"></label>
      <label>frequency_penalty<input id="frequencyPenalty" type="number" step="0.01" value="0"></label>
    </div>
    <div class="row">
      <label>tq_so8_off<select id="tqSo8Off"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_so8_learned<select id="tqSo8Learned"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_triality_off<select id="tqTrialityOff"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_triality_mix<input id="tqTrialityMix" type="number" step="0.01" placeholder="(default)"></label>
    </div>
    <div class="row">
      <label>tq_rotation_seed<input id="tqRotationSeed" type="number" placeholder="(default)"></label>
      <label>tq_artifact<input id="tqArtifact" type="text" placeholder="C:\path\to\artifact.json"></label>
      <label></label>
      <label></label>
    </div>
    <label>stop_sequence (1行1個)</label>
    <textarea id="stop" rows="3"></textarea>
  </div>
  <div class="toolbar">
    <button onclick="generateOnce()">Generate</button>
    <button onclick="generateStream()">Generate Stream</button>
    <button onclick="retryLast()">Retry Last</button>
    <button onclick="abortGen()">Abort</button>
    <button onclick="checkStatus()">Check</button>
  </div>
  <h3>Output</h3>
  <pre id="out"></pre>
  </div>
  <div>
  <div class="panel">
  <strong>UI Mode</strong>
  <div class="toolbar">
    <button onclick="switchMode('chat')">Chat</button>
    <button onclick="switchMode('instruct')">Instruct</button>
    <button onclick="switchMode('story')">Storywriter</button>
    <button onclick="switchMode('adventure')">Adventure</button>
  </div>
  <div id="uiMode">Mode: chat</div>
  </div>
  <h3>Preset Diff</h3>
  <pre id="presetDiff"></pre>
  <h3>Server Presets</h3>
  <pre id="serverPresets"></pre>
  <h3>Generation History</h3>
  <pre id="genHistory"></pre>
  <h3>Event Log</h3>
  <pre id="eventLog"></pre>
  </div>
  </div>
  <script>
    const KEY = 'hypura_kobold_presets_v2';
    const MODE_KEY = 'hypura_kobold_ui_mode';
    const THEME_KEY = 'hypura_kobold_ui_theme';
    let lastRequest = null;
    let generationBusy = false;
    let latestError = '';

    function parseSamplerOrder(v) {
      return v.split(',').map(x => Number(x.trim())).filter(x => Number.isInteger(x));
    }

    function readForm() {
      const stop = document.getElementById('stop').value.split('\n').map(s => s.trim()).filter(Boolean);
      const tqSo8Off = document.getElementById('tqSo8Off').value;
      const tqSo8Learned = document.getElementById('tqSo8Learned').value;
      const tqTrialityOff = document.getElementById('tqTrialityOff').value;
      const tqTrialityMixRaw = document.getElementById('tqTrialityMix').value.trim();
      const tqRotationSeedRaw = document.getElementById('tqRotationSeed').value.trim();
      const tqArtifact = document.getElementById('tqArtifact').value.trim();
      const body = {
        prompt: document.getElementById('prompt').value,
        max_length: Number(document.getElementById('maxlen').value),
        temperature: Number(document.getElementById('temp').value),
        top_k: Number(document.getElementById('topk').value),
        top_p: Number(document.getElementById('topp').value),
        min_p: Number(document.getElementById('minp').value),
        rep_pen: Number(document.getElementById('repen').value),
        rep_pen_range: Number(document.getElementById('repenRange').value),
        seed: Number(document.getElementById('seed').value),
        top_a: Number(document.getElementById('topa').value),
        tfs: Number(document.getElementById('tfs').value),
        typical: Number(document.getElementById('typical').value),
        mirostat: Number(document.getElementById('mirostat').value),
        mirostat_tau: Number(document.getElementById('mirostatTau').value),
        mirostat_eta: Number(document.getElementById('mirostatEta').value),
        dynatemp_range: Number(document.getElementById('dynatempRange').value),
        dynatemp_exponent: Number(document.getElementById('dynatempExponent').value),
        smoothing_factor: Number(document.getElementById('smoothingFactor').value),
        presence_penalty: Number(document.getElementById('presencePenalty').value),
        frequency_penalty: Number(document.getElementById('frequencyPenalty').value),
        sampler_order: parseSamplerOrder(document.getElementById('samplerOrder').value),
        stop_sequence: stop
      };
      if (tqSo8Off) body.tq_so8_off = tqSo8Off === '1';
      if (tqSo8Learned) body.tq_so8_learned = tqSo8Learned === '1';
      if (tqTrialityOff) body.tq_triality_off = tqTrialityOff === '1';
      if (tqTrialityMixRaw) body.tq_triality_mix = Number(tqTrialityMixRaw);
      if (tqRotationSeedRaw) body.tq_rotation_seed = Number(tqRotationSeedRaw);
      if (tqArtifact) body.tq_artifact = tqArtifact;
      return body;
    }

    function writeForm(v) {
      document.getElementById('prompt').value = v.prompt ?? '';
      document.getElementById('maxlen').value = v.max_length ?? 256;
      document.getElementById('temp').value = v.temperature ?? 0.8;
      document.getElementById('topk').value = v.top_k ?? 40;
      document.getElementById('topp').value = v.top_p ?? 0.9;
      document.getElementById('minp').value = v.min_p ?? 0.05;
      document.getElementById('repen').value = v.rep_pen ?? 1.0;
      document.getElementById('repenRange').value = v.rep_pen_range ?? 64;
      document.getElementById('seed').value = v.seed ?? 42;
      document.getElementById('topa').value = v.top_a ?? 0;
      document.getElementById('tfs').value = v.tfs ?? 1;
      document.getElementById('typical').value = v.typical ?? 1;
      document.getElementById('mirostat').value = v.mirostat ?? 0;
      document.getElementById('mirostatTau').value = v.mirostat_tau ?? 5;
      document.getElementById('mirostatEta').value = v.mirostat_eta ?? 0.1;
      document.getElementById('dynatempRange').value = v.dynatemp_range ?? 0;
      document.getElementById('dynatempExponent').value = v.dynatemp_exponent ?? 1;
      document.getElementById('smoothingFactor').value = v.smoothing_factor ?? 0;
      document.getElementById('presencePenalty').value = v.presence_penalty ?? 0;
      document.getElementById('frequencyPenalty').value = v.frequency_penalty ?? 0;
      document.getElementById('samplerOrder').value = (v.sampler_order ?? [6,0,1,3,4,2,5]).join(',');
      document.getElementById('stop').value = (v.stop_sequence ?? []).join('\n');
      document.getElementById('tqSo8Off').value = v.tq_so8_off === undefined ? '' : (v.tq_so8_off ? '1' : '0');
      document.getElementById('tqSo8Learned').value = v.tq_so8_learned === undefined ? '' : (v.tq_so8_learned ? '1' : '0');
      document.getElementById('tqTrialityOff').value = v.tq_triality_off === undefined ? '' : (v.tq_triality_off ? '1' : '0');
      document.getElementById('tqTrialityMix').value = v.tq_triality_mix ?? '';
      document.getElementById('tqRotationSeed').value = v.tq_rotation_seed ?? '';
      document.getElementById('tqArtifact').value = v.tq_artifact ?? '';
    }

    function applyStage(stage) {
      const stageMap = {
        short: { max_length: 64, prompt: '短く挨拶して。' },
        medium: { max_length: 256, prompt: '数段落で説明して。' },
        long: { max_length: 512, prompt: '長めに説明して。' }
      };
      const cur = readForm();
      const patch = stageMap[stage] || stageMap.short;
      const contextHint = stage === 'long' ? '8192' : '4096';
      writeForm({ ...cur, ...patch });
      setStatus(`Preset applied: ${stage} (recommended context ${contextHint})`, false);
    }

    function setStatus(message, isError = false) {
      const el = document.getElementById('connStatus');
      el.textContent = message;
      el.className = isError ? 'status err' : 'status ok';
      if (isError) latestError = message;
    }

    function setBusy(busy) {
      generationBusy = busy;
      document.getElementById('busyState').textContent = `State: ${busy ? 'generating' : 'idle'}`;
      for (const b of document.querySelectorAll('button')) {
        if (['Abort', 'Recheck Connection', 'Refresh Models', 'Switch Model'].includes(b.textContent.trim())) continue;
        b.disabled = busy;
      }
      document.querySelector('button[onclick="abortGen()"]').disabled = !busy;
    }

    function switchMode(mode) {
      localStorage.setItem(MODE_KEY, mode);
      document.getElementById('uiMode').textContent = `Mode: ${mode}`;
      const presets = {
        chat: { temperature: 0.8, top_p: 0.92, max_length: 256 },
        instruct: { temperature: 0.6, top_p: 0.9, max_length: 320 },
        story: { temperature: 0.95, top_p: 0.96, max_length: 512 },
        adventure: { temperature: 0.85, top_p: 0.94, max_length: 384 }
      };
      writeForm({ ...readForm(), ...(presets[mode] || presets.chat) });
    }

    function applyTheme(theme, source = 'local') {
      const t = ['light', 'dark', 'classic'].includes(theme) ? theme : 'classic';
      document.body.setAttribute('data-theme', t);
      document.getElementById('themeSelect').value = t;
      document.getElementById('themeStatus').textContent = `Theme source: ${source} (${t})`;
      localStorage.setItem(THEME_KEY, t);
    }

    async function syncThemeFromServer() {
      try {
        const r = await fetch('/api/extra/ui-theme');
        const j = await r.json();
        const localTheme = localStorage.getItem(THEME_KEY);
        if (localTheme) {
          applyTheme(localTheme, 'local');
        } else {
          applyTheme(j.theme || 'classic', 'server');
        }
      } catch {
        applyTheme(localStorage.getItem(THEME_KEY) || 'classic', 'fallback');
      }
    }

    async function setThemeServer(theme) {
      try {
        await fetch('/api/extra/ui-theme', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ theme })
        });
      } catch {}
    }

    function changeThemeFromSelect() {
      const theme = document.getElementById('themeSelect').value;
      applyTheme(theme, 'ui');
      setThemeServer(theme);
    }

    async function checkConnection() {
      try {
        const r = await fetch('/api/v1/model');
        const j = await r.json();
        setStatus(`Connection OK: ${j.result || 'unknown model'}`, false);
        document.getElementById('modelStatus').textContent = `Model: ${j.result || '-'}`;
      } catch (e) {
        setStatus(`Connection ERROR: ${String(e)}`, true);
      }
    }

    async function refreshModelList() {
      try {
        const r = await fetch('/api/extra/models');
        const j = await r.json();
        const list = document.getElementById('modelList');
        list.innerHTML = '';
        for (const m of (j.models || [])) {
          const opt = document.createElement('option');
          opt.value = m.path;
          opt.textContent = `${m.name}${m.selected ? ' (active)' : ''}`;
          if (m.selected) opt.selected = true;
          list.appendChild(opt);
        }
      } catch (e) {
        setStatus(`Model list ERROR: ${String(e)}`, true);
      }
    }

    async function switchModel() {
      if (generationBusy) return;
      const path = document.getElementById('modelList').value;
      if (!path) return;
      setBusy(true);
      setStatus('Switching model...', false);
      try {
        const r = await fetch('/api/extra/model/switch', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ path })
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
        setStatus(`Model switched: ${j.model} (context=${j.context})`, false);
        document.getElementById('modelStatus').textContent = `Model: ${j.model}`;
        await refreshModelList();
      } catch (e) {
        setStatus(`Switch ERROR: ${String(e)}`, true);
      } finally {
        setBusy(false);
      }
    }

    function toCliArgs(v) {
      const args = [
        '--temperature', String(v.temperature),
        '--top-k', String(v.top_k),
        '--top-p', String(v.top_p),
        '--min-p', String(v.min_p),
        '--repeat-penalty', String(v.rep_pen),
        '--repeat-last-n', String(v.rep_pen_range),
        '--seed', String(v.seed),
      ];
      if (v.top_a !== undefined) args.push('--top-a', String(v.top_a));
      if (v.tfs !== undefined) args.push('--tfs', String(v.tfs));
      if (v.typical !== undefined) args.push('--typical', String(v.typical));
      if (v.mirostat !== undefined) args.push('--mirostat', String(v.mirostat));
      if (v.mirostat_tau !== undefined) args.push('--mirostat-tau', String(v.mirostat_tau));
      if (v.mirostat_eta !== undefined) args.push('--mirostat-eta', String(v.mirostat_eta));
      if (v.dynatemp_range !== undefined) args.push('--dynatemp-range', String(v.dynatemp_range));
      if (v.dynatemp_exponent !== undefined) args.push('--dynatemp-exp', String(v.dynatemp_exponent));
      if (v.smoothing_factor !== undefined) args.push('--smoothing-factor', String(v.smoothing_factor));
      if (v.presence_penalty !== undefined) args.push('--presence-penalty', String(v.presence_penalty));
      if (v.frequency_penalty !== undefined) args.push('--frequency-penalty', String(v.frequency_penalty));
      if (Array.isArray(v.sampler_order) && v.sampler_order.length > 0) args.push('--sampler-order', `"${v.sampler_order.join(',')}"`);
      if (Array.isArray(v.stop_sequence)) for (const s of v.stop_sequence) args.push('--stop', `"${s}"`);
      if (typeof v.tq_so8_off === 'boolean') args.push('--tq-so8-off', v.tq_so8_off ? 'true' : 'false');
      if (typeof v.tq_so8_learned === 'boolean') args.push('--tq-so8-learned', v.tq_so8_learned ? 'true' : 'false');
      if (typeof v.tq_triality_off === 'boolean') args.push('--tq-triality-off', v.tq_triality_off ? 'true' : 'false');
      if (v.tq_triality_mix !== undefined) args.push('--tq-triality-mix', String(v.tq_triality_mix));
      if (v.tq_rotation_seed !== undefined) args.push('--tq-rotation-seed', String(v.tq_rotation_seed));
      if (v.tq_artifact) args.push('--tq-artifact', `"${v.tq_artifact}"`);
      return `hypura serve --model ./model.gguf ${args.join(' ')}`;
    }

    function parseCliBridge(input) {
      const out = {};
      const read = (flag) => {
        const i = input.indexOf(flag);
        if (i < 0) return null;
        const rest = input.slice(i + flag.length).trim();
        const m = rest.match(/^"([^"]+)"|^(\S+)/);
        return m ? (m[1] ?? m[2]) : null;
      };
      const asNum = (v) => (v == null ? undefined : Number(v));
      out.temperature = asNum(read('--temperature'));
      out.top_k = asNum(read('--top-k'));
      out.top_p = asNum(read('--top-p'));
      out.min_p = asNum(read('--min-p'));
      out.rep_pen = asNum(read('--repeat-penalty'));
      out.rep_pen_range = asNum(read('--repeat-last-n'));
      out.seed = asNum(read('--seed'));
      out.top_a = asNum(read('--top-a'));
      out.tfs = asNum(read('--tfs'));
      out.typical = asNum(read('--typical'));
      out.mirostat = asNum(read('--mirostat'));
      out.mirostat_tau = asNum(read('--mirostat-tau'));
      out.mirostat_eta = asNum(read('--mirostat-eta'));
      out.dynatemp_range = asNum(read('--dynatemp-range'));
      out.dynatemp_exponent = asNum(read('--dynatemp-exp'));
      out.smoothing_factor = asNum(read('--smoothing-factor'));
      out.presence_penalty = asNum(read('--presence-penalty'));
      out.frequency_penalty = asNum(read('--frequency-penalty'));
      const order = read('--sampler-order');
      if (order) out.sampler_order = parseSamplerOrder(order);
      const tqSo8Off = read('--tq-so8-off');
      if (tqSo8Off) out.tq_so8_off = tqSo8Off === 'true';
      const tqSo8Learned = read('--tq-so8-learned');
      if (tqSo8Learned) out.tq_so8_learned = tqSo8Learned === 'true';
      const tqTrialityOff = read('--tq-triality-off');
      if (tqTrialityOff) out.tq_triality_off = tqTrialityOff === 'true';
      out.tq_triality_mix = asNum(read('--tq-triality-mix'));
      out.tq_rotation_seed = asNum(read('--tq-rotation-seed'));
      const tqArtifact = read('--tq-artifact');
      if (tqArtifact) out.tq_artifact = tqArtifact;
      return out;
    }

    function exportPresetCli() {
      const cmd = toCliArgs(readForm());
      document.getElementById('cliBridge').value = cmd;
    }

    function importPresetCli() {
      const input = document.getElementById('cliBridge').value;
      const parsed = parseCliBridge(input);
      writeForm({ ...readForm(), ...parsed });
    }

    function presets() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; } }
    function persist(p) { localStorage.setItem(KEY, JSON.stringify(p)); refreshPresetList(); }
    function refreshPresetList() {
      const list = document.getElementById('presetList');
      const p = presets();
      list.innerHTML = '';
      Object.keys(p).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        list.appendChild(opt);
      });
    }

    function savePreset() {
      const name = (document.getElementById('presetName').value || '').trim();
      if (!name) return;
      const p = presets();
      p[name] = { ...readForm(), _meta: { version: 2, saved_at: new Date().toISOString() } };
      persist(p);
      savePresetServer(name, p[name]);
    }

    function loadPreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (name && p[name]) writeForm(p[name]);
    }

    function deletePreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (name && p[name]) {
        delete p[name];
        persist(p);
        deletePresetServer(name);
      }
    }

    async function savePresetServer(name, payload) {
      try {
        await fetch('/api/extra/presets/save', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ name, payload })
        });
        await refreshServerPanels();
      } catch {}
    }

    async function deletePresetServer(name) {
      try {
        await fetch('/api/extra/presets/delete', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ name })
        });
        await refreshServerPanels();
      } catch {}
    }

    async function refreshServerPanels() {
      try {
        const [presetRes, historyRes, eventRes] = await Promise.all([
          fetch('/api/extra/presets/list'),
          fetch('/api/extra/history'),
          fetch('/api/extra/events')
        ]);
        const presetJson = await presetRes.json();
        const historyJson = await historyRes.json();
        const eventJson = await eventRes.json();
        document.getElementById('serverPresets').textContent = JSON.stringify(presetJson, null, 2);
        document.getElementById('genHistory').textContent = JSON.stringify(historyJson, null, 2);
        document.getElementById('eventLog').textContent = JSON.stringify(eventJson, null, 2);
      } catch (e) {
        document.getElementById('eventLog').textContent = `[panel refresh error] ${String(e)}`;
      }
    }

    function diffPreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (!name || !p[name]) return;
      const current = readForm();
      const saved = p[name];
      const keys = Array.from(new Set([...Object.keys(current), ...Object.keys(saved)])).sort();
      const diffs = [];
      for (const k of keys) {
        if (k === '_meta') continue;
        const a = JSON.stringify(current[k]);
        const b = JSON.stringify(saved[k]);
        if (a !== b) diffs.push(`${k}: current=${a} | preset=${b}`);
      }
      document.getElementById('presetDiff').textContent = diffs.length > 0 ? diffs.join('\n') : 'no diff';
    }

    async function generateOnce() {
      if (generationBusy) return;
      const body = readForm();
      lastRequest = body;
      document.getElementById('out').textContent = 'Generating...';
      setBusy(true);
      const startedAt = performance.now();
      try {
        const r = await fetch('/api/v1/generate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const j = await r.json();
        const text = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
        document.getElementById('out').textContent = text;
        document.getElementById('lastSuccess').textContent = `Last success: ${new Date().toLocaleTimeString()}`;
        document.getElementById('mEval').textContent = String(text.length);
        document.getElementById('mTotal').textContent = (performance.now() - startedAt).toFixed(1);
        document.getElementById('tokenProgress').textContent = `Progress: ${text.length} chars / ${(performance.now() - startedAt).toFixed(0)} ms`;
      } catch (e) {
        document.getElementById('out').textContent = `[error] ${String(e)}`;
        setStatus(`Generate ERROR: ${String(e)}`, true);
      } finally {
        await refreshServerPanels();
        setBusy(false);
      }
    }

    async function generateStream() {
      if (generationBusy) return;
      const body = { ...readForm(), stream: true };
      lastRequest = body;
      document.getElementById('out').textContent = '';
      const startedAt = performance.now();
      setBusy(true);
      try {
        const r = await fetch('/api/extra/generate/stream', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let tokenCount = 0;
        let latestTokPerSec = null;
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          buf += dec.decode(value, {stream: true});
          const lines = buf.split('\n');
          buf = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const obj = JSON.parse(line);
              if (obj.token) {
                tokenCount += 1;
                latestTokPerSec = obj.tok_per_sec ?? latestTokPerSec;
                document.getElementById('out').textContent += obj.token;
                document.getElementById('tokenProgress').textContent = `Progress: ${tokenCount} token / ${(performance.now() - startedAt).toFixed(0)} ms`;
              }
              if (obj.done) {
                const elapsedMs = Math.max(1, performance.now() - startedAt);
                const tps = latestTokPerSec ?? ((tokenCount * 1000.0) / elapsedMs);
                document.getElementById('mTok').textContent = tps.toFixed(2);
                document.getElementById('mEval').textContent = String(obj.eval_count ?? '-');
                document.getElementById('mTotal').textContent = (obj.total_duration ? obj.total_duration / 1e6 : elapsedMs).toFixed(1);
                document.getElementById('mPrompt').textContent = String(obj.prompt_eval_ms ?? '-');
                document.getElementById('lastSuccess').textContent = `Last success: ${new Date().toLocaleTimeString()}`;
                document.getElementById('out').textContent += '\n\n[done]';
              }
            } catch {}
          }
        }
      } catch (e) {
        document.getElementById('out').textContent += `\n\n[error] ${String(e)}`;
        setStatus(`Stream ERROR: ${String(e)}`, true);
      } finally {
        await refreshServerPanels();
        setBusy(false);
      }
    }

    async function retryLast() {
      if (!lastRequest) return;
      document.getElementById('out').textContent = '';
      writeForm(lastRequest);
      await generateStream();
    }

    async function abortGen() {
      await fetch('/api/extra/abort', { method: 'POST' });
      setBusy(false);
    }

    async function checkStatus() {
      const r = await fetch('/api/extra/generate/check');
      const j = await r.json();
      const msg = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
      document.getElementById('out').textContent += '\n\n[check] ' + msg;
    }

    refreshPresetList();
    setBusy(false);
    syncThemeFromServer();
    switchMode(localStorage.getItem(MODE_KEY) || 'chat');
    refreshModelList();
    checkConnection();
    refreshServerPanels();
    setInterval(refreshServerPanels, 5000);
  </script>
</body>
</html>"#,
    )
}

async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let info = state.gguf_info.lock().unwrap().clone();
    let model_name = state.model_name.lock().unwrap().clone();
    Json(TagsResponse {
        models: vec![ModelTag {
            name: model_name.clone(),
            model: model_name,
            size: info.file_size,
            details: ModelDetails {
                format: "gguf".into(),
                family: info.architecture,
                parameter_size: format_parameter_size(info.parameter_count),
                quantization_level: info.quantization,
            },
        }],
    })
}

async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShowRequest>,
) -> Json<ShowResponse> {
    let info = state.gguf_info.lock().unwrap().clone();
    let model_name = state.model_name.lock().unwrap().clone();
    let requested_model = req
        .name
        .as_deref()
        .or(req.model.as_deref())
        .unwrap_or(model_name.as_str())
        .to_string();
    Json(ShowResponse {
        details: ModelDetails {
            format: "gguf".into(),
            family: info.architecture.clone(),
            parameter_size: format_parameter_size(info.parameter_count),
            quantization_level: info.quantization.clone(),
        },
        model_info: serde_json::json!({
            "general.name": requested_model,
            "general.architecture": info.architecture,
            "general.context_length": info.context_length,
            "general.parameter_count": info.parameter_count,
            "hypura.turboquant.mode": state.turboquant.mode.as_str(),
            "hypura.turboquant.schema": state.turboquant.schema_label(),
            "hypura.turboquant.config_path": state.turboquant.source_label(),
            "hypura.turboquant.runtime_status": turboquant_runtime_status(
                state.turboquant.mode,
                state.turboquant.config.is_some(),
            ),
        }),
    })
}

async fn kobold_model_handler(State(state): State<Arc<AppState>>) -> Json<KoboldModelResponse> {
    let model_name = state.model_name.lock().unwrap().clone();
    Json(KoboldModelResponse {
        result: model_name,
    })
}

fn collect_gguf_models(model_dir: &PathBuf, active_path: &str) -> anyhow::Result<Vec<AvailableModelItem>> {
    let mut models = Vec::new();
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        let full = path.to_string_lossy().to_string();
        let name = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| full.clone());
        models.push(AvailableModelItem {
            name,
            selected: full.eq_ignore_ascii_case(active_path),
            path: full,
        });
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

async fn models_handler(State(state): State<Arc<AppState>>) -> Response {
    let active_path = state.model_path.lock().unwrap().to_string_lossy().to_string();
    match collect_gguf_models(&state.model_dir, &active_path) {
        Ok(models) => Json(AvailableModelsResponse {
            models,
            active_model_path: active_path,
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("failed to scan models: {e}") })),
        )
            .into_response(),
    }
}

async fn model_switch_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelSwitchRequest>,
) -> Response {
    if state.generation_in_progress.load(Ordering::Relaxed) {
        push_gui_event(&state, "warn", "model switch rejected: generation in progress");
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "generation in progress; abort first" })),
        )
            .into_response();
    }

    let next_model_path = PathBuf::from(req.path.trim());
    if !next_model_path.exists() {
        push_gui_event(&state, "error", "model switch failed: model path does not exist");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "model path does not exist" })),
        )
            .into_response();
    }

    let context = req.context.unwrap_or(state.default_context).max(256);
    let path_for_setup = next_model_path.clone();
    let setup = match tokio::task::spawn_blocking(move || {
        crate::compute::inference::resolve_runtime_setup(
            &path_for_setup,
            context,
            TurboQuantMode::Exact,
            None,
        )
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            push_gui_event(&state, "error", format!("model inspect failed: {e}"));
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("failed to inspect model: {e}") })),
            )
                .into_response();
        }
        Err(e) => {
            push_gui_event(&state, "error", format!("model inspect task failed: {e}"));
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("runtime task failed: {e}") })),
            )
                .into_response();
        }
    };

    let file_size = match fs::metadata(&next_model_path) {
        Ok(m) => m.len(),
        Err(e) => {
            push_gui_event(&state, "error", format!("model metadata read failed: {e}"));
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("failed to read model metadata: {e}") })),
            )
                .into_response();
        }
    };

    let gguf_info = GgufInfo {
        file_size,
        architecture: setup.metadata.architecture.clone(),
        parameter_count: setup.metadata.parameter_count,
        quantization: setup
            .metadata
            .quantization
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        context_length: setup.metadata.context_length,
    };

    let config = crate::compute::inference::InferenceConfig {
        n_ctx: context,
        ..crate::compute::inference::InferenceConfig::default()
    };
    let n_gpu_layers = setup.n_gpu_layers;
    let plan = setup.plan.clone();
    let gguf = setup.gguf.clone();
    let turboquant = setup.turboquant.clone();
    let path_for_load = next_model_path.clone();

    let loaded_next = match tokio::task::spawn_blocking(move || {
        crate::compute::inference::load_model(
            &path_for_load,
            &config,
            n_gpu_layers,
            &plan,
            &gguf,
            &turboquant,
        )
    })
    .await
    {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            push_gui_event(&state, "error", format!("model load failed: {e}"));
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("failed to load model: {e}") })),
            )
                .into_response();
        }
        Err(e) => {
            push_gui_event(&state, "error", format!("model load task failed: {e}"));
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("load task failed: {e}") })),
            )
                .into_response();
        }
    };

    let mut loaded_guard = state.loaded_model.lock().unwrap();
    *loaded_guard = loaded_next;
    drop(loaded_guard);

    let next_model_name = state.loaded_model.lock().unwrap().model_name.clone();
    *state.model_name.lock().unwrap() = next_model_name;
    *state.model_path.lock().unwrap() = next_model_path;
    *state.gguf_info.lock().unwrap() = gguf_info;

    let model_name = state.model_name.lock().unwrap().clone();
    push_gui_event(&state, "info", format!("model switched: {model_name} (ctx={context})"));
    Json(ModelSwitchResponse {
        success: true,
        model: model_name,
        context,
    })
    .into_response()
}

fn push_gui_event(state: &Arc<AppState>, level: &str, message: impl Into<String>) {
    let mut events = state.gui_events.lock().unwrap();
    events.push_front(GuiEventItem {
        ts: now_rfc3339(),
        level: level.to_string(),
        message: message.into(),
    });
    while events.len() > 200 {
        events.pop_back();
    }
}

fn push_gui_history(
    state: &Arc<AppState>,
    mode: &str,
    model: String,
    prompt_chars: usize,
    output_chars: usize,
    tok_per_sec_avg: Option<f64>,
    total_ms: u64,
) {
    let mut history = state.gui_history.lock().unwrap();
    history.push_front(GuiHistoryItem {
        ts: now_rfc3339(),
        mode: mode.to_string(),
        model,
        prompt_chars,
        output_chars,
        tok_per_sec_avg,
        total_ms,
    });
    while history.len() > 200 {
        history.pop_back();
    }
}

async fn gui_presets_list_handler(State(state): State<Arc<AppState>>) -> Json<GuiPresetListResponse> {
    let mut presets: Vec<GuiPresetItem> = state.gui_presets.lock().unwrap().values().cloned().collect();
    presets.sort_by(|a, b| a.name.cmp(&b.name));
    Json(GuiPresetListResponse { presets })
}

async fn gui_presets_save_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuiPresetSaveRequest>,
) -> Response {
    let name = req.name.trim();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    state.gui_presets.lock().unwrap().insert(
        name.to_string(),
        GuiPresetItem {
            name: name.to_string(),
            payload: req.payload,
            updated_at: now_rfc3339(),
        },
    );
    push_gui_event(&state, "info", format!("preset saved: {name}"));
    Json(serde_json::json!({ "success": true })).into_response()
}

async fn gui_presets_delete_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuiPresetDeleteRequest>,
) -> Response {
    let name = req.name.trim();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    state.gui_presets.lock().unwrap().remove(name);
    push_gui_event(&state, "info", format!("preset deleted: {name}"));
    Json(serde_json::json!({ "success": true })).into_response()
}

async fn gui_history_handler(State(state): State<Arc<AppState>>) -> Json<GuiHistoryResponse> {
    let items = state.gui_history.lock().unwrap().iter().cloned().collect();
    Json(GuiHistoryResponse { items })
}

async fn gui_events_handler(State(state): State<Arc<AppState>>) -> Json<GuiEventsResponse> {
    let items = state.gui_events.lock().unwrap().iter().cloned().collect();
    Json(GuiEventsResponse { items })
}

async fn ui_theme_get_handler(State(state): State<Arc<AppState>>) -> Json<UiThemeResponse> {
    Json(UiThemeResponse {
        theme: state.ui_theme.lock().unwrap().clone(),
    })
}

async fn ui_theme_set_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UiThemeUpdateRequest>,
) -> Response {
    let t = req.theme.trim().to_ascii_lowercase();
    let next = match t.as_str() {
        "light" | "dark" | "classic" => t,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "theme must be light|dark|classic" })),
            )
                .into_response()
        }
    };
    *state.ui_theme.lock().unwrap() = next.clone();
    std::env::set_var("HYPURA_UI_THEME", &next);
    push_gui_event(&state, "info", format!("ui theme changed: {next}"));
    Json(serde_json::json!({ "success": true, "theme": next })).into_response()
}

fn begin_generation(state: &Arc<AppState>) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut slot = state.active_cancel.lock().unwrap();
        *slot = Some(cancel.clone());
    }
    state.generation_in_progress.store(true, Ordering::Relaxed);
    cancel
}

fn end_generation(state: &Arc<AppState>) {
    state.generation_in_progress.store(false, Ordering::Relaxed);
    let mut slot = state.active_cancel.lock().unwrap();
    *slot = None;
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    apply_turboquant_runtime_overrides(&req.options);
    let sampling = build_sampling(&req.options);
    let stop_sequences = req.options.stop.clone().unwrap_or_default();
    let prompt = req.prompt;
    let model_name = state.model_name.lock().unwrap().clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let sampler_order = req.options.sampler_order.clone();

    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        match result {
            Ok(gen_result) => {
                let model_name = state_for_task.model_name.lock().unwrap().clone();
                push_gui_history(
                    &state_for_task,
                    "generate",
                    model_name,
                    prompt.chars().count(),
                    gen_result.text.chars().count(),
                    Some(gen_result.tok_per_sec_avg),
                    started.elapsed().as_millis() as u64,
                );
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Generation error: {e}");
                push_gui_event(&state_for_task, "error", format!("generate failed: {e}"));
            }
        }
    });

    if req.stream {
        let body = streaming::ndjson_generate_stream(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        );
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        // Non-streaming: collect all tokens, return single JSON
        let result = collect_generate(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        )
        .await;
        Json(result).into_response()
    }
}

async fn kobold_generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> Json<KoboldGenerateResponse> {
    apply_turboquant_runtime_overrides_kobold(&req);
    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(k) = req.top_k {
        sampling.top_k = k;
    }
    if let Some(a) = req.top_a {
        sampling.top_a = a;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(tfs) = req.tfs {
        sampling.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        sampling.typical = typ;
    }
    if let Some(mp) = req.min_p {
        sampling.min_p = mp;
    }
    if let Some(n) = req.max_length {
        sampling.max_tokens = n;
    }
    if let Some(rep_pen) = req.rep_pen {
        sampling.repeat_penalty = rep_pen;
    }
    if let Some(rep_range) = req.rep_pen_range {
        sampling.repeat_last_n = rep_range;
    }
    let sampler_order = req.sampler_order.clone();
    let stop_sequences = req.stop_sequence.unwrap_or_default();

    let prompt = req.prompt;
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let (result_tx, _result_rx) = oneshot::channel::<GenerationResult>();
    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();

    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        if let Ok(gen_result) = result {
            let model_name = state_for_task.model_name.lock().unwrap().clone();
            push_gui_history(
                &state_for_task,
                "kobold-generate",
                model_name,
                prompt.chars().count(),
                gen_result.text.chars().count(),
                Some(gen_result.tok_per_sec_avg),
                started.elapsed().as_millis() as u64,
            );
            let _ = result_tx.send(gen_result);
        } else if let Err(e) = result {
            push_gui_event(&state_for_task, "error", format!("kobold generate failed: {e}"));
        }
    });

    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }

    Json(KoboldGenerateResponse {
        results: vec![KoboldGenerateResult {
            text: full_response,
        }],
    })
}

async fn kobold_generate_stream_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> Response {
    apply_turboquant_runtime_overrides_kobold(&req);
    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(k) = req.top_k {
        sampling.top_k = k;
    }
    if let Some(a) = req.top_a {
        sampling.top_a = a;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(tfs) = req.tfs {
        sampling.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        sampling.typical = typ;
    }
    if let Some(mp) = req.min_p {
        sampling.min_p = mp;
    }
    if let Some(n) = req.max_length {
        sampling.max_tokens = n;
    }
    if let Some(rep_pen) = req.rep_pen {
        sampling.repeat_penalty = rep_pen;
    }
    if let Some(rep_range) = req.rep_pen_range {
        sampling.repeat_last_n = rep_range;
    }
    let sampler_order = req.sampler_order.clone();
    let stop_sequences = req.stop_sequence.unwrap_or_default();
    let prompt = req.prompt;

    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let model_name = state.model_name.lock().unwrap().clone();
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        if let Ok(gen_result) = result {
            let model_name_inner = state_for_task.model_name.lock().unwrap().clone();
            push_gui_history(
                &state_for_task,
                "kobold-stream",
                model_name_inner,
                prompt.chars().count(),
                gen_result.text.chars().count(),
                Some(gen_result.tok_per_sec_avg),
                started.elapsed().as_millis() as u64,
            );
            let _ = result_tx.send(gen_result);
        } else if let Err(e) = result {
            push_gui_event(&state_for_task, "error", format!("kobold stream failed: {e}"));
        }
    });

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);
    tokio::spawn(async move {
        while let Some(token) = token_rx.recv().await {
            let line = serde_json::json!({
                "token": token.text,
                "done": false,
                "model": model_name.clone(),
            })
            .to_string()
                + "\n";
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }
        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        let line = serde_json::json!({
            "token": "",
            "done": true,
            "model": model_name.clone(),
            "total_duration": total_ns,
            "load_duration": load_duration_ns,
            "eval_count": result.as_ref().map(|r| r.tokens_generated),
            "tok_per_sec_avg": result.as_ref().map(|r| r.tok_per_sec_avg),
            "prompt_eval_ms": result.as_ref().map(|r| r.prompt_eval_ms),
        })
        .to_string()
            + "\n";
        let _ = tx.send(Ok(line)).await;
    });

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
        .into_response()
}

async fn kobold_generate_check_handler(State(state): State<Arc<AppState>>) -> Json<KoboldGenerateResponse> {
    let status = if state.generation_in_progress.load(Ordering::Relaxed) {
        "busy"
    } else {
        ""
    };
    Json(KoboldGenerateResponse {
        results: vec![KoboldGenerateResult {
            text: status.to_string(),
        }],
    })
}

async fn kobold_abort_handler(State(state): State<Arc<AppState>>) -> Json<KoboldAbortResponse> {
    let mut success = false;
    if let Some(flag) = state.active_cancel.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
        success = true;
    }
    Json(KoboldAbortResponse { success })
}

async fn kobold_true_max_context_length_handler(
    State(state): State<Arc<AppState>>,
) -> Json<KoboldTrueMaxContextLengthResponse> {
    Json(KoboldTrueMaxContextLengthResponse {
        value: state.gguf_info.lock().unwrap().context_length,
    })
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    apply_turboquant_runtime_overrides(&req.options);
    let sampling = build_sampling(&req.options);
    let stop_sequences = req.options.stop.clone().unwrap_or_default();
    let prompt = format_chat_prompt(&req.messages);
    let model_name = state.model_name.lock().unwrap().clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let sampler_order = req.options.sampler_order.clone();

    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        match result {
            Ok(gen_result) => {
                let model_name = state_for_task.model_name.lock().unwrap().clone();
                push_gui_history(
                    &state_for_task,
                    "chat",
                    model_name,
                    prompt.chars().count(),
                    gen_result.text.chars().count(),
                    Some(gen_result.tok_per_sec_avg),
                    started.elapsed().as_millis() as u64,
                );
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Chat generation error: {e}");
                push_gui_event(&state_for_task, "error", format!("chat failed: {e}"));
            }
        }
    });

    if req.stream {
        let body = streaming::ndjson_chat_stream(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        );
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        let result = collect_chat(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        )
        .await;
        Json(result).into_response()
    }
}

// ── Helpers ──

fn build_sampling(opts: &GenerateOptions) -> crate::compute::ffi::SamplingParams {
    let mut s = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = opts.temperature {
        s.temperature = t;
    }
    if let Some(k) = opts.top_k {
        s.top_k = k;
    }
    if let Some(a) = opts.top_a {
        s.top_a = a;
    }
    if let Some(p) = opts.top_p {
        s.top_p = p;
    }
    if let Some(tfs) = opts.tfs {
        s.tfs = tfs;
    }
    if let Some(typ) = opts.typical {
        s.typical = typ;
    }
    if let Some(mp) = opts.min_p {
        s.min_p = mp;
    }
    if let Some(rp) = opts.repeat_penalty {
        s.repeat_penalty = rp;
    }
    if let Some(rn) = opts.repeat_last_n {
        s.repeat_last_n = rn;
    }
    if let Some(n) = opts.num_predict {
        s.max_tokens = n;
    }
    if let Some(seed) = opts.seed {
        s.seed = seed;
    }
    s
}

fn apply_turboquant_runtime_overrides(opts: &GenerateOptions) {
    if let Some(v) = opts.tq_so8_off {
        std::env::set_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        std::env::set_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        std::env::set_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            std::env::remove_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            std::env::set_var("LLAMA_TURBOQUANT_ARTIFACT", path);
        }
    }
}

fn apply_turboquant_runtime_overrides_kobold(opts: &KoboldGenerateRequest) {
    if let Some(v) = opts.tq_so8_off {
        std::env::set_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        std::env::set_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        std::env::set_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            std::env::remove_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            std::env::set_var("LLAMA_TURBOQUANT_ARTIFACT", path);
        }
    }
}

fn turboquant_runtime_status(mode: TurboQuantMode, has_config: bool) -> &'static str {
    if mode == TurboQuantMode::Exact {
        "inactive"
    } else if !has_config {
        "unresolved"
    } else if mode == TurboQuantMode::PaperFullKv {
        "experimental-full-kv"
    } else if has_config {
        "faithful-attached"
    } else {
        "unresolved"
    }
}

async fn collect_generate(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> GenerateResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    GenerateResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        response: full_response,
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

async fn collect_chat(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> ChatResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    ChatResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content: full_response,
        },
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}
