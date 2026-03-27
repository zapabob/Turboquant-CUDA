# Hypura TurboQuant Handoff Plan

Date: 2026-03-27  
Author: Codex  
Target repo: `https://github.com/zapabob/hypura`  
Inspected local clone: `H:\tmp\hypura`

## 1. Executive Summary

This handoff is for integrating TurboQuant-style KV-cache codecs from
`Turboquant-CUDA` into `hypura`, while keeping the responsibilities clear:

- `Turboquant-CUDA` remains the math/reference repo.
- `hypura` becomes the runtime/GGUF integration repo.
- The first production target in `hypura` should be **paper-faithful K-only**.
- `full_kv` should be added as a validation path, not as the default runtime path.
- K/V-separated research codecs should be added only after the paper baseline path
  is stable and benchmarked.

The practical conclusion from the current Qwen3.5-9B evidence is:

- KV reduction is real.
- `K` compression is comparatively robust.
- `V` compression is the fragile path.
- Therefore the safest migration order in `hypura` is:
  1. sidecar config loading
  2. K-only paper baseline runtime
  3. optional full-KV paper baseline
  4. research K/V-separated codecs

## 2. What Hypura Already Has

`hypura` is already close to the right runtime layer for this work:

- It is a Rust runtime around llama.cpp FFI.
- It has GGUF parsing.
- It has explicit KV cache handling.
- It has a model loading path and a serving path.
- It already distinguishes compute, model metadata, cache, scheduler, and I/O.

Relevant files discovered during inspection:

- `src/model/gguf.rs`
  - GGUF header/metadata reader
  - best place to attach GGUF metadata discovery for TurboQuant
- `src/compute/inference.rs`
  - model load path
  - session config
  - `LoadedModel`
  - generation entrypoints
- `src/cache/kv_cache.rs`
  - current KV cache manager / windowing logic
  - natural insertion point for a TurboQuant-aware KV backend
- `src/cli/run.rs`
  - CLI load path for single prompt / interactive mode
- `src/cli/serve.rs`
  - service startup path
- `src/server/routes.rs`
  - request-level serving integration point
- `src/compute/nvme_backend.rs`
  - backend/eval callback path
  - if llama.cpp callback-level hooks are needed later, this is one likely site
- `src/model/metadata.rs`
  - model metadata normalization helper, likely useful for sidecar wiring

Also relevant:

- `Cargo.toml`
  - currently `license = "MIT"`
- `RESEARCH_INTEGRATION_PLAN.md`
  - ongoing systems/research roadmap already exists

## 3. Integration Goal

The handoff target is **not** "convert TurboQuant into a weight-quantized GGUF".

The target is:

- GGUF remains the weight container.
- TurboQuant becomes a **KV-cache runtime codec**.
- `hypura` loads:
  - `model.gguf`
  - `turboquant_config.paper.json` or `turboquant_config.research.json`
  - optional artifact blob / sidecar tensors
- During generation, `hypura` compresses K/V as tokens are appended and
  reconstructs or estimates as needed during attention.

## 4. Recommended Scope Split

### Phase 0: Sidecar + plumbing only

No math port yet. Only wire formats and configuration.

Deliverables:

- A `TurboQuantSidecarConfig` Rust type
- JSON loader for:
  - `turboquant_config.paper.json`
  - `turboquant_config.research.json`
- CLI flags / env support to pass a sidecar path
- Validation that the config kind matches the selected runtime mode

Why first:

- This de-risks file format and launch UX.
- It cleanly separates GGUF concerns from codec concerns.
- It lets later agents work independently on math/runtime pieces.

### Phase 1: Paper-faithful K-only runtime

This is the first real runtime target.

Behavior:

- K uses paper baseline:
  - explicit norm
  - random Haar rotation
  - coordinate-wise Lloyd-Max
  - residual QJL estimator
- V remains exact

Why first:

- It is the strongest path from the Qwen evidence.
- It avoids the main failure mode already seen in `full_kv`.
- It gives a runtime bridge that is aligned with both paper and empirical safety.

### Phase 2: Paper-faithful full-KV runtime

Behavior:

- K and V both use paper baseline codec

Role:

- validation path only
- benchmark path only
- not the default serving mode

Why:

- Needed to test paper-faithful deployment inside `hypura`
- But current evidence suggests it should not be treated as the default runtime

### Phase 3: Research codecs

Only after Phase 1/2 are stable.

Behavior:

- K may keep paper-faithful or structured-rotation variants
- V may use:
  - protected-V
  - protected-V + low-rank residual
  - sensitivity-aware exact channel retention

## 5. Exact File/Module Plan For Hypura

## 5.1 Config and sidecar loading

Primary files:

- `src/cli/run.rs`
- `src/cli/serve.rs`
- `src/model/gguf.rs`
- `src/model/metadata.rs`
- new module recommended:
  - `src/model/turboquant_sidecar.rs`

Tasks:

1. Add sidecar path option to CLI.
2. Add auto-discovery convention:
   - `model.gguf`
   - `model.turboquant_config.paper.json`
   - `model.turboquant_config.research.json`
3. Define Rust enums:
   - `TurboQuantSchemaKind`
   - `PaperTurboQuantConfig`
   - `ResearchTurboQuantConfig`
4. Validate schema kind before model load finishes.
5. Expose selected config inside `LoadedModel`.

Acceptance:

- `hypura run model.gguf --turboquant-config <path>` loads and validates config
- missing or mismatched config fails early and clearly

## 5.2 KV runtime abstraction

Primary files:

- `src/cache/kv_cache.rs`
- new module recommended:
  - `src/cache/turboquant.rs`
  - or
  - `src/cache/kv_codec.rs`

Tasks:

1. Refactor KV cache manager so cache storage and cache policy are separate.
2. Introduce a codec trait, for example:

```rust
trait KvCodec {
    fn append_k(&mut self, layer: u32, head: u32, token: u32, data: &[f32]) -> anyhow::Result<()>;
    fn append_v(&mut self, layer: u32, head: u32, token: u32, data: &[f32]) -> anyhow::Result<()>;
    fn score_k(&self, layer: u32, head: u32, query: &[f32], token_range: std::ops::Range<u32>) -> anyhow::Result<Vec<f32>>;
    fn read_v(&self, layer: u32, head: u32, token_range: std::ops::Range<u32>) -> anyhow::Result<Vec<f32>>;
}
```

3. Implement at least:
   - `ExactKvCodec`
   - `PaperKeyOnlyCodec`
   - placeholder `PaperFullKvCodec`

Acceptance:

- Existing exact behavior preserved under `ExactKvCodec`
- K-only path can be enabled without touching V logic

## 5.3 Inference path integration

Primary files:

- `src/compute/inference.rs`
- possibly `src/compute/backend.rs`
- possibly `src/compute/ffi.rs`

Tasks:

1. Store the selected TurboQuant codec/config inside `LoadedModel`.
2. During generation setup, initialize per-layer/per-head codec state.
3. Route K append/read through the codec.
4. Keep V exact in Phase 1.
5. Add explicit runtime mode enum, for example:
   - `Exact`
   - `PaperKeyOnly`
   - `PaperFullKv`
   - `ResearchKvSplit`

Acceptance:

- One prompt can run in exact mode and paper key-only mode from the same executable
- The mode is visible in logs and bench output

## 5.4 Serving and API surface

Primary files:

- `src/cli/serve.rs`
- `src/server/routes.rs`
- `src/server/chat.rs`
- `src/server/ollama_types.rs`

Tasks:

1. Add server startup config for TurboQuant mode.
2. Surface the active mode in `/api/show`-style metadata if appropriate.
3. Ensure request handlers do not silently fall back from sidecar mode to exact mode.

Acceptance:

- Serving startup prints:
  - schema kind
  - runtime mode
  - config path

## 6. Sidecar Format Recommendation

Do not embed full TurboQuant runtime state into GGUF first.

Use sidecar files first:

- `model.gguf`
- `turboquant_config.paper.json`
- `turboquant_config.research.json`
- optional future artifact blob:
  - `turboquant_artifacts.bin`
  - or `turboquant_artifacts.safetensors`

Reason:

- faster iteration
- no GGUF spec fight in Phase 0
- easier rollback
- clearer separation of weight format vs runtime codec

## 7. Recommended Phase Order For The Next Agent

### Phase A: Plumbing

- Add sidecar loader
- Add runtime mode enum
- Add config validation
- No math/runtime behavior change yet

### Phase B: Exact + key-only baseline

- Add `ExactKvCodec`
- Add `PaperKeyOnlyCodec`
- Wire into generation
- Keep V exact

### Phase C: Bench + parity

- Add bench mode comparing:
  - exact
  - paper key-only
- record:
  - tok/s
  - memory usage
  - context length behavior

### Phase D: Paper full-KV

- Add `PaperFullKvCodec`
- keep benchmark-only flag
- do not make default

### Phase E: Research codecs

- Add `ResearchKvSplit`
- `K = paper-like`, `V = protected or low-rank`

## 8. Acceptance Criteria Per Phase

### Phase A

- Config loads successfully
- Wrong schema fails early
- Existing exact runtime unchanged

### Phase B

- `hypura run` accepts paper key-only mode
- Exact and paper key-only both complete generation
- Bench output reports memory and speed separately per mode

### Phase C

- Dedicated benchmark command compares exact vs paper key-only
- Results are serializable to JSON/CSV

### Phase D

- full-KV mode compiles and runs
- full-KV is clearly marked experimental

### Phase E

- research mode is behind explicit flag
- paper baseline remains the default baseline path

## 9. Risks To Flag To The Next Agent

1. **Do not treat TurboQuant as weight quantization.**
   - GGUF is still the weight file
   - TurboQuant is runtime KV-cache compression

2. **Do not start with full-KV as default.**
   - The current evidence says V is the fragile path

3. **Do not mix paper and research schemas.**
   - `paper` must stay strict
   - `research` must stay sidecar-ready

4. **Do not silently degrade to exact mode.**
   - if sidecar load fails, emit explicit failure

5. **Do not port too much at once.**
   - sidecar plumbing first
   - K-only first
   - full-KV later

## 10. Suggested First Task For The New Agent

If another agent takes over immediately, the highest-value first implementation
task is:

> Add `TurboQuantSidecarConfig` loading to `hypura`, plumb a `--turboquant-config`
> CLI flag through `run` and `serve`, and introduce a runtime mode enum with
> `Exact` and `PaperKeyOnly` variants, without changing inference math yet.

This is the safest first landing because it creates the integration seam without
committing the runtime to a fragile V-path design.
