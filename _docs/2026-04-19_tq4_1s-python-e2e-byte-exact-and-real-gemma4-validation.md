# 2026-04-19 TQ4_1S Python E2E Byte-Exact And Real Gemma4 Validation

## Overview

Aligned the vendored Python `TQ4_1S` GGUF codec with the research-faithful `zapabob/llama.cpp` reference math, added byte-exact regression coverage, validated converter behavior on toy GGUF fixtures and a real Gemma4 E4B Q8_0 GGUF, and locked the C++ reference runtime with a packed-byte golden.

## Background / Requirements

- The Python-side converter had to emit `TQ4_1S` bytes that match the `llama.cpp` reference runtime exactly.
- The end-to-end path had to cover a real model:
  `C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf`
- Gemma4 preserve rules had to remain in place.
- `ffn_down -> q4_k` remains deferred; this phase keeps `ffn_down` as `q8_0`.
- KV-side production default remains `key_only_block_so8_triality_vector`.

## Assumptions / Decisions

- Byte-exact means the Python packed bytes match the C++ reference codec semantics for the same logical rows.
- The Python `TQ4_1S` implementation follows:
  - 32-wide sign-flip plus normalized FWHT/RHT
  - 16 Lloyd-Max centroids
  - midpoint-based index selection
  - dual fp16 half-block scales
  - 9-point scale search
  - 6 refinement iterations
- Real-file validation reuses an existing converted artifact when a valid cached output is present on a drive with sufficient free space.
- Because `C:` had effectively no free space during validation, large-artifact acceptance used `H:` scratch storage.

## Changed Files

- `C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\gguf-py\gguf\quants.py`
- `C:\Users\downl\Desktop\Turboquant-CUDA\tests\test_weight_gguf.py`
- `C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\tests\test-turboquant-runtime-reference.cpp`

## Implementation Details

- Replaced the placeholder Python `TQ4_1S` quantizer/dequantizer with the research-faithful block math used by the C++ reference runtime.
- Corrected the packed block layout to `d0(fp16) + d1(fp16) + 16 packed nibble bytes`.
- Added Python reference helpers inside tests to make byte-level expectations explicit and auditable.
- Added converter tests that verify:
  - selected Q8_0 tensors become `TQ4_1S`
  - Gemma4 PLE / projector / `inp_gate` preserve paths remain Q8_0
  - emitted bytes equal the reference codec output
  - weight payload JSON matches the actual emitted weight plan
  - invalid source tensor classes fail loudly
- Added real Gemma4 E2E acceptance that checks:
  - converted `blk.2.attn_q.weight` is `TQ4_1S`
  - `blk.2.ffn_down.weight` remains `Q8_0`
  - `per_layer_token_embd.weight` and `blk.2.inp_gate.weight` remain byte-identical to source
  - metadata namespaces exist
- Added a C++ golden-byte test so future drift in `llama.cpp` reference packing fails immediately.

## Commands Run

```powershell
uv run python -m pytest tests\test_weight_gguf.py -q
uv run python -m pytest tests\test_weight_gguf.py -k "not real_gemma4" -q
uv run python -m pytest tests\test_weight_gguf.py -k real_gemma4 -q
uv run python -m pytest tests\test_weight_gguf.py tests\test_triality_contract.py -q
cmake --build C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref --config Release --target test-turboquant-runtime-reference -- /m:2
C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-runtime-reference.exe
C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-artifact.exe
C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-gguf-metadata.exe
```

## Test / Verification Results

- `tests/test_weight_gguf.py tests/test_triality_contract.py`: `15 passed`
- `test-turboquant-runtime-reference.exe`: `6 tests`, `18 assertions`, all pass
- `test-turboquant-artifact.exe`: `3 tests`, `12 assertions`, all pass
- `test-turboquant-gguf-metadata.exe`: `5 tests`, `23 assertions`, all pass
- Real Gemma4 acceptance passed using a valid cached converted artifact on `H:\`.

## Residual Risks

- The real-model acceptance cache is artifact-based reuse. If the cached file is deleted, regenerating it still requires large free disk capacity and substantial runtime.
- This phase does not implement native CUDA `TQ4_1S` matmul or `TQ4_1S -> q8_0 scratch + cuBLAS`.
- `ffn_down -> q4_k` remains deferred and is intentionally still `q8_0`.
- The Python codec and C++ reference runtime are aligned, but full native GGML type integration for inference remains a separate track.

## Recommended Next Actions

- Add a reusable validation script that can regenerate or verify a cached real-model artifact outside pytest.
- Implement the large-prefill scratch path (`TQ4_1S -> q8_0 + cuBLAS`) next.
- Only after scratch-path verification, move to small-batch fused CUDA kernels.
