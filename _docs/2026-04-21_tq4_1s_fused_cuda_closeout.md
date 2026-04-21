# 2026-04-21 TQ4_1S Fused CUDA Closeout

## Scope

- close the `TQ4_1S` CUDA line from "loadable and staged" to
  "loadable, parity-checked, and smoke-verified"
- keep the shared Triality ABI fail-closed while preserving public
  `vector`, `spinor_plus_proxy`, and `spinor_minus_proxy` views
- verify the real Gemma 4 `TQ4_1S` GGUF on this PC after the CUDA path lands

## Landed runtime shape

- **Large prefill / large batch**
  - contiguous `TQ4_1S` weights route through
    `TQ4_1S -> q8_0 scratch -> fp16 -> cuBLAS`
- **Small decode**
  - contiguous `TQ4_1S` weights can use the fused packed-weight CUDA MMVQ path
- **Fallback behavior**
  - unsupported or non-contiguous layouts continue to use the existing
    dequantized fallback line rather than changing correctness behavior

## Root repo changes in play

- `turboquant/triality_contract.py`
  - canonical Triality mode / view normalization
  - fail-closed shared ABI checks for bundle completeness, orthogonality, and
    determinant metadata
- `turboquant/gguf_profiles.py`
  - bridge compatibility for vector / plus / minus / best_per_layer views
- `tests/test_triality_contract.py`
  - coverage for canonicalization and shared ABI metadata emission

## Vendored llama.cpp changes in play

- `ggml/src/ggml-common.h`
  - `TQ4_1S` row traits for CUDA-side type plumbing
- `ggml/src/ggml-cuda/common.cuh`
  - `ggml_cuda_type_traits<GGML_TYPE_TQ4_1S>`
- `ggml/src/ggml-cuda/convert.cuh`
  - CUDA conversion entry point for `TQ4_1S -> q8_0`
- `ggml/src/ggml-cuda/convert.cu`
  - contiguous CUDA scratch conversion kernel for packed `TQ4_1S`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - routing of large-batch contiguous `TQ4_1S` through staged cuBLAS
- `ggml/src/ggml-cuda/mmvq.cu`
  - small-batch fused MMVQ dispatch for `GGML_TYPE_TQ4_1S`
- `ggml/src/ggml-cuda/vecdotq.cuh`
  - fused packed-weight `TQ4_1S x q8_1` dot math
- `tests/test-turboquant-cuda-kernels.cu`
  - focused CUDA parity test for:
    - CPU reference
    - CUDA dequantized weights
    - fused packed-weight CUDA dot
    - `q8_0` scratch path

## Verification

### Root repo

- `uv run python -m pytest tests\test_triality_contract.py -q`
  - `12 passed`
- `uv run python -m pytest tests\test_weight_gguf.py -q`
  - `6 passed`
- `uv run python scripts\validate_repo_contract.py`
  - `Repository contract OK.`

### Vendored llama.cpp reference / metadata

- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-artifact.exe`
  - `4 tests, 15 assertions, PASS`
- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-gguf-metadata.exe`
  - `7 tests, 28 assertions, PASS`
- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-runtime-reference.exe`
  - `9 tests, 27 assertions, PASS`

### Static CUDA parity

- `H:\llama-cuda-tq4-static\bin\test-turboquant-cuda-kernels.exe`
  - `cpu_reference=22.45673943`
  - `cuda_dequant_dot=22.45673943`
  - `fused_dot=22.45673752`
  - `scratch_dot=22.45544434`
  - `deq_err=0.00000000`
  - `fused_err=0.00000191`
  - `scratch_err=0.00129509`
  - `3 assertions, PASS`

### Real-model CUDA smoke on this PC

Model:

- `C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.tq4_1s.gguf`

Commands:

- `zapabob/llama.cpp/build-tq4-cuda/bin/Release/llama-bench.exe -m ... -ngl 99 -p 1 -n 0 -t 1 -r 1 --no-warmup -v`
  - `pp1 = 0.07 +/- 0.00 t/s`
  - `EXIT=0`
- `zapabob/llama.cpp/build-tq4-cuda/bin/Release/llama-bench.exe -m ... -ngl 99 -p 1 -n 1 -t 1 -r 1 --no-warmup -v`
  - `tg1 = 0.07 +/- 0.00 t/s`
  - `EXIT=0`

Observed runtime facts:

- real GGUF load succeeded
- `type tq4_1s: 228 tensors`
- CUDA offload succeeded with `ngl 99`
- both `pp1` and `tg1` completed

## Final boundary

- the repo now has a **real fused packed-weight CUDA path** for small decode
  plus a **dedicated `q8_0` scratch path** for large-batch staged execution
- this is still **not** presented as the final tuned performance architecture
- the routing threshold remains policy, not a published universal optimum
- the repo claim is correctness, compatibility, and end-to-end survivability of
  TurboQuant artifacts first; broad speed leadership remains a later question
