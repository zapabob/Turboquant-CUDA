# 2026-04-21 Triality Views And TQ4_1S Q8_0 Scratch Kernel

## Scope

- extend the shared Triality ABI so `vector`, `spinor_plus_proxy`, and
  `spinor_minus_proxy` remain fail-closed but are no longer hard-rejected
  outside the vector view
- accept `key_only_block_so8_triality_plus`,
  `key_only_block_so8_triality_minus`, and
  `key_only_block_so8_triality_best_per_layer` as canonical runtime modes
  where the metadata bundle is complete and internally consistent
- add a dedicated CUDA `TQ4_1S -> q8_0` scratch conversion path as the next
  concrete step toward a staged large-prefill path

## Root repo changes

- `turboquant/triality_contract.py`
  - added canonical runtime modes for vector / plus / minus / best_per_layer
  - normalized public aliases onto those canonical modes
  - added runtime-mode/view consistency validation
  - kept the production default on `key_only_block_so8_triality_vector`
- `turboquant/gguf_profiles.py`
  - extended Hypura bridge / serve-command compatibility to the new Triality
    runtime modes
- `tests/test_triality_contract.py`
  - added coverage for plus / minus / best_per_layer normalization and metadata
    emission

## Vendored llama.cpp changes

- `src/llama-turboquant.cpp`
  - widened the strict loader to accept Triality `vector`, `spinor+`, and
    `spinor-` views when the runtime mode matches the selected view
  - continued to reject malformed shared ABI payloads and unsupported
    best_per_layer artifact files
- `tests/test-turboquant-artifact.cpp`
  - added artifact acceptance for the spinor-plus compatibility view
- `tests/test-turboquant-gguf-metadata.cpp`
  - added GGUF acceptance for spinor-minus plus best_per_layer shared ABI
- `ggml/src/ggml-cuda/convert.cuh`
  - declared `ggml_get_to_q8_0_cuda(...)`
- `ggml/src/ggml-cuda/convert.cu`
  - added a dedicated contiguous CUDA kernel that converts packed `TQ4_1S`
    blocks into `block_q8_0` scratch buffers
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - wired the new `TQ4_1S -> q8_0 scratch -> fp16 -> cuBLAS` path into the
    large-batch fp16 mul-mat branch for contiguous `TQ4_1S` weights

## Verification

- `uv run python -m pytest tests\test_triality_contract.py -q`
- `uv run python -m pytest tests\test_weight_gguf.py -k "not real_gemma4_e2e" -q`
- `uv run python -m pytest tests\test_weight_gguf.py -k real_gemma4_e2e -q`
- `uv run python scripts\validate_repo_contract.py`
- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-artifact.exe`
- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-gguf-metadata.exe`
- `zapabob/llama.cpp/build-turboquant-ref/bin/Release/test-turboquant-runtime-reference.exe`

## Current boundary

- the new CUDA work is a **dedicated scratch conversion kernel**, not yet a
  fully fused packed-weight `TQ4_1S` GEMM
- `ggml-cuda.dll` was rebuilt with the new conversion path, but a fresh
  `llama-bench` rebuild in the CUDA tree did not complete inside the command
  timeout window in this run
