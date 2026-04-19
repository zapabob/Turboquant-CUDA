# 2026-04-20 Triality Shared ABI And Fail-Closed Runtime

## Summary

Implemented the shared TurboQuant ABI pass needed to keep `Turboquant-CUDA`
and `zapabob/llama.cpp` aligned around the production K-side runtime mode
`key_only_block_so8_triality_vector`, learned SO(8) validation, and stricter
fail-closed metadata handling.

This pass does **not** finish the full CUDA optimization roadmap yet.
It strengthens compatibility, canonical naming, metadata validation, and the
large-batch staged CUDA routing line that prefers cuBLAS for `TQ4_1S`.

## Root repo changes

- Canonicalized Triality runtime mode and public view aliases in
  `turboquant/triality_contract.py`.
- Added shared ABI metadata keys for:
  - `codec`
  - `rotation_block_size`
  - `view_bundle_complete`
  - `orthogonality_error`
  - `determinant_error_max`
- Switched Triality proxy payloads and metadata to the production runtime mode
  `key_only_block_so8_triality_vector`.
- Tightened metadata validation so malformed or incomplete Triality payloads
  fail loudly.
- Updated `turboquant/gguf_profiles.py` so bridge metadata preserves the new
  Triality/SO(8) metrics and normalizes compatibility aliases.
- Updated `turboquant/research_extension/k_triality.py` so exported metadata
  records SO(8) orthogonality and determinant metrics.

## llama.cpp changes

- Added runtime config support for:
  - `triality_view`
  - `require_artifact`
- Canonicalized env and GGUF metadata alias handling to the production runtime
  mode.
- Tightened GGUF load validation to require:
  - `codec=tq4_1s`
  - `rotation_block_size=8`
  - `view_bundle_complete=true`
  - non-negative SO(8) metrics
  - production runtime mode
  - vector view
- Extended SO(8) validation to reject blocks whose determinant deviates from
  `1`.
- Added a staged CUDA routing preference in `ggml-cuda.cu` so `TQ4_1S`
  large-batch matrix multiplies prefer the cuBLAS-backed path.

## Important non-claim

This pass is **not** the same thing as a dedicated `TQ4_1S -> q8_0 scratch`
kernel or a fused packed-weight CUDA matmul kernel.

What is implemented now:

- shared ABI normalization
- fail-closed Triality metadata validation
- learned SO(8) determinant/orthogonality checks
- staged CUDA routing preference for larger `TQ4_1S` matmuls

What remains for the performance line:

- dedicated `TQ4_1S -> q8_0 scratch + cuBLAS`
- fused packed-weight CUDA kernel
- explicit decode/prefill routing benchmarks on real workloads

## Verification

Ran:

```powershell
uv run python -m pytest tests\test_triality_contract.py -q
uv run python -m pytest tests\test_weight_gguf.py -k "not real_gemma4_e2e" -q
```

Result:

- `test_triality_contract.py`: `10 passed`
- `test_weight_gguf.py -k "not real_gemma4_e2e"`: `5 passed, 1 deselected`

Ran:

```powershell
cmake --build build-turboquant-ref --config Release --target test-turboquant-artifact
.\build-turboquant-ref\bin\Release\test-turboquant-artifact.exe
.\build-turboquant-ref\bin\Release\test-turboquant-runtime-reference.exe
.\build-turboquant-ref\bin\Release\test-turboquant-gguf-metadata.exe
```

Result:

- `test-turboquant-artifact.exe`: `3 tests / 12 assertions / 0 failures`
- `test-turboquant-runtime-reference.exe`: `9 tests / 27 assertions / 0 failures`
- `test-turboquant-gguf-metadata.exe`: `6 tests / 25 assertions / 0 failures`

Ran CUDA rebuild and load smoke:

```powershell
cmake --build H:\llama-cuda-tq4-build --config Release --target llama-bench
H:\llama-cuda-tq4-build\bin\Release\llama-bench.exe `
  -m "C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.tq4_1s.gguf" `
  -ngl 99 -n 0 -p 1 -t 1 -r 1 --no-warmup -v
```

Result:

- CUDA build succeeded against the updated `zapabob/llama.cpp` tree.
- `llama-bench` loaded the real Gemma4 `TQ4_1S` GGUF successfully.
- Loader reported `type tq4_1s: 228 tensors`.
- Runtime reached `llama_perf_context_print` with CUDA offload active.
