# 2026-04-19 llama.cpp TQ4_1S ggml loadable path

## Summary

Implemented the missing `ggml` runtime path for `TQ4_1S` in vendored `zapabob/llama.cpp` so `TQ4_1S` tensors are no longer reference-only metadata.

This change adds:

- `GGML_TYPE_TQ4_1S` at enum slot `36` for compatibility with existing TurboQuant GGUF artifacts
- `block_tq4_1s` row layout in `ggml-common.h`
- research-faithful `TQ4_1S` quantize/dequantize in `ggml-quants.c`
- CPU type traits and `vec_dot` path using `Q8_0` activations
- row validation and `ggml_quantize_chunk()` support
- focused tests proving byte-exact packing and CPU dot-product correctness

## Files changed

- `zapabob/llama.cpp/ggml/include/ggml.h`
- `zapabob/llama.cpp/ggml/src/ggml-common.h`
- `zapabob/llama.cpp/ggml/src/ggml.c`
- `zapabob/llama.cpp/ggml/src/ggml-quants.h`
- `zapabob/llama.cpp/ggml/src/ggml-quants.c`
- `zapabob/llama.cpp/ggml/src/ggml-cpu/quants.h`
- `zapabob/llama.cpp/ggml/src/ggml-cpu/quants.c`
- `zapabob/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`
- `zapabob/llama.cpp/ggml/src/ggml-cpu/arch-fallback.h`
- `zapabob/llama.cpp/tests/test-turboquant-runtime-reference.cpp`

## Verification

Build:

```powershell
cmake --build zapabob/llama.cpp/build-turboquant-ref --config Release --target test-turboquant-runtime-reference llama-bench -- /m:1
```

Focused tests:

```powershell
& "C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-runtime-reference.exe"
& "C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-artifact.exe"
& "C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-gguf-metadata.exe"
```

Result:

- `test-turboquant-runtime-reference.exe`: `8` tests, `25` assertions, all pass
- `test-turboquant-artifact.exe`: `3` tests, `12` assertions, all pass
- `test-turboquant-gguf-metadata.exe`: `5` tests, `23` assertions, all pass

Real GGUF loadability:

```powershell
& "C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\llama-bench.exe" `
  -m "C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\supergemma4-Q8_0.tq4_1s.gguf" `
  -ngl 0 -n 0 -p 1 -t 1 -r 1 --no-warmup -v
```

Observed evidence:

- metadata dump shows `type tq4_1s: 228 tensors`
- `load_tensors:` enumerates tensors without `failed to load model`
- command exits `0`
- `llama_perf_context_print: load time = 28321.47 ms`

## Notes

- This closes the CPU `ggml` load/dequant/matmul path for `TQ4_1S`.
- CUDA fused kernels and large-prefill scratch conversion are still separate follow-up work.
- The short `llama-bench` run confirms the former `Failed to load model` blocker is gone for the provided GGUF.
