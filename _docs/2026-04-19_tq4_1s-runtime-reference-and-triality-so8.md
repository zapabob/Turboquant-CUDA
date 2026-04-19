# 2026-04-19 TQ4_1S Runtime Reference and Triality/SO8

## Summary

- Added a research-faithful `TQ4_1S` reference codec and native packed-weight matvec path to
  `zapabob/llama.cpp/src/llama-turboquant.cpp`.
- Replaced the placeholder Triality centroid selection with vector-level k-means style training.
- Added strict SO(8) validation for learned rotation artifacts.
- Aligned K-side production mode handling with `key_only_block_so8_triality_vector`.

## Why

The previous line had metadata and artifact plumbing for TurboQuant, but no faithful
reference implementation for:

- `TQ4_1S` packed-weight math
- Triality codebook learning beyond a placeholder copy-slice
- learned SO(8) artifact validation

That gap made it too easy to confuse "contract exists" with "runtime math exists".

## Research Notes

Primary references used:

- Google Research blog on TurboQuant compression principles
- TurboQuant ICLR 2026 paper line for WHT + Lloyd-Max quantization
- `TheTom/turboquant_plus` weight-compression writeup for `TQ4_1S` policy and centroids
- `TheTom/llama-cpp-turboquant` runtime code for:
  - `block_tq4_1s`
  - RHT forward/inverse
  - 16-level Lloyd-Max centroids
  - native packed-weight matvec strategy
  - split between small-batch native path and large-prefill scratch conversion

Key conclusion:

- Google's TurboQuant material is the right anchor for WHT + centroid quantization.
- `Triality` and `learned SO8` are not directly named as canonical Google paper terms here;
  they are better treated as a Zapabob-line runtime extension informed by learned rotation work.

## Implementation

Files changed:

- `zapabob/llama.cpp/src/llama-turboquant.h`
- `zapabob/llama.cpp/src/llama-turboquant.cpp`
- `zapabob/llama.cpp/tests/test-turboquant-runtime-reference.cpp`
- `zapabob/llama.cpp/tests/CMakeLists.txt`

Added runtime reference APIs:

- `llama_turboquant_validate_so8_rotation(...)`
- `llama_turboquant_quantize_tq4_1s_reference(...)`
- `llama_turboquant_dequantize_tq4_1s_reference(...)`
- `llama_turboquant_mul_mat_tq4_1s_reference(...)`

Implementation details:

- `TQ4_1S` uses:
  - 32-element blocks
  - sign-flip + normalized FWHT forward rotation
  - 16 Lloyd-Max centroids
  - dual half-block fp16 scales
  - nibble-packed indices
- Native reference matvec keeps weights packed and pre-rotates the activation once per 32-wide block.
- Triality training now performs repeated vector-level reassignment and centroid updates.
- Artifact load now rejects malformed learned SO(8) rotations and malformed Triality codebooks.

## Verification

Built with:

```powershell
cmake -S C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp `
  -B C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref `
  -DLLAMA_BUILD_TESTS=ON -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF `
  -DCMAKE_ASM_COMPILER="C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\ml64.exe"
```

Executed:

```powershell
cmake --build C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref `
  --config Release --target test-turboquant-runtime-reference -- /m:2

& C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-runtime-reference.exe

cmake --build C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref `
  --config Release --target test-turboquant-artifact test-turboquant-gguf-metadata -- /m:2

& C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-artifact.exe
& C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref\bin\Release\test-turboquant-gguf-metadata.exe
```

Results:

- `test-turboquant-runtime-reference.exe`: PASS
- `test-turboquant-artifact.exe`: PASS
- `test-turboquant-gguf-metadata.exe`: PASS

## Remaining Gap

This completes the **reference runtime math**.

It does **not** yet complete:

- fused CUDA `TQ4_1S` kernels in this repo's current submodule line
- large-prefill scratch-conversion path
- converter/runtime unification for the Python-side GGUF exporter

So the current state is:

- reference math: present
- artifact validation: present
- native optimized GPU path: still pending
