## Summary

Implemented the two-layer compatibility bridge that preserves this repo's
canonical Triality learned SO(8) ABI while accepting TheTom-style public
TurboQuant UX surfaces.

The canonical internal line remains:

- weight codec: `tq4_1s`
- K-side production mode: `key_only_block_so8_triality_vector`
- Triality public views: `vector`, `spinor_plus_proxy`, `spinor_minus_proxy`

The public compatibility layer now accepts:

- K cache/public aliases: `triality-vector`, `triality-plus`,
  `triality-minus`, `best_per_layer`, `q8_0`
- V cache/public aliases: `q8_0`, `turbo4`, `turbo3`, `turbo2`

## Code Changes

### Python contract and bridge

Files:

- `turboquant/triality_contract.py`
- `turboquant/gguf_profiles.py`
- `tests/test_triality_contract.py`
- `tests/test_turboquant_gguf_profiles.py`

Changes:

- added explicit public cache-type normalization for K and V surfaces
- embedded `hypura.turboquant.cache_type_k` and
  `hypura.turboquant.cache_type_v` into shared metadata
- validated cache-type/runtime-mode consistency fail-closed
- preserved canonical runtime modes while accepting public aliases
- kept `vector` as the default public Triality view while preserving
  `plus`/`minus`
- added explicit coverage for `turbo2` and `turbo3` V-side aliases

### Vendored llama.cpp shared ABI

Files:

- `zapabob/llama.cpp/src/llama-turboquant.h`
- `zapabob/llama.cpp/src/llama-turboquant.cpp`
- `zapabob/llama.cpp/tests/test-turboquant-gguf-metadata.cpp`

Changes:

- runtime config now tracks public `cache_type_k` and `cache_type_v`
- runtime allow checks derive effective cache types from `mode` when cache
  types are left unspecified
- env/metadata parsing normalizes Triality and TheTom-style public aliases
- GGUF shared ABI validation now requires and validates cache-type metadata
- shared ABI preserves `spinor_plus_proxy` and `spinor_minus_proxy`
- `best_per_layer` remains gated by complete shared ABI metadata

## Verification

### Python

Command:

```powershell
uv run python -m pytest tests\test_triality_contract.py tests\test_turboquant_gguf_profiles.py -q
```

Result:

- `21 passed`

### Vendored llama.cpp

Static compatibility build was required because Windows shared builds gate the
TurboQuant internal tests.

Configure:

```powershell
cmake -S zapabob/llama.cpp `
  -B zapabob/llama.cpp/build-turboquant-ref-static-compat2 `
  -G "Visual Studio 18 2026" `
  -DBUILD_TESTING=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DGGML_CUDA=OFF `
  -DBUILD_SHARED_LIBS=OFF `
  -DHAS_AVX_1=TRUE `
  -DHAS_AVX2_1=TRUE `
  -DHAS_FMA_1=TRUE `
  -DHAS_AVX512_1=FALSE `
  -DHAS_AVX512_2=FALSE
```

Build:

```powershell
cmake --build zapabob/llama.cpp/build-turboquant-ref-static-compat2 `
  --config Release `
  --target test-turboquant-artifact test-turboquant-gguf-metadata test-turboquant-runtime-reference
```

Results:

- `test-turboquant-artifact.exe`: PASS, 4 tests / 15 assertions
- `test-turboquant-gguf-metadata.exe`: PASS, 7 tests / 32 assertions
- `test-turboquant-runtime-reference.exe`: PASS, 9 tests / 30 assertions

## Notes

- The Windows shared-build gate in `tests/CMakeLists.txt` remains intact.
- On Windows, `test-turboquant-artifact.exe` was reliable only when run
  outside a parallel launcher because the executable could remain file-locked
  after timeout/abort.
- This slice implemented the alias/metadata/runtime bridge, not a new codec.
- The canonical production K-side line is still
  `key_only_block_so8_triality_vector`.
