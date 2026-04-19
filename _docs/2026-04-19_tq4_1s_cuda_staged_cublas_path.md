# TQ4_1S CUDA staged cuBLAS path

## Summary

`zapabob/llama.cpp` の `ggml-cuda` に `TQ4_1S` の staged CUDA path を追加した。

今回の目的は fused kernel ではなく、`TQ4_1S` weight tensor を CUDA 上で
`fp16/fp32` scratch へ dequant し、既存の cuBLAS GEMM 経路で実行できるようにすること。

## Files changed

- `zapabob/llama.cpp/ggml/src/ggml-cuda/convert.cu`
- `zapabob/llama.cpp/tests/CMakeLists.txt`

## CUDA implementation

`convert.cu` に以下を追加した。

- `TQ4_1S` centroid table / sign pattern の device constants
- block-level inverse RHT (`FWHT + sign restore`) helper
- `block_tq4_1s -> float[32]` dequant helper
- contiguous converter:
  - `dequantize_row_tq4_1s_cuda`
- non-contiguous converter:
  - `dequantize_tq4_1s_cuda`
- converter dispatch hookup:
  - `ggml_get_to_fp16_cuda`
  - `ggml_get_to_fp32_cuda`
  - `ggml_get_to_bf16_cuda`
  - `ggml_get_to_fp16_nc_cuda`
  - `ggml_get_to_fp32_nc_cuda`
  - `ggml_get_to_bf16_nc_cuda`

これにより、`ggml_cuda_op_mul_mat_cublas()` 側は新しい専用分岐なしで、
既存の `to_fp16` / `to_fp32` converter から `TQ4_1S` を scratch 展開して
cuBLAS へ流せる。

## Test/build adjustment

Windows shared build の `tests/CMakeLists.txt` では、TurboQuant test 群が
internal helper symbol に直接触れていたため link 失敗した。

そのため以下 3 test を `NOT WIN32 OR NOT BUILD_SHARED_LIBS` で gate した。

- `test-turboquant-artifact.cpp`
- `test-turboquant-gguf-metadata.cpp`
- `test-turboquant-runtime-reference.cpp`

これは CUDA runtime 本線の build を通すための調整であり、
TurboQuant 内部 test 自体を削除したわけではない。

## Verification

### Configure CUDA build

```powershell
cmake -S C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp `
  -B H:\llama-cuda-tq4-build `
  -G "Visual Studio 18 2026" `
  -DGGML_CUDA=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_ASM_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/ml64.exe"
```

### Build CUDA runtime

```powershell
cmake --build H:\llama-cuda-tq4-build --config Release --target ggml-cuda -- /m:1
cmake --build H:\llama-cuda-tq4-build --config Release --target llama-bench -- /m:1
```

### CUDA smoke run

```powershell
H:\llama-cuda-tq4-build\bin\Release\llama-bench.exe `
  -m C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\supergemma4-Q8_0.tq4_1s.gguf `
  -ngl 99 -n 1 -p 1 -t 1 -r 1 --no-warmup
```

Observed results:

- backend: `CUDA`
- device: `NVIDIA GeForce RTX 3060`
- model metadata reports `type tq4_1s: 228 tensors`
- layers offloaded to GPU successfully
- `pp1` and `tg1` both completed on CUDA

Saved stdout summary:

- `H:\llama-cuda-tq4-build\tq4_1s_cuda_smoke.txt`

## Current status

Completed:

- CPU `ggml` load/dequant/matmul path for `TQ4_1S`
- CUDA staged dequant path for `TQ4_1S`
- real GGUF load with CUDA offload
- 1-token CUDA smoke through `llama-bench`

Not completed yet:

- `TQ4_1S -> q8_0 scratch + cuBLAS` specialized path
- fused CUDA `TQ4_1S` matmul kernel
- Windows shared-build export strategy for internal TurboQuant test helpers

## Notes

This staged CUDA path is correctness-first.
It is expected to be slower than a native packed-weight fused kernel, but it makes
`TQ4_1S` GGUFs actually loadable and executable on CUDA in the current tree.
