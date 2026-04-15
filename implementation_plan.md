# TurboQuant on Qwen3.5-9B for RTX 3060 / Windows / CodexApp

## Status Note

This document captures the original first-pass scope. The repository has since
grown to include research extensions such as triality-proxy SO(8) K-side
rotations and a llama.cpp consumption path, but the ordering principle still
holds: offline correctness and reproducibility come before online generation
integration.

## 1. Objective

This repository implements a research-grade TurboQuant prototype for KV-cache
compression experiments on Qwen3.5-9B under the following operating
constraints:

- Host OS: Windows 11
- Agent surface: CodexApp, native Windows agent
- Shell: PowerShell
- GPU: RTX 3060
- Primary model runtime: Hugging Face Transformers
- Weight compression: bitsandbytes-based low-bit loading where available
- Scope focus: offline KV capture and offline attention-score validation first
- Explicit non-goal for first pass: full online generation-path replacement

The first milestone is not end-to-end acceleration. The first milestone is a
mathematically faithful implementation that validates the TurboQuant
inner-product estimator against real KV tensors extracted from Qwen3.5-9B.

## 2. Design Stance

This project deliberately separates four concerns:

1. paper-faithful quantization math
2. model-specific tensor capture
3. offline validation
4. later online integration

This separation is mandatory.

## 3. Paper-Aligned Scope

### Stage 1: distortion-minimizing coordinate quantization

Implement:

- random orthogonal rotation of the input vector
- per-coordinate scalar quantization
- Lloyd-Max codebook fitting
- quantize / dequantize API
- bit-width parameterization for 2, 3, 4 bits first

### Stage 2: unbiased inner-product correction

Implement:

- residual computation after Stage 1
- 1-bit QJL-style correction path on the residual
- asymmetric inner-product estimator API
- test harness for bias and variance

## 4. Explicit First-Pass Scope

### In scope

- synthetic-vector validation
- real KV capture from Qwen3.5-9B
- offline comparison of exact attention logits vs TurboQuant-estimated logits
- metrics export
- plots and CSV summaries
- clear interfaces for later Hugging Face integration

### Out of scope for first pass

- replacing Hugging Face attention during autoregressive generation
- serving through vLLM
- full llama.cpp runtime parity and key-only CUDA optimization
- speculative decoding
- vision-path processing
- long-context production benchmarking

## 5. Environment Contract

- Python 3.12.x only for the main development path
- Windows-native PowerShell execution
- `pathlib` for filesystem handling
- no bash or GNU utility assumptions in scripts

## 6. Deliverables

The first implementation milestone is complete only when all of the following
exist:

- working TurboQuant core package
- passing unit tests
- synthetic validation metrics and plots
- Qwen3.5-9B KV capture script
- offline attention-score validation script
- markdown benchmark summary
- reproducible artifacts in `artifacts/`
