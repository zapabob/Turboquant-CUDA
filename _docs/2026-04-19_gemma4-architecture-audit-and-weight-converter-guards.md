# Gemma 4 architecture audit and converter guards

Date: 2026-04-19

## Purpose

Audit the Hugging Face Gemma 4 E4B architecture against the local GGUF export so
the new offline `Q8_0 -> tq4_1s` converter does not accidentally quantize
Gemma 4 specific side paths.

## Primary sources checked

- Hugging Face Transformers Gemma 4 docs
- Hugging Face `google/gemma-4-E4B-it` model card
- Hugging Face `google/gemma-4-E4B-it` `config.json`
- Local GGUF metadata for:
  - `C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf`

## Confirmed architecture facts

- Gemma 4 E4B is multimodal and uses `Gemma4ForConditionalGeneration`.
- The text backbone has:
  - `num_hidden_layers = 42`
  - `num_attention_heads = 8`
  - `num_key_value_heads = 2`
  - `head_dim = 256`
  - `global_head_dim = 512`
  - `sliding_window = 512`
  - `num_kv_shared_layers = 18`
  - `hidden_size_per_layer_input = 256`
- Layer types alternate between sliding and full attention, with full-attention
  layers carrying wider K/V projections.
- Small Gemma 4 variants use per-layer embeddings (PLE).

## Local GGUF alignment

The inspected local GGUF matched the HF E4B text config on the fields that
matter for weight conversion:

- `gemma4.block_count = 42`
- `gemma4.embedding_length = 2560`
- `gemma4.attention.head_count = 8`
- `gemma4.attention.head_count_kv = 2`
- `gemma4.attention.key_length = 512`
- `gemma4.attention.shared_kv_layers = 18`
- `gemma4.embedding_length_per_layer_input = 256`

Observed Gemma 4 specific tensors that must stay protected during the first
converter slice:

- `per_layer_token_embd.weight`
- `per_layer_model_proj.weight`
- `per_layer_proj_norm.weight`
- `blk.*.inp_gate.weight`
- `blk.*.proj.weight`
- `blk.*.layer_output_scale.weight`

## Changes landed

- Added offline `TQ4_1S` support to vendored `gguf-py`.
- Added `turboquant.weight_gguf.convert_weight_turboquant_gguf`.
- Added Gemma 4 preserve rules so the first converter slice only rewrites the
  standard decoder attention/FFN matmuls.
- Relaxed weight-plan validation so actual emitted tensor plans can honestly
  record `q8_0` passthrough for codecs not implemented yet.

## Verification

Ran:

```powershell
uv run python -m pytest tests\test_weight_gguf.py tests\test_triality_contract.py -q
```

Result:

- `12 passed`

## Remaining gaps

- `tq4_1s` is still an offline artifact codec here, not a completed
  `llama.cpp`/CUDA runtime path.
- `general.file_type` is written as `GUESSED` because there is no dedicated
  top-level GGUF file-type enum for `tq4_1s` yet.
- `ffn_down -> q4_k` remains a metadata/runtime target, not an implemented
  offline encoder in this repo.
