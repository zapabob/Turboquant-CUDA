# 2026-03-27 Exact vs TurboQuant Chat Compare Codex

## Objective

Move beyond the raw runtime smoke and compare:

- exact cache generation
- TurboQuant KV cache generation

on the same chat-template prompt with the same generation settings.

## Implementation

Added:

- `H:\Qwen3.5-9B-SOT-Deployment\qwen35_rtx3060\scripts\compare_exact_vs_turboquant_chat.py`

Updated:

- `H:\Qwen3.5-9B-SOT-Deployment\qwen35_rtx3060\README.md`

## Script behavior

The comparison script:

1. loads the local official `Qwen3.5-9B` folder once in 4-bit mode
2. formats the input with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
3. runs exact cache generation
4. runs TurboQuant cache generation
5. compares generated token ids
6. writes a JSON artifact with:
   - prompts
   - exact text
   - TurboQuant text
   - timing
   - prefix token match length
   - first mismatch

## Expected output artifact

`H:\Qwen3.5-9B-SOT-Deployment\qwen35_rtx3060\data\runtime\exact_vs_turboquant_chat.json`

## Notes

- This is the right next step after the basic runtime smoke because it tells us whether the TurboQuant path is merely executable or also behaviorally close to the exact cache path.
- The comparison stays small and deterministic so it remains practical on the RTX 3060.

## Observed result

Run:

```powershell
& .\.venv\Scripts\python.exe scripts\compare_exact_vs_turboquant_chat.py --model H:\Qwen3.5-9B-official-hf --bits 4 --max-new-tokens 8
```

Observed summary:

- exact elapsed: `5.162s`
- TurboQuant elapsed: `3.308s`
- matching prefix tokens: `0`
- exact tokens: `[90700, 8340, 25, 271, 16, 13, 220, 2972]`
- TurboQuant tokens: `[1206, 220, 13, 220, 220, 220, 220, 220]`
- exact escaped text: `Thinking Process:\n\n1.  **`
- TurboQuant escaped text: `To .     `

Interpretation:

- The runtime path is operational.
- For this short chat-template prompt, the TurboQuant path diverges from exact generation at the very first generated token.
- This means the implementation is now ready for true quality evaluation, but is not yet behaviorally close enough to exact generation on this smoke case.

## Decode-step attribution result

Added:

- `H:\Qwen3.5-9B-SOT-Deployment\qwen35_rtx3060\scripts\compare_decode_step_diffs.py`

Run:

```powershell
& .\.venv\Scripts\python.exe scripts\compare_decode_step_diffs.py --model H:\Qwen3.5-9B-official-hf --bits 4
```

Output:

- `H:\Qwen3.5-9B-SOT-Deployment\qwen35_rtx3060\data\runtime\decode_step_diffs.json`

Observed summary:

- `decode_token_id_used = 90700`
- key-only decode argmax match: `1`
- full-KV decode argmax match: `0`
- key-only decode cosine: `0.830677`
- full-KV decode cosine: `0.627187`
- key-only prefill argmax match: `1`
- full-KV prefill argmax match: `0`

Layer drift summary:

- key-only prefill hidden-state cosine drops below `0.99` at layer `8`
- full-KV prefill hidden-state cosine drops below `0.99` at layer `4`
- key-only decode hidden-state cosine drops below `0.95` at layer `12`
- full-KV decode hidden-state cosine drops below `0.95` at layer `4`
- last decode hidden-state cosine:
  - key-only: `0.633874`
  - full-KV: `0.065355`

Interpretation:

- The key-only path is materially closer to exact than the full-KV path.
- The main runtime quality loss is now most plausibly coming from value quantization rather than key quantization.
- The next best implementation target is to keep runtime evaluation focused on key-only first, then improve or redesign the value path separately.
