# 実装ログ — 2026-04-04 — Multiscreen KV + Kobold 互換 GUI

- **Turboquant-CUDA 同期**: `turboquant/research_extension/multiscreen_kv.py` 新規、`ChannelBitAllocation.from_multiscreen_relevance` / `make_bitwidths_from_relevance` 追加、`tests/test_multiscreen_kv.py` 追加（`uv run python -m pytest` 全通過）。
- **vendor**: `git submodule` で `vendor/llama.cpp` → `https://github.com/zapabob/llama.cpp.git`。
- **Rust GUI**: `rust/kobold_gguf_gui` — `llama-server` 子プロセス + Axum で `POST /api/v1/generate` → バックエンド `POST /v1/completions` 変換。ビルド: `cd rust` して `cargo build -p kobold_gguf_gui`。

※ 初回は `git submodule update --init --recursive`。llama.cpp は CMake で `llama-server` をビルドして GUI の exe 欄に指定すること。
