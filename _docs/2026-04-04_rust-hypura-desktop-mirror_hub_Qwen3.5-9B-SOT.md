# 実装ログ: Rust/hypura デスクトップミラー・セマバンプ・GGUF切替

- **日付 (UTC):** 2026-04-04（エージェント記録時）
- **ワークツリー:** `hub_Qwen3.5-9B-SOT`

## なにしたか（なんｊ風）

- ざっくり言うと Rust 一式を `C:\Users\downl\Desktop\hypura-main\hypura-main` に **コピー**した（`rust/` 相当 + `vendor/llama.cpp`）。デスクトップ側で `cargo build` したら **ディスク空き不足**でコケたから、**H: の hub `rust/` でビルドする**か **`CARGO_TARGET_DIR` を H: に向ける**のが安全って結論になったわｗ
- **セマンティックバージョン:** `hypura` **0.3.0→0.4.0**、`hypura-sys` **0.2.0→0.3.0**、`kobold_gguf_gui` **0.1.0→0.2.0**（`Cargo.lock` も整合済みの前提）
- **GGUF 切替:** `kobold_gguf_gui` に **設定JSON永続化**（`%APPDATA%\hypura\kobold_gguf_gui_settings.json`）と **Recent GGUF** コンボを足した。Start 成功時にパスを recent に積む

## CoT（仮説→検証）

1. **仮説:** デスクトップに独立ワークスペースを置くと、`hypura-sys` の `../../vendor/llama.cpp` だけだとパスがズレる  
   **検証:** `build.rs` に `../vendor/llama.cpp`（ワークスペース直下）と従来の `../../vendor/` の両方 + **`LLAMA_CPP_DIR` / `HYPURA_LLAMA_CPP_DIR`** を追加した
2. **仮説:** GGUF は毎回 Browse だとだるい  
   **検証:** recent リスト + 保存ボタン + 終了時保存

## ビルドコマンド（PowerShell）

```powershell
cd H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\rust
$env:HYPURA_NO_CUDA = "1"   # hypura-sys を CPU のみに（CUDA 無し環境）
cargo build -p kobold_gguf_gui --release
```

デスクトップ側でビルドする場合、空き容量が足りなければ:

```powershell
$env:CARGO_TARGET_DIR = "H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\rust\target"
cd C:\Users\downl\Desktop\hypura-main\hypura-main
cargo build -p kobold_gguf_gui --release
```

## Python ワンライナー（時刻確認用）

```text
py -3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())"
```

## 注意

- 通知音 WAV の自動再生はこの環境では未実行（手元で `marisa_owattaze.wav` を鳴らす運用でお願いします）
