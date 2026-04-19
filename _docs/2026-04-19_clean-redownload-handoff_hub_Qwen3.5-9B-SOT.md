# 2026-04-19 clean redownload handoff

## 目的

次回は `zapabob/Turboquant-CUDA` から repo を正しく再ダウンロードして作業を再開する前提で、
これまでの主要 thread の成果・未完事項・読む順番をまとめる。

対象 thread:

- `019d9065-ab0a-7143-b897-1045157e3ea2`
- `019d90c4-dce1-7ef1-b12e-32e5f3e577f3`
- `019d90fb-bc31-7d41-9258-ffcf89186ef3`
- `019d919c-031e-7fd2-a830-1a20f17cd710`
- `019d91da-708a-7b51-880e-6083986c23ab`
- `019d926a-3654-73c1-90e0-1e00e249227d`
- `019d92c4-f4d7-7891-a33a-383157953f29`
- `019d9c47-5646-7a81-a02f-1d76672f7b22`
- `019da01c-1a2f-78e2-97a4-b75d9631a4c7`

## 今回 main に載せた現行前提

- `vendor/llama.cpp` の canonical path は `zapabob/llama.cpp` に変更した。
- submodule は `zapabob/llama.cpp` の `9276d42f4` を指す。
- `repo_contract.toml`、Rust build resolver、Studio 既定値、関連テストは
  新パスに更新済み。
- `vendor/...` は一部 resolver に fallback としてだけ残してある。
- CUDA 12.8 用の `torch` は repo 直下 `.venv` ではなく、
  `C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv`
  で動作確認済み。

関連ログ:

- `_docs/2026-04-19_vendor-llama-cpp_zapabob-sync_02.md`
- `_docs/2026-04-19_zapabob-llama-cpp-path-rename_hub_Qwen3.5-9B-SOT.md`
- `_docs/2026-04-19_uv-cu128-torch-env_hub_Qwen3.5-9B-SOT.md`

## thread 別の要点

### 1. `019d9065-ab0a-7143-b897-1045157e3ea2`

build contract を固めた起点。

- `repo_contract.toml`
- `turboquant/repo_contract.py`
- `scripts/validate_repo_contract.py`
- `scripts/build_rust_workspace.ps1`

ここで確立した前提:

- source-of-truth は `zapabob/Turboquant-CUDA` と `zapabob/llama.cpp`
- Windows では disk pressure を前提に `C:` target を使う
- Rust / CMake は `NUM_JOBS=1`、`GGML_CCACHE=OFF` が安定

未完:

- GGUF / BF16 / Q8.0 から TurboQuant に落とす実用圧縮経路はまだ設計段階

### 2. `019d90c4-dce1-7ef1-b12e-32e5f3e577f3`

`zapabob/llama.cpp` から vendored runtime を selective sync した thread。

- TurboQuant / quantize / GGUF metadata / Gemma4 dependency closure を優先
- Windows 上で build と `llama-turboquant` smoke を確認

重要:

- upstream の取り込みは単純 fast-forward ではなく、TurboQuant 系と依存閉包を
 選別して広げた
- `test-quantize-fns` と `llama-turboquant` smoke はこの系統の有効な確認手段

### 3. `019d90fb-bc31-7d41-9258-ffcf89186ef3`

GGUF に Hypura/Kobold bridge metadata を埋め込む流れを作った thread。

主な成果:

- `pack_turboquant_gguf.py`
- `turboquant.gguf_profiles`
- `--hypura-compatible-profile`
- `hypura.turboquant.*` bridge metadata
- `run_hypura_kobold_proxy.py`

状態:

- packaging と manifest readback は成功
- 最終 server smoke は Windows CMake / build 挙動に阻まれて partial

### 4. `019d919c-031e-7fd2-a830-1a20f17cd710`

README に integration-project note を入れた docs-only thread。

意図:

- この repo は単独完結 repo ではなく、
  `zapabob/Turboquant-CUDA`、
  `zapabob/llama.cpp`、
  `zapabob/Hypura`
  の統合プロジェクトであることを明示

### 5. `019d91da-708a-7b51-880e-6083986c23ab`

RTX 3060 12GB lane を Qwen-first に寄せて整理し始めた thread。

成果:

- 3060 lane presets
- capture metadata / contract coverage
- 12GB-only Qwen matrix の枠組み
- vendored runtime gating の partial 実装

未完:

- 12GB matrix の end-to-end closeout はこの thread 単独では未完

### 6. `019d926a-3654-73c1-90e0-1e00e249227d`

Hypura 0.7.0 upstream sync を別 branch で完了して push した thread。

ポイント:

- `hypura 0.7.0`
- `hypura-sys 0.7.0`
- `RotationPolicy::TrialityVector`
- branch push は完了

注意:

- これは main 直載せではなく scoped branch 作業の closeout 色が強い
- clean clone 後に「main に何が入っていて何が branch 止まりか」を再確認すること

### 7. `019d92c4-f4d7-7891-a33a-383157953f29`

runtime audit と HF/runtime online eval を整えた大きめの closeout。

主な成果:

- `turboquant/runtime_eval.py`
- `turboquant/eval_stats.py`
- `turboquant/adapters/hf_qwen/online_eval.py`
- `scripts/eval_runtime_qwen.py`
- `scripts/eval_hf_online_qwen.py`
- `scripts/export_online_eval_report.py`
- `scripts/audit_zapabob_runtime.py`

この thread の結論:

- HF online eval は診断用であって、runtime 実証そのものではない
- current-main `zapabob/llama.cpp` runtime 側で claim できることと、
  replay/HF だけで見えていることは README 上でも明確に分けるべき
- stock `/v1/completions` では prompt `token_logprobs` が取れず、
  MCQ loglikelihood claim は runtime evidence としては未成立

### 8. `019d9c47-5646-7a81-a02f-1d76672f7b22`

README を短くしたあと、統計・図表セクションを戻した thread。

重要:

- README は短くするだけではだめで、研究 evidence への直接リンクを落としてはいけない
- 具体的には以下を top-level で見えるように保つ:
  - Eval Output Layout
  - Pareto Frontiers
  - Paper Baseline Reference Results
  - Triality Advantage Figures

### 9. `019da01c-1a2f-78e2-97a4-b75d9631a4c7`

別 thread の変更を forward-port しようとしたが、current main に新しい形で
既に吸収済みだったことを確認した thread。

結論:

- `codex://threads/019d928b-7425-7520-b89c-a62fc02aaf5c`
  の変更は no-op forward-port
- focused pytest で裏取り済み

## 次に読む順番

clean clone 後はこの順が安全:

1. `README.md`
2. `CLAUDE.md`
3. `repo_contract.toml`
4. `_docs/2026-04-19_zapabob-llama-cpp-path-rename_hub_Qwen3.5-9B-SOT.md`
5. `_docs/2026-04-19_uv-cu128-torch-env_hub_Qwen3.5-9B-SOT.md`
6. `_docs/2026-04-19_vendor-llama-cpp_zapabob-sync_02.md`
7. `_docs/2026-04-17_current-main-zapabob-runtime-audit_hub_Qwen3.5-9B-SOT.md`
8. `_docs/2026-04-18_README-runtime-audit-cleanup_hub_Qwen3.5-9B-SOT.md`

その後、必要に応じて memory 側 rollout summary を thread ごとに参照する。

## clean clone 後の最初の確認コマンド

```powershell
git submodule update --init --recursive
python -c "from pathlib import Path; print(Path('zapabob/llama.cpp').exists())"
python scripts\validate_repo_contract.py
```

PyTorch/CUDA を使うなら:

```powershell
& C:\Users\downl\AppData\Local\Temp\hub-qwen-cu128-venv\Scripts\Activate.ps1
uv sync --active --extra cu128 --extra dev --link-mode copy
```

## 未解決 / 次回の優先事項

1. `zapabob/Turboquant-CUDA` clean clone 上で、今回 push した
   `zapabob/llama.cpp` canonical path 変更が期待通り見えるか再確認する
2. repo-local `.venv` ではなく external `C:` env を使っている暫定状態を、
   clean clone 後にどう扱うか決める
3. GGUF / BF16 / Q8.0 から TurboQuant へ落とす実用 compression/export path を、
   `019d9065` の設計メモから再開する
4. Hypura/Kobold の final server smoke を Windows 上で通す
5. README の evidence-first 方針を維持したまま、mainline doc を今後も崩さない

## 補足

- 歴史的 `_docs` や古い rollout summary に `vendor/llama.cpp` という表記が残るのは
  正常。これは当時の実 path を記録しているだけ。
- 現在の operational canonical path は `zapabob/llama.cpp`。
