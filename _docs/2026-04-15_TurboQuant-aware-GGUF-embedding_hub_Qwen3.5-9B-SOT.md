# TurboQuant-aware GGUF Embedding Implementation Log

## Overview

Implemented the first end-to-end vertical slice for a single-file
`TurboQuant-aware GGUF` packaging flow. The new packaging layer copies an
existing GGUF losslessly, embeds a paper-faithful `paper` runtime profile plus a
production `so8_triality_vector` runtime profile, and stores learned Triality
rotation tensors directly inside the output GGUF.

## Background / requirements

- User request: move from design into implementation for a single GGUF artifact
  that can carry selectable TurboQuant runtime profiles.
- Repository constraints:
  - correctness and reproducibility first
  - Windows-native / PowerShell-safe workflow
  - explicit, reviewable patches
  - fail loudly on unsupported metadata or shape mismatches
- Quantization rules:
  - keep paper-faithful and research / Triality paths distinct
  - production K-side reference is the Triality vector-view path

## Assumptions / decisions

- Scoped this slice to the packaging ABI first rather than immediately wiring
  `vendor/llama.cpp` runtime loading.
- Kept the base GGUF weights and metadata intact, then appended TurboQuant
  manifests and extra rotation tensors under a dedicated `turboquant.*`
  namespace.
- Reserved `exact` as the default no-op runtime profile rather than silently
  enabling a lossy profile.
- Embedded one concrete `bits_total` setting per packaged profile family in this
  slice. The script defaults to `3.5` bits/channel, which is the conservative
  quality-first default discussed during design.
- Rejected repackaging files that already carry `turboquant.*` metadata or
  tensors to avoid silently stacking incompatible embedded manifests.

## Changed files

- Added:
  - `turboquant/gguf_profiles.py`
  - `scripts/pack_turboquant_gguf.py`
  - `tests/test_turboquant_gguf_profiles.py`
  - `_docs/2026-04-15_TurboQuant-aware-GGUF-embedding_hub_Qwen3.5-9B-SOT.md`
- Updated:
  - `turboquant/__init__.py`

## Implementation details

- Added `turboquant.gguf_profiles` with:
  - vendored `gguf-py` loader import helper
  - GGUF metadata readers for `head_dim` and `block_count`
  - `build_paper_gguf_profile(...)`
  - `build_so8_triality_vector_gguf_profile(...)`
  - `package_turboquant_gguf(...)`
  - `read_turboquant_gguf_manifest(...)`
- The packager:
  - copies source GGUF metadata and tensor bytes without re-quantizing base
    tensors
  - writes a top-level `turboquant.schema_version`, profile list, default
    profile, and package timestamp
  - writes per-profile manifest JSON plus flat scalar / array metadata
  - embeds vector-view Triality rotations as extra float32 tensors named like
    `turboquant.profile.so8_triality_vector.layer_00.bits_3p5.rotation`
- Added a Windows-friendly CLI script:
  - `scripts\pack_turboquant_gguf.py`
  - default profiles: `paper,so8_triality_vector`
  - default runtime default profile: `exact`
  - default concrete bit setting: `3.5`
- Added pytest coverage for:
  - GGUF metadata inference
  - package + round-trip manifest reading
  - original tensor preservation
  - quantized Q8_0 tensor byte/type preservation
  - embedded rotation tensor presence
  - rejection of already-packaged GGUF inputs
- Fixed a real-file regression in quantized tensor copying:
  - source GGUF quantized tensors expose `reader.shape` as the logical tensor
    shape and `reader.data.shape` as the byte-layout shape
  - the initial packer mistakenly passed the logical shape back through the
    quantized writer path, which double-applied the byte-shape conversion for
    `Q8_0`
  - the final implementation now preserves quantized tensors by forwarding the
    raw `uint8` byte-layout shape directly and only supplies explicit logical
    shape overrides for non-`uint8` tensor payloads

## Commands run

```powershell
uv run python -m pytest tests\test_turboquant_gguf_profiles.py -q
@'
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

repo = Path(r'H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT')
sys.path.insert(0, str(repo))

from turboquant.allocation import ChannelBitAllocation
from turboquant.gguf_profiles import import_vendor_gguf
from turboquant.research_extension.k_triality import TrialityRotationArtifact, save_triality_proxy_rotations
from turboquant.schema import build_turboquant_artifact_metadata

tmp = Path(tempfile.mkdtemp(prefix='tq-gguf-cli-'))
source = tmp / 'toy.gguf'
rotation_dir = tmp / 'rotations'
rotation_dir.mkdir(parents=True, exist_ok=True)
output = tmp / 'toy.turboquant.gguf'

gguf = import_vendor_gguf()
writer = gguf.GGUFWriter(source, arch='qwen35', use_temp_file=False)
writer.add_name('toy-qwen')
writer.add_uint32('general.file_type', 7)
writer.add_uint32('qwen35.block_count', 2)
writer.add_uint32('qwen35.embedding_length', 16)
writer.add_uint32('qwen35.attention.head_count', 2)
writer.add_uint32('qwen35.attention.key_length', 8)
writer.add_tensor('token_embd.weight', np.arange(16, dtype=np.float32).reshape(4, 4))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()

metadata = build_turboquant_artifact_metadata(
    total_bits=3.5,
    qjl_bits=1,
    qjl_dim=8,
    rotation_policy='block_so8_learned',
    rotation_seed=17,
    qjl_seed=71,
    triality_mode='triality_proxy',
    triality_view='vector',
    width=8,
    allocation=ChannelBitAllocation.preset(effective_bits=2.5, width=8),
)
save_triality_proxy_rotations(
    [
        TrialityRotationArtifact(
            layer_idx=0,
            bits=3.5,
            view='vector',
            rotation=torch.eye(8, dtype=torch.float32),
            rotation_seed=17,
            qjl_seed=71,
            metadata=metadata,
        ),
        TrialityRotationArtifact(
            layer_idx=1,
            bits=3.5,
            view='vector',
            rotation=2.0 * torch.eye(8, dtype=torch.float32),
            rotation_seed=17,
            qjl_seed=71,
            metadata=metadata,
        ),
    ],
    rotation_dir,
)

cmd = [
    sys.executable,
    str(repo / 'scripts' / 'pack_turboquant_gguf.py'),
    '--input-gguf',
    str(source),
    '--output-gguf',
    str(output),
    '--rotation-dir',
    str(rotation_dir),
    '--force',
]
completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
print(completed.stdout)
assert output.exists(), output
'@ | uv run python -
uv run python scripts\pack_turboquant_gguf.py `
  --input-gguf "C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" `
  --output-gguf "C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.turboquant.gguf" `
  --rotation-dir "H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\artifacts\research_extension\triality_full_train_prod_bf16\rotations" `
  --bits 3.5 `
  --force
@'
from pathlib import Path
import sys

repo = Path(r'H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT')
sys.path.insert(0, str(repo))

from turboquant.gguf_profiles import read_turboquant_gguf_manifest

output = Path(r'C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.turboquant.gguf')
manifest = read_turboquant_gguf_manifest(output)
print(manifest.default_profile)
print(sorted(manifest.profiles.keys()))
print(manifest.profiles["so8_triality_vector"].metadata["layer_indices"])
'@ | uv run python -
```

## Test / verification results

- `uv run python -m pytest tests\test_turboquant_gguf_profiles.py -q`: passed
  - `7 passed`
- CLI smoke test for `scripts\pack_turboquant_gguf.py`: passed
  - output included:
    - `default_profile=exact`
    - `embedded_profiles=paper, so8_triality_vector`
    - `base_architecture=qwen35`
- CLI smoke test for `scripts\run_hypura_kobold_proxy.py`: passed
  - `--help` renders the Kobold-compatible proxy launcher arguments
- Real-file packaging for the user-provided Q8_0 GGUF: passed
  - output file:
    - `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.turboquant.gguf`
  - output size:
    - `9,529,605,472` bytes
  - embedded manifest check:
    - `default_profile=exact`
    - `profiles=['paper', 'so8_triality_vector']`
    - `so8_triality_vector.layer_indices=[0,1,2,3,4,5,6,7]`
  - embedded Hypura bridge check:
    - `bridge_source_profile=so8_triality_vector`
    - `bridge_mode=research-kv-split`
    - `bridge_rotation_policy=triality_vector`
    - `bridge_triality_view=vector`
    - `bridge_rotation_seed=70367`
  - launcher dry-run check:
    - `cargo run --release -p hypura -- serve ... --turboquant-mode research-kv-split`

## Kobold-compatible bridge

- The packaged GGUF now optionally embeds `hypura.turboquant.*` metadata in
  addition to the repo-owned `turboquant.*` manifest.
- `auto` bridge selection chooses the embedded `so8_triality_vector` profile
  when it is present, so unsupported Kobold-style clients can be served through
  `hypura serve` without an extra sidecar.
- Paper-faithful embedded profiles are intentionally *not* bridged through
  Hypura GGUF metadata yet because the current Hypura paper path still requires
  a parsed paper sidecar config. The packer now fails loudly if that bridge is
  requested.
- A new launcher script exists:
  - `scripts\run_hypura_kobold_proxy.py`
  - default mode `gguf-auto` reads the embedded Hypura bridge metadata from the
    GGUF and launches a Kobold-compatible API on port `5001`.

## Residual risks

- `vendor/llama.cpp` does not yet consume the embedded `turboquant.*` GGUF
  manifest. This slice fixes the container format and packaging flow first.
- The embedded runtime profiles currently package one concrete `bits_total`
  setting per run. Multi-bit in-file families are still a future extension.
- Triality embedding currently stores learned rotation tensors only; it does not
  yet embed a full runtime loader contract for per-layer KV owner/reuse logic in
  `llama.cpp`.
- The real 9B user GGUF was successfully repackaged, but the embedded Triality
  profile currently covers layers `0..7` because that is the learned production
  rotation set available in this workspace.
- Unsupported KoboldCpp itself still does not execute TurboQuant natively. The
  supported compatibility path in this repo is the Kobold-compatible `Hypura`
  proxy.

## 2026-04-15 smoke-test follow-up

Attempted the previously-missing real `cargo run` smoke test for the Kobold
compatibility path, using the packaged user GGUF:

- Release + CUDA + default generator:
  - `cargo run --release -p hypura -- serve ... --turboquant-mode research-kv-split`
  - failed during `hypura-sys` CMake configure with Windows `try_run` cleanup:
    `CheckSourceRuns.cmake` could not remove `cmTC_*.exe` because another
    process still held the file lock.
- Release + CUDA + `CMAKE_GENERATOR=Ninja` + `CARGO_TARGET_DIR` on `C:`:
  - reproduced the same `try_run` file-lock failure.
- Release + CUDA + `SOURCE_DATE_EPOCH=1` + `Ninja` + `CARGO_TARGET_DIR` on `C:`:
  - avoided the `FindSIMD.cmake` `try_run` failure and advanced into the CUDA
    build proper (`nvcc.exe` observed under the `ninja.exe` process tree),
    but did not finish within the verification window.
- Debug + CPU-only (`HYPURA_NO_CUDA=1`) + `SOURCE_DATE_EPOCH=1` + `Ninja`:
  - still did not reach server readiness within a 45-minute smoke-test window,
    and no `hypura.exe` was emitted under the dedicated target directory before
    the verification timeout.

Outcome:

- The missing smoke test was attempted multiple ways and remains **blocked by
  local Rust/CMake build throughput and Windows-specific CMake behavior**, not
  by the new GGUF packaging or Hypura bridge metadata.
- Temporary logs from these attempts were written under:
  - `H:\Qwen3.5-9B-SOT-Deployment\hub_Qwen3.5-9B-SOT\_tmp\`

## Recommended next actions

- Add `vendor/llama.cpp` metadata loading for `turboquant.default_profile`,
  profile selection, and per-layer rotation tensor lookup.
- Extend the runtime CLI to choose `--turboquant-profile exact|paper|so8_triality_vector`.
- After loader wiring is in place, run a reduced real-model validation on the
  user-provided Q8_0 GGUF and compare exact vs embedded-profile replay outputs.
