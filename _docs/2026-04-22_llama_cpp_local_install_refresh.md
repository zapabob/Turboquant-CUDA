## Overview

Refreshed the local `llama.cpp` install on this PC so the current vendored
TurboQuant-aware `llama.cpp` artifacts are directly callable from the user PATH
slot at:

- `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`

Also wired OpenClaw desktop local config to the installed `llama-server.exe`
path.

## Background / Requirements

- The user asked to make the local `llama.cpp` artifacts usable on this PC via
  copy/overwrite installation.
- This PC already had `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`
  on the user PATH.
- The old install contained only legacy `llama-turboquant.exe` plus stale GGML
  DLLs, not the current vendored `llama-server.exe` / `llama-cli.exe`.

## Decisions

- Reused the existing PATH-backed install directory instead of introducing a new
  Program Files install root.
- Installed the current vendored `llama.cpp` shared-build runtime artifacts from
  `build-turboquant-ref-compat\bin\Release` first, then replaced the server-side
  runtime with the dedicated CUDA build from `H:\llama-cuda-tq4-build\bin\Release`.
- Preserved the legacy `llama-turboquant.exe` and `tqllama.cmd` already present
  in the install directory.
- Set OpenClaw desktop local env `LLAMA_CPP_SERVER_EXE` explicitly to the
  installed `llama-server.exe`.

## Changed Files

- `C:\Users\downl\Desktop\clawdbot-main3\clawdbot-main\.openclaw-desktop\.env`
- local install directory contents under
  `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`

## Implementation Details

1. Stopped a stale CUDA `MSBuild` process that was still hanging around from a
   prior `llama-server` build attempt.
2. Backed up the existing install directory contents into a timestamped backup
   folder under the install root.
3. Copied current vendored `llama.cpp` runtime artifacts into the install `bin`
   directory, overwriting older DLLs as needed.
4. Completed the dedicated CUDA `llama-server` target build in
   `H:\llama-cuda-tq4-build` and verified `--list-devices` against the RTX 3060.
5. Backed up the refreshed local install one more time, then overwrote the
   installed server-side binaries with the CUDA build outputs:
   `llama-server.exe`, `llama.dll`, `llama-common.dll`, `ggml.dll`,
   `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll`, `mtmd.dll`,
   and `llama-bench.exe`.
6. Added `LLAMA_CPP_SERVER_EXE` to OpenClaw desktop local env so repo-local
   launch helpers resolve the installed server path explicitly.
7. Set user-level `LLAMA_CPP_SERVER_EXE` to the same installed path for
   machine-wide reuse.
8. Moved the install directory to the front of the persisted user PATH.

## Commands Run

```powershell
Get-Process -Id 27816 -ErrorAction SilentlyContinue | Stop-Process -Force
```

```powershell
$installRoot = 'C:\Users\downl\AppData\Local\Programs\llama-turboquant'
$installBin = Join-Path $installRoot 'bin'
$backupDir = Join-Path $installRoot ('backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item (Join-Path $installBin '*') $backupDir -Force -Recurse
```

```powershell
$sourceBin = 'C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp\build-turboquant-ref-compat\bin\Release'
$patterns = @('llama*.exe', 'llama*.dll', 'ggml*.dll', 'mtmd.dll')
foreach ($pattern in $patterns) {
  Get-ChildItem $sourceBin -Filter $pattern | Copy-Item -Destination $installBin -Force
}
```

```powershell
where.exe llama-server
& 'C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe' --help
& 'C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-cli.exe' --help
[Environment]::SetEnvironmentVariable(
  'LLAMA_CPP_SERVER_EXE',
  'C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe',
  'User'
)
```

```powershell
cmake --build 'H:\llama-cuda-tq4-build' --config Release --target llama-server -- /m:1 /v:minimal
& 'H:\llama-cuda-tq4-build\bin\Release\llama-server.exe' --list-devices
& 'C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe' --list-devices
```

## Verification Results

- The dedicated CUDA `llama-server` build completed successfully from
  `H:\llama-cuda-tq4-build`.
- Installed `llama-server.exe --help` succeeds.
- Installed `llama-cli.exe --help` succeeds.
- Installed `llama-server.exe --list-devices` reports:
  `CUDA0: NVIDIA GeForce RTX 3060`.
- Installed `llama-server.exe --version` reports:
  `version: 8876 (a18d78a9d)`.
- OpenClaw desktop local env now contains:
  `LLAMA_CPP_SERVER_EXE=C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe`
- User-level `LLAMA_CPP_SERVER_EXE` now points to the same installed path.
- Backup created at:
  `C:\Users\downl\AppData\Local\Programs\llama-turboquant\backup-20260422-164816`
- Second CUDA overwrite backup created at:
  `C:\Users\downl\AppData\Local\Programs\llama-turboquant\backup-cuda-20260422-172105`

## Residual Risks

- The long-lived current Codex process still resolves the winget-provided
  `llama-server.exe` first via `where.exe`; new shells and tools that read the
  persisted user environment should prefer the refreshed install path, and
  OpenClaw is pinned explicitly through `LLAMA_CPP_SERVER_EXE`.
- `llama-cli.exe` in the install directory is still the earlier shared-build
  binary. The server-side CUDA runtime is the requested primary deliverable and
  is now installed and verified.

## Recommended Next Actions

- If you want the CLI side rebuilt on the same CUDA tree too, build
  `llama-cli` from `H:\llama-cuda-tq4-build` and overwrite the install `bin`
  again.
- If you want OpenClaw launchers refreshed too, run a quick
  `pnpm llama-cpp:launch` smoke from `clawdbot-main`.
