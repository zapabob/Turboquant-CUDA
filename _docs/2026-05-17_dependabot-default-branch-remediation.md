# 2026-05-17 Dependabot Default-Branch Remediation

## Summary

Remediated the open Dependabot alerts reported against the default branch
(`main`) by updating locked package versions and removing an optional evaluation
dependency that currently has no patched transitive release.

## Alerts

| Ecosystem | Manifest | Package | Advisory | Remediation |
| --- | --- | --- | --- | --- |
| pip | `uv.lock` | `urllib3` | `GHSA-qccp-gfcp-xxvc`, `GHSA-mf9v-mfxr-j63j` | Add a top-level `urllib3>=2.7.0` constraint and regenerate `uv.lock`. |
| pip | `uv.lock` | `sqlitedict` | `GHSA-g4r7-86gm-pgqc` | Remove `lm-eval[api]` from the `eval` extra because the alert has no patched `sqlitedict` release. |
| cargo | `rust/Cargo.lock` | `rand` | `GHSA-cq8v-f236-94qc` | Update the locked `rand 0.9.x` entry from `0.9.2` to `0.9.3`. |

## Notes

- Runtime evaluation command construction remains in `turboquant.runtime_eval`.
  The project no longer installs or locks `lm-eval[api]` from the `eval` extra
  until upstream removes or patches the vulnerable `sqlitedict` dependency.
- The `uv.lock` refresh used the installed WinGet `uv 0.11.3` binary to preserve
  lockfile revision 3. The older Python Scripts `uv 0.6.6` rewrote the lockfile
  into an older format and was intentionally not used for the final lockfile.
- Cargo lock refresh required `CARGO_HTTP_CHECK_REVOKE=false` on this Windows
  host because the local TLS revocation check failed while contacting crates.io.

## Verification Commands

```powershell
gh api repos/zapabob/Turboquant-CUDA/dependabot/alerts
& 'C:\Users\downl\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe' lock --native-tls --upgrade-package urllib3
uv tree --package urllib3
uv tree --package sqlitedict
$env:CARGO_HTTP_CHECK_REVOKE='false'; cargo update --manifest-path rust\Cargo.toml -p rand@0.9.2 --precise 0.9.3
$env:CARGO_HTTP_CHECK_REVOKE='false'; cargo metadata --manifest-path rust\Cargo.toml --locked --format-version 1 --no-deps
& 'C:\Users\downl\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe' --native-tls run python -m pytest tests\test_runtime_eval.py tests\test_eval_scripts.py tests\test_triality_so8_audit.py -q
& 'C:\Users\downl\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe' --native-tls run python scripts\validate_repo_contract.py
```

## Verification Results

- `urllib3 v2.7.0` is locked.
- `sqlitedict` and `lm-eval` are no longer present in `uv.lock`.
- `rand 0.9.3` is locked; `rand 0.9.2` is no longer present.
- Focused Python regression: `17 passed, 1 skipped`.
- Repository contract: `Repository contract OK.`
