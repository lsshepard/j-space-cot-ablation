# Handoff: band `[27,31]` full93 EM check

**Date:** 2026-07-26  
**Branch tip:** commit on `runpod-band31-check` (off `main` / merged `runs-cleanup`)  
**Model:** `Qwen/Qwen3-4B`, CUDA A100-80GB, bfloat16

## Kill confirmation ([27,33] pilot)

- Aborted mid-load GSM8K pilot started with `--band-end 33` (`scripts/06_run_grid.py`).
- No `grid_gsm8k_pilot/traces.jsonl` was written; partial dir removed.
- GPU confirmed idle (`0 MiB`) before the band check.

## Full93 multihop EM — band `[27,31]`

Same fixture/knobs as prior full93; only `band_end` changed (31 vs 33).  
Re-ran clean + J + random (seeds 0,1,2); did not reuse old arms.

Artifacts: `gates/multihop_em_full93_b27-31.{jsonl,_summary.json}`, `gates/em_gate_ci.json`.

| Population | n | clean | J | random | random − J | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| clean **correct** | 34 | 1.000 | 0.529 | 0.784 | **+0.255** | **[+0.049, +0.451]** |
| clean **wrong** | 59 | 0.000 | 0.169 | 0.102 | −0.068 | [−0.169, +0.028] |
| all | 93 | 0.366 | 0.301 | 0.351 | +0.050 | [−0.054, +0.151] |

### vs prior `b27-33` full93 (`runs/2026-07-26_instrument-run5`)

| | clean-correct J | random | rand−J CI |
| --- | --- | --- | --- |
| `[27,33]` | 0.529 | 0.765 | +0.235 [+0.049, +0.422] |
| `[27,31]` | 0.529 | 0.784 | +0.255 [+0.049, +0.451] |

Clean-correct J-specific drop still clear (CI excludes 0). **`[27,31]` OK for GSM8K pilot.**

## GSM8K pilot status (sibling run dir)

- Timing probe (CoT+J, n=5, band was still `[27,33]` in CLI): mean **381 s/problem** → chose n=15.
- Pilot restarted **fresh** with `--band-start 27 --band-end 31` under `runs/2026-07-26_gsm8k-pilot/` (in progress at commit time; not claimed finished here).
