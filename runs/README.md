# Runs

Experiment outputs: traces, calibration, gates, and RunPod session notes. **Not** library code.

## Paper reproduction (read these)

| Folder | What |
| --- | --- |
| [`2026-07-26_gsm8k-pilot/`](2026-07-26_gsm8k-pilot/) | GSM8K n=15 grid + calibration |
| [`2026-07-27_math500-pilot/`](2026-07-27_math500-pilot/) | MATH L1/L3/L4/L5 + AIME grids; orchestration shell scripts |
| [`backtrack-analysis/`](backtrack-analysis/) | Regex + judge backtrack analysis (539-span sample) |
| [`2026-07-26_band31-check/`](2026-07-26_band31-check/) | Multihop gate at band [27,31] (Fig. calibration right panel) |
| [`2026-07-26_instrument-run5/`](2026-07-26_instrument-run5/) | Instrument validation gates + diagnostics |

Each dated folder may include `HANDOFF.md` (pod shutdown notes), `calibration/`, grid subfolders, and `.log` files.

## Trace files (committed)

```
runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl
runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl
runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl
```

Most other `*.jsonl` under `runs/` are gitignored (gate runs, judge batches). Exceptions: `backtrack-analysis/judge_verdicts.jsonl`.

## Working directory

Scripts write to `JSPACE_RUN_DIR` (default `runs/active`). Seed calibration:

```bash
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
```

## Archive / development

| Folder | Notes |
| --- | --- |
| `2026-07-24_calibration` | Initial 4B band/budget calibration |
| `2026-07-25_instrument-run4` | Pre-fix instrument era |
| `2026-07-26_smoke-gsm8k-n6` | Tiny smoke run |
| `archive_early-plumbing` | Local preview leftovers |
| `active` | Default write target (mostly gitignored) |

## RunPod one-offs

Shell orchestrators and monitors live **inside** the run they served, e.g.:

- `2026-07-27_math500-pilot/orchestrate_afk.sh`
- `2026-07-27_math500-pilot/orchestrate_aime_after_l5.sh`
- `2026-07-27_math500-pilot/watch_progress.sh`

Do not run `--fresh` on committed grid out-names.
