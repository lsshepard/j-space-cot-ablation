# Runs

Experiment outputs and session notes live here — not permanent product code.

## Convention

- One flat folder per run: `YYYY-MM-DD_<slug>/` (kind + id in the name).
- No required internal layout: dump whatever the scripts produced.
- Include a `HANDOFF.md` when a pod/session finishes.

## Working directory

Scripts write under `Settings.run_dir` (default `runs/active`).

```bash
export JSPACE_RUN_DIR=runs/2026-07-26_my-grid   # optional; creates a dedicated run folder
```

`runs/active/calibration/` is seeded with the locked band and token budgets so `04`–`06` keep working without pointing at an old dated folder.

## Current folders

| Folder | Contents |
| --- | --- |
| `2026-07-24_calibration` | 4B band / budgets / lens sanity |
| `2026-07-25_instrument-run4` | Handoff (pre-fix evidence era) |
| `2026-07-26_instrument-run5` | Confirmed instrument gates + diagnostics |
| `2026-07-26_smoke-gsm8k-n6` | Tiny gsm8k CoT smoke |
| `archive_early-plumbing` | Local preview / early harness leftovers |
| `active` | Default write target for the next run |
