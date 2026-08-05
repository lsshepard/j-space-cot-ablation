# Scripts

Numbered pipeline for calibration, the main grid, and report regeneration.

## Regenerate paper figures (no GPU)

```bash
./scripts/reproduce_report.sh
```

## Full experiment pipeline (GPU)

```bash
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export JSPACE_RUN_DIR=runs/$(date +%F)_my-run

python scripts/00_calibrate_token_budgets.py --datasets gsm8k,math500,aime --problems-per-dataset 5
python scripts/01_harness_smoke.py --limit 5
python scripts/02_lens_sanity.py
python scripts/03_band_and_loading.py --n-prompts 64
python scripts/04_multihop_gate.py --fixture tests/fixtures/lens-eval-multihop.json \
  --limit 93 --band-start 27 --band-end 31
python scripts/05_timing_probe.py --band-start 27 --band-end 31
python scripts/06_run_grid.py --datasets gsm8k --limit 15 --band-start 27 --band-end 31 \
  --early-stop-on-answer --out-name grid_gsm8k_pilot
# … repeat 06 for math500 levels and aime (see runs/2026-07-27_math500-pilot/README.md)
```

## Script index

| Script | Purpose |
| --- | --- |
| `00` | Calibrate per-dataset token budgets |
| `01` | Harness smoke test |
| `02` | Lens arithmetic-intermediate sanity check |
| `03` | Band diagnostic + number-token loading |
| `04` | Multihop EM gate (instrument validation) |
| `05` | Timing probe for cell sizing |
| `06` | Main 6-condition × difficulty grid (resumes by default) |
| `07` | Emit backtrack span jobs for judging |
| `08` | Local preview (small model) |
| `09–13` | Instrument diagnostics (probes, gold-LP triad, CIs) |
| `14–16` | Trace utilities (rescore, retro-cap, VRAM probe) |
| `17` | Regex backtrack rates + stratified judge sample |
| `18` | Merge judge verdicts, agreement stats |
| `19–20` | Backtrack precision + adjusted-rate plots |
| `21` | EM + CoT-protection plots (bootstrap) |
| `22` | Curated final report figures |
| `23` | Backtrack rate + count two-panel figure |
| `24` | Gold-answer logprob plots |
| `reproduce_report.sh` | Run `17–24` for all paper figures |

Config: `jspace.config.Settings` and `JSPACE_*` env vars (see root README).
