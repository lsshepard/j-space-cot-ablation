# Handoff: GSM8K pilot n=15 — band `[27,31]`, ablated ×6, early-stop

**Date:** 2026-07-26 (finished ~2026-07-27 UTC)  
**Model:** `Qwen/Qwen3-4B`, A100-80GB, bfloat16  
**Out:** `grid_gsm8k_pilot/` (90 traces)

## Knobs

| Knob | Value |
| --- | --- |
| Band | **[27, 31]** |
| k / exclude_topk | 10 / 10 |
| Random seeds | **0 only** (full runs should use 0,1,2) |
| Ablated CoT budget | **5394** (`JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6`) |
| Clean CoT budget | 2048 |
| Early-stop-on-answer | **on** (require closed `</think>` + `#### num\n`) |
| Sync cleanup | on-device exclusion / selection (hot path) |

Timing probe (pre-pilot, CoT+J @ ×3/2697): mean **381 s/problem** → chose n=15.

## Kill / archive trail

- Aborted `[27,33]` mid-load (no traces).
- `grid_gsm8k_pilot_aborted_b27-31_x3` — early ×3 abort.
- `grid_gsm8k_pilot_partial_no-earlystop_x6` — **54 traces** before early-stop work (**keep**).
- `grid_gsm8k_pilot_broken_earlystop_midnum` — mid-`#### 1` bug forensics.

## Per-condition (pooled n=15)

| Condition | EM | cap-hits | mean len | max len |
| --- | --- | --- | --- | --- |
| direct_clean | 0.933 | 0/15 | 177 | 314 |
| direct_j_ablated | 0.800 | 0/15 | 175 | 303 |
| direct_random | 0.933 | 0/15 | 160 | 313 |
| cot_clean | 0.933 | 1/15 | 783 | 2048 |
| cot_j_ablated | 0.867 | **6/15** | **2918** | 5394 |
| cot_random | 1.000 | 0/15 | 963 | 3946 |

## Stratified by direct_clean correct (n=14 / 1)

| Arm | clean-correct EM | cap | mean len |
| --- | --- | --- | --- |
| cot_clean | **1.000** | 0/14 | 693 |
| cot_j_ablated | **0.929** | **5/14** | **2741** |
| cot_random | **1.000** | 0/14 | 915 |

CoT-protection sketch (clean-correct):  
`(direct_clean − direct_j) − (cot_clean − cot_j) = (1.000 − 0.857) − (1.000 − 0.929) = **+0.071**`  
(tiny n; do not overclaim).

## vs n=6 smoke (`runs/2026-07-26_smoke-gsm8k-n6`)

Smoke: direct ceiling 1.0 all arms; CoT EM **identical** 0.83 clean/J/random; J only lengthened.  
This pilot: **escapes identical CoT EM** on clean-correct (J 0.929 vs clean/random 1.0) and shows clear **J length + cap-hit** (6/15 vs 0 random). Still near-ceiling; not a decisive “J kills math CoT” claim from pooled EM alone.

## Early-stop outcome

**0/90 early_stops.** Cap still hit on 6/15 `cot_j` (and 1/15 `cot_clean`). Likely cause: post-answer / answer-draft loops **inside an open `<think>`**, which the stop rule deliberately ignores. Fixing in-think stop is the main MATH-500 wall-time lever; do not treat current early-stop as solving J spin.

## Go / no-go

- **Go** for MATH-500 L1–L2 (or full L1–L5 @ n=15) as next difficulty probe — instrument bite visible via length/caps; small CoT EM separation on clean-correct.
- Keep reporting stratified EM + cap/length; full runs need random seeds 0,1,2.
- Prefer improving early-stop (in-think / boxed completion) before burning multi-day MATH ablated CoT at 16k ceiling.
- Do **not** parallelize GPU jobs while `HF_HOME` is on FUSE; ~50% util is sync-bound.

## Reproduce

```bash
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_RUN_DIR=runs/2026-07-26_gsm8k-pilot
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
python -u scripts/06_run_grid.py --datasets gsm8k --limit 15 \
  --band-start 27 --band-end 31 --k 10 --random-seeds 0 \
  --early-stop-on-answer --out-name grid_gsm8k_pilot
# omit --fresh to resume
```
