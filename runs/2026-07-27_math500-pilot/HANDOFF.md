# HANDOFF — MATH500 + AIME A40 run (pod shutdown 2026-07-29)

**Written:** 2026-07-29 ~19:45 UTC  
**Reason:** User shutting down RunPod pod. Resume on a new GPU from this branch.

---

## Repo / branch

| Item | Value |
|---|---|
| Repo | `lsshepard/j-space-cot-ablation` |
| Branch | `runpod-run-gsm8k-math500-n15` |
| Clone path | `/root/j-space-cot-ablation` (or your path) |
| Latest commit at shutdown | `0795e96` (+ uncommitted shutdown checkpoint) |

Push (if not already pushed):
```bash
git push https://x-access-token:${GH_TOKEN}@github.com/lsshepard/j-space-cot-ablation.git runpod-run-gsm8k-math500-n15
```

---

## Machine notes (previous pod)

| Item | Value |
|---|---|
| GPU | NVIDIA A40 48GB |
| PyTorch | **2.6.0+cu124** (NOT cu130 — driver only supports CUDA 12.8) |
| Venv | `.venv` in repo root |
| HF cache | `HF_HOME=/workspace/.cache/huggingface` |

VRAM probe at 8k passed (~11 GB peak on short runs; L4/L5 peaks ~25 GB observed).

---

## Frozen science knobs

```bash
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa          # NOT eager
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME=/workspace/.cache/huggingface
```

**Never `--fresh`** on existing out-names.

---

## Progress at shutdown

### MATH500 (`grid_math500_pilot/traces.jsonl`)

| Level | Arms | Problems | Status | cot_clean EM |
|---|---:|---:|---|---:|
| L1 | 90/90 | 15/15 | **DONE** | 0.867 |
| L3 | 90/90 | 15/15 | **DONE** | 0.933 |
| L5 | 90/90 | 15/15 | **DONE** | 0.733 |
| **L4** | **65/90** | **10/15** | **PARTIAL** | 0.636 (n=11) |
| L2 | 0/90 | 0/15 | **NOT STARTED** | — |

**Total MATH traces:** 335  
**Stopped mid-arm:** `math500-test-67-L4` — `direct_random` just finished; **`cot_random` next** (5/6 arms on problem 11).

### AIME (`grid_aime_pilot/traces.jsonl`)

| | |
|---|---|
| Arms | **90/90 DONE** |
| cot_clean EM | **0.467** (7/15) |
| CoT-protection | **−0.267** (J hurts CoT; direct floored at 1/15) |

---

## Results summary (for write-up)

### MATH500 per-level EM (`cot_clean` / protection)

| Level | cot_clean | directΔ−cotΔ (protection) | cot_j cap% |
|---|---:|---:|---:|
| L1 | 0.867 | 0.000 | 47% |
| L3 | 0.933 | +0.133 | 73% |
| L4 (partial n=11) | 0.636 | 0.000 | 55% |
| L5 | 0.733 | −0.067 | 53% |

**L1:** No accuracy separation across arms (all 0.867); J adds length/cap cost only.  
**L3:** Strong clean EM; modest positive protection signal (+0.13).  
**L5:** Harder; slight negative protection (−0.07).  
**L4:** Too partial for firm conclusions.

### AIME (final n=15)

| Condition | EM | Cap % |
|---|---:|---:|
| direct_clean | 0.067 | 93% |
| cot_clean | 0.467 | 47% |
| direct_j_ablated | 0.067 | 87% |
| cot_j_ablated | 0.200 | 87% |
| cot_random | 0.333 | 67% |
| direct_random | 0.000 | 100% |

CoT flips: J hurt 5×, J helped 1×, both✓ 2×, both✗ 7×. Direct floored — protection metric unreliable on direct arm.

---

## What to run next (priority order)

### 1. Finish L4 (resume)

```bash
cd /root/j-space-cot-ablation && source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME=/workspace/.cache/huggingface

python -u scripts/06_run_grid.py --datasets math500 --limit 15 --levels 4 \
  --band-start 27 --band-end 31 --k 10 --random-seeds 0 \
  --early-stop-on-answer --out-name grid_math500_pilot \
  2>&1 | tee -a runs/2026-07-27_math500-pilot/grid_math500_pilot.log
```

Expect: `resuming: 335 completed runs` → remaining **25 L4 arms** (finish problems 11–15).

### 2. Run L2 (after L4 hits 90/90)

```bash
python -u scripts/06_run_grid.py --datasets math500 --limit 15 --levels 2 \
  --band-start 27 --band-end 31 --k 10 --random-seeds 0 \
  --early-stop-on-answer --out-name grid_math500_pilot \
  2>&1 | tee -a runs/2026-07-27_math500-pilot/grid_math500_pilot.log
```

### 3. Optional: detach with orchestrator

`runs/2026-07-27_math500-pilot/orchestrate_post_aime_l4_l2.sh` was written for L4→L2 chaining but **AIME is already done** — for resume, either run commands above manually or adapt the script to skip the AIME wait block.

---

## Key paths

| Path | Role |
|---|---|
| `runs/2026-07-27_math500-pilot/README.md` | Original runbook |
| `runs/2026-07-27_math500-pilot/HANDOFF.md` | **This file** |
| `runs/2026-07-27_math500-pilot/AGENT_CONTEXT.md` | Earlier context (stale) |
| `grid_math500_pilot/traces.jsonl` | MATH traces (resume) |
| `grid_aime_pilot/traces.jsonl` | AIME traces (complete) |
| `grid_math500_pilot.log` / `grid_aime_pilot.log` | Per-arm logs |

---

## Git commit history (this run)

| Commit | Content |
|---|---|
| `c6c8b0c` | L1 complete |
| `09effd4` | L3 complete |
| `7d2d2fd` | L5 complete + partial AIME |
| `0795e96` | AIME complete + L4 partial (~63/90) |
| *(shutdown)* | L4 at 65/90 + this HANDOFF |

---

## Invariants / pitfalls

- **sdpa** not eager; **8k ceiling** global
- Retro-cap applied to first 45 L1 traces; backup at `traces.jsonl.pre_retro_cap.bak`
- `cot_j_ablated` cap hits dominate wall time (~7–12 min/arm; up to 60+ min on 8k spins)
- One GPU job at a time
- Bootstrap CIs deferred; per-problem `correct` in traces is sufficient
- PyTorch must be **cu124** on this driver — reinstall if venv rebuilt: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

---

## Pod shutdown

Grid and orchestrators **killed cleanly** at user request (~19:45 UTC 2026-07-29). Traces flushed to disk.
