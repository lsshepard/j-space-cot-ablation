# MATH500 full run — A40 migration handoff

**Branch tip should include commit with 8k ceiling + retro-cap + this README.**  
**Before killing the old A100 pod: `git push` (traces are force-added; jsonl is normally gitignored).**

This is the **budget-limited full MATH500 cell run** (n=15 per level), not a tiny pilot.

---

## Current state (when A100 pod stopped)

| Item | Value |
|---|---|
| Run dir | `runs/2026-07-27_math500-pilot/` |
| Traces | `grid_math500_pilot/traces.jsonl` — **45 / 90 L1 arms** done |
| Levels done | L1 partial (7 full 6-arm problems + partial `math500-test-79-L1`) |
| Ceiling | **8000** tokens (all arms) |
| Retro-cap | 4 traces with `len>8k` truncated + rescored (`scripts/15_retro_cap_traces.py`); **0 correct flips**. Backup: `traces.jsonl.pre_retro_cap.bak` |
| Band | `[27, 31]`, k=10, seed 0, early-stop on |
| Model | `Qwen/Qwen3-4B`, bf16, **sdpa** (eager OOMs on long dual-KV) |
| Ablated mult | `JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6` |
| Next level plan | Finish **L1** → **L3** @8k → if L3 `cot_clean` EM ≲0.4 (or clearly worse than L1) then **L2**, else **L5** |

Science knobs frozen aside from the cost ceiling.

---

## Why A40 + 8k

- Long dual-KV ablated CoT dominated wall time (~45–60 min per 16k spin).
- L1 data: almost all successful CoT finishes ≪8k; 16k tails were mostly spin (one correct answer existed by ~768 tokens).
- A40 (**48GB**, ~$0.44/hr) should fit **8k dual KV + sdpa**; was risky at 16k.
- Still expect ~50% util (lens full-vocab GEMV + dual forward), not a “slow GPU bug.”

---

## Step 0 — machine setup

```bash
cd /root/j-space-cot-ablation   # or your clone path
git fetch && git checkout <branch-with-handoff> && git pull

python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'   # or whatever this repo uses — match A100 env if listed in pyproject

# Caches: prefer fast local disk if FUSE was a bottleneck on A100
export HF_HOME=/workspace/.cache/huggingface   # or local NVMe path
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
```

Confirm GPU:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
# expect something like NVIDIA A40, 46068 MiB
```

---

## Step 1 — VRAM probe (required on A40)

Do **not** start the full grid until this passes.

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

# Warmup short
python -u scripts/16_a40_vram_probe.py --tokens 1024

# Full ceiling stress (may take tens of minutes if it doesn't early-stop)
python -u scripts/16_a40_vram_probe.py --tokens 8000
```

**Pass criteria:** exits 0, no `CUDA out of memory`, `peak_allocated_gb` ideally **≲44**.

**If OOM at 8000:** stop and either stay on A100 or lower ceiling to **6000**, re-run `scripts/15_retro_cap_traces.py --cap 6000 --inplace` on traces (from the `.bak` if needed), update this README, then continue.

---

## Step 2 — finish L1 (resume, no --fresh)

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

python -u scripts/06_run_grid.py --datasets math500 --limit 15 --levels 1 \
  --band-start 27 --band-end 31 --k 10 \
  --random-seeds 0 \
  --early-stop-on-answer \
  --out-name grid_math500_pilot \
  2>&1 | tee -a "$JSPACE_RUN_DIR/grid_math500_pilot.log"
```

Expect: `resuming: 45 completed runs` then remaining L1 arms to **90**.

---

## Step 3 — L3, then branch L2 vs L5

After L1 hits 90 arms, summarize EM (especially `cot_clean` / clean-correct / length+caps). Then:

```bash
# Same env as above, ceiling still 8000
python -u scripts/06_run_grid.py --datasets math500 --limit 15 --levels 3 \
  --band-start 27 --band-end 31 --k 10 \
  --random-seeds 0 \
  --early-stop-on-answer \
  --out-name grid_math500_pilot \
  2>&1 | tee -a "$JSPACE_RUN_DIR/grid_math500_pilot.log"
```

**Branch rule (from experiment owner):**
- If L3 `cot_clean` EM is **low** (≲ **0.4**, or clearly much worse than L1) → run `--levels 2`
- Else → run `--levels 5`

Never run the default L1–L5 loop; always pass `--levels`.

---

## Step 4 — when done

1. Write `HANDOFF.md` in this run dir: per-level EM, length/cap tables, clean-correct CoT deltas, any OOM/resume notes.
2. Commit run meta + handoff (+ force-add traces if still gitignored).
3. Optional: AIME with same knobs after MATH.

---

## Important invariants

- **Never** `--fresh` on this out-name (wipes resume).
- **sdpa** not eager.
- **8k ceiling** via `JSPACE_TOKEN_BUDGET_CEILING=8000` (now applies to **clean and ablated**).
- Retro-cap used **truncate + rescore**, not blanket `correct=False`.
- One GPU job at a time if `HF_HOME` is on FUSE.

---

## Code pointers

| Path | Role |
|---|---|
| `scripts/06_run_grid.py` | `--levels`, resume, `empty_cache` between arms |
| `scripts/15_retro_cap_traces.py` | 8k counterfactual truncate+rescore |
| `scripts/16_a40_vram_probe.py` | A40 OOM/peak VRAM probe |
| `src/jspace/token_budgets.py` | global ceiling |
| `src/jspace/generate.py` | dual-KV ablation decode; sparse early-stop |
| `AFK_PLAN.md` | earlier plan notes (ceiling was 10k/16k; **8k supersedes**) |

---

## Clarify with owner if

- L3 “low EM” threshold should differ from ~0.4
- Whether to also run random seeds 1,2 (currently seed **0** only)
- Whether AIME starts immediately after MATH
