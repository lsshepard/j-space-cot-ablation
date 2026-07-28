# Agent context — MATH500 A40 run (2026-07-27)

**Written:** 2026-07-27 ~15:47 UTC  
**Purpose:** Recover SSH/agent context if the Cursor session is lost. This pod keeps running jobs independently of IDE connectivity.

---

## Repo / branch

| Item | Value |
|---|---|
| Repo | `lsshepard/j-space-cot-ablation` |
| Clone path | `/root/j-space-cot-ablation` |
| Branch | `runpod-run-gsm8k-math500-n15` |
| Base handoff commit | `c64f4ba` (8k ceiling migration) |
| L1 checkpoint commit | `c6c8b0c` (*Complete MATH500 L1 cell on A40*) — **1 commit ahead of origin** (user may have pushed via `GH_TOKEN`) |

---

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA A40 48GB |
| PyTorch | **2.6.0+cu124** (had to downgrade from cu130 — driver only supports CUDA 12.8) |
| Venv | `/root/j-space-cot-ablation/.venv` |
| HF cache | `HF_HOME=/workspace/.cache/huggingface` |

### Frozen science knobs

```bash
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa          # NOT eager
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME=/workspace/.cache/huggingface
```

Grid args: `--datasets math500 --limit 15 --band-start 27 --band-end 31 --k 10 --random-seeds 0 --early-stop-on-answer --out-name grid_math500_pilot`

**Never use `--fresh`** on `grid_math500_pilot` (wipes resume).

---

## Run layout

| Path | Role |
|---|---|
| `runs/2026-07-27_math500-pilot/README.md` | Primary runbook |
| `runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl` | Resume-safe traces |
| `runs/2026-07-27_math500-pilot/grid_math500_pilot/run_meta.json` | Run metadata |
| `runs/2026-07-27_math500-pilot/grid_math500_pilot.log` | Per-arm stdout log |
| `runs/2026-07-27_math500-pilot/orchestrate_afk.sh` | **Pod-side orchestrator** (not Cursor) |
| `runs/2026-07-27_math500-pilot/orchestrate_afk.log` | Orchestrator log |
| `runs/2026-07-27_math500-pilot/monitor.log` | 5-min GPU/trace snapshots (L1 phase only) |
| `runs/2026-07-27_math500-pilot/BRANCH_LEVEL` | **Optional override** before L3 ends: `2` or `5` |

---

## Progress at last write

| Level | Arms | Problems | Status |
|---|---:|---:|---|
| **L1** | 90/90 | 15/15 | **Complete** (committed `c6c8b0c`) |
| **L3** | 33/90 | 5/15 | **In progress** (~problem 6, `math500-test-19-L3`) |
| **L5/L2** | 0/90 | — | Queued by orchestrator |

**Total traces:** 123 lines in `traces.jsonl`

### L1 results (final)

All six arms: **EM = 0.867** (13/15). No accuracy separation; `cot_j_ablated` mean len **4632** vs `cot_clean` **1619**, cap hit **47%** vs **7%**.

### L3 results (partial, ~5 problems)

`cot_clean` EM ~0.80 so far. One J-helped flip (test-2), one J-hurt (test-10). Still well above 0.4 branch threshold.

### ETA (rough)

~7.5 min/arm → L3 ~7h remaining at last estimate; L5 similar after that.

---

## Processes on pod (detached — survive connection loss)

Check with:
```bash
ps aux | grep -E 'orchestrate_afk|06_run_grid' | grep -v grep
```

At last write:
- **Orchestrator:** `bash runs/2026-07-27_math500-pilot/orchestrate_afk.sh` (PID ~15414, `nohup`, TTY=?)
- **L3 grid:** `python scripts/06_run_grid.py ... --levels 3` (PID ~430903)

Started via `nohup`; stdin → `/dev/null`. **Losing SSH/Cursor does NOT kill these.** Stopping the RunPod pod does.

---

## Orchestrator plan

`orchestrate_afk.sh` runs **on the pod**, not in Cursor:

1. ~~Wait for L1 90/90~~ ✓
2. **Run L3** (in progress — orchestrator blocked inside `run_grid 3`)
3. When L3 hits 90/90 → **default L5** (updated from auto EM branch)
4. Optional n+3 extension (18, 21, …) on L1,L3,branch level

### Branch override (intervene before L3 finishes)

```bash
# Pick L2 instead of default L5:
echo 2 > runs/2026-07-27_math500-pilot/BRANCH_LEVEL

# Explicit L5:
echo 5 > runs/2026-07-27_math500-pilot/BRANCH_LEVEL
```

Orchestrator reads `BRANCH_LEVEL` when L3 completes. If absent → **L5**.

---

## VRAM probe

- 1024 tokens: **passed** (~10.6 GB peak)
- 8000 probe: **aborted early** to start grid; runtime showed ~11 GB, no OOM — A40 is sufficient at 8k.

---

## Git state (uncommitted at last write)

Modified since `c6c8b0c`:
- `grid_math500_pilot/traces.jsonl` (L3 partial)
- `grid_math500_pilot.log`

Untracked:
- `orchestrate_afk.sh`, `orchestrate_afk.log`, `monitor.log`, `vram_probe_8000.log`

Commit L3 when complete; final `HANDOFF.md` when all three levels done.

---

## Quick recovery commands

```bash
cd /root/j-space-cot-ablation
source .venv/bin/activate
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot HF_HOME=/workspace/.cache/huggingface

# Progress
wc -l runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl
tail -10 runs/2026-07-27_math500-pilot/grid_math500_pilot.log
tail -10 runs/2026-07-27_math500-pilot/orchestrate_afk.log
nvidia-smi

# If grid died but orchestrator alive — resume manually (NO --fresh):
python -u scripts/06_run_grid.py --datasets math500 --limit 15 --levels 3 \
  --band-start 27 --band-end 31 --k 10 --random-seeds 0 \
  --early-stop-on-answer --out-name grid_math500_pilot \
  2>&1 | tee -a runs/2026-07-27_math500-pilot/grid_math500_pilot.log
```

Push with exported PAT:
```bash
git push https://x-access-token:${GH_TOKEN}@github.com/lsshepard/j-space-cot-ablation.git runpod-run-gsm8k-math500-n15
```

---

## Decisions / notes

- User went AFK; wanted max throughput on A40 (not upgrade GPU).
- Bootstrap CIs deferred to plotting time; traces have per-problem `correct` for later error bars.
- Retro-cap at 8k already applied to first 45 traces; backup at `traces.jsonl.pre_retro_cap.bak`.
- One GPU job at a time if HF cache on FUSE (cache is on `/workspace` here).
