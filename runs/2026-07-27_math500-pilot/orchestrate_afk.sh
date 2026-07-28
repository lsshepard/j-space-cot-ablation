#!/usr/bin/env bash
# AFK orchestrator: L1 → L3 → L2|L5 → extend n+3 per level while time allows.
set -euo pipefail

ROOT="/root/j-space-cot-ablation"
RUN_DIR="$ROOT/runs/2026-07-27_math500-pilot"
OUT="grid_math500_pilot"
TRACES="$RUN_DIR/$OUT/traces.jsonl"
LOG="$RUN_DIR/orchestrate_afk.log"
MONITOR="$RUN_DIR/monitor.log"
BRANCH_FILE="$RUN_DIR/BRANCH_LEVEL"  # optional override: echo 2 or 5 > this file before L3 ends

cd "$ROOT"
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
export JSPACE_ATTN_IMPLEMENTATION=sdpa
export JSPACE_TOKEN_BUDGET_CEILING=8000
export JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export JSPACE_RUN_DIR=runs/2026-07-27_math500-pilot
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

count_level() {
  local lv="$1"
  python3 - <<PY
import json
from pathlib import Path
p = Path("$TRACES")
if not p.exists():
    print(0)
    raise SystemExit
n = 0
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    pid = r.get("problem_id", "")
    if pid.endswith("-L${lv}"):
        n += 1
print(n)
PY
}

level_em() {
  local lv="$1" cond="$2"
  python3 - <<PY
import json
from pathlib import Path
p = Path("$TRACES")
flags = []
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("problem_id", "").endswith("-L${lv}") and r.get("condition") == "${cond}":
        flags.append(bool(r.get("correct")))
if not flags:
    print("nan")
else:
    print(sum(flags) / len(flags))
PY
}

wait_for_level() {
  local lv="$1" target="$2"
  while true; do
    local n
    n=$(count_level "$lv")
    local gpu
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] L${lv}: ${n}/${target} traces | GPU ${gpu}" >> "$MONITOR"
    if [ "$n" -ge "$target" ]; then
      log "L${lv} complete: ${n}/${target}"
      return 0
    fi
    # If grid not running and under target, something died
    if ! pgrep -f "scripts/06_run_grid.py.*grid_math500_pilot" >/dev/null; then
      log "WARN: grid not running but L${lv} only ${n}/${target} — will retry launch"
      return 1
    fi
    sleep 300
  done
}

run_grid() {
  local levels="$1" limit="$2"
  log "START grid levels=${levels} limit=${limit}"
  python -u scripts/06_run_grid.py \
    --datasets math500 --limit "$limit" --levels "$levels" \
    --band-start 27 --band-end 31 --k 10 \
    --random-seeds 0 \
    --early-stop-on-answer \
    --out-name "$OUT" \
    2>&1 | tee -a "$RUN_DIR/grid_math500_pilot.log"
  local rc=${PIPESTATUS[0]}
  log "END grid levels=${levels} limit=${limit} exit=${rc}"
  return "$rc"
}

pick_branch_level() {
  local l1_em l3_em override
  l1_em=$(level_em 1 cot_clean)
  l3_em=$(level_em 3 cot_clean)
  log "Branch check: L1 cot_clean EM=${l1_em} L3 cot_clean EM=${l3_em}"

  if [ -f "$BRANCH_FILE" ]; then
    override=$(tr -d '[:space:]' <"$BRANCH_FILE")
    case "$override" in
      2|5)
        log "Using BRANCH_LEVEL override: L${override}"
        echo "$override"
        return 0
        ;;
      *)
        log "WARN: ignoring invalid BRANCH_LEVEL='$override' (want 2 or 5); defaulting to L5"
        ;;
    esac
  fi

  log "No BRANCH_LEVEL override — defaulting to L5 (write 'echo 2 > $BRANCH_FILE' to pick L2)"
  echo 5
}

log "=== AFK orchestrator started ==="

# Phase 1: wait for in-flight L1 (limit=15, 90 arms)
if wait_for_level 1 90; then
  :
else
  log "Resuming L1 after stall"
  run_grid 1 15 || true
  wait_for_level 1 90 || true
fi

# Phase 2: L3
if [ "$(count_level 3)" -lt 90 ]; then
  run_grid 3 15 || true
  wait_for_level 3 90 || true
fi

# Phase 3: L2 or L5
BRANCH=$(pick_branch_level)
log "Selected branch level L${BRANCH}"
if [ "$(count_level "$BRANCH")" -lt 90 ]; then
  run_grid "$BRANCH" 15 || true
  wait_for_level "$BRANCH" 90 || true
fi

# Phase 4: extend n+3 iteratively (18, 21, 24, ...) on all three levels
LEVELS="1,3,${BRANCH}"
LIMIT=18
while [ "$LIMIT" -le 30 ]; do
  arms_per_level=$((LIMIT * 6))
  l1=$(count_level 1)
  l3=$(count_level 3)
  lb=$(count_level "$BRANCH")
  log "Extension check limit=${LIMIT}: L1=${l1} L3=${l3} L${BRANCH}=${lb} (need ${arms_per_level} each)"
  if [ "$l1" -ge "$arms_per_level" ] && [ "$l3" -ge "$arms_per_level" ] && [ "$lb" -ge "$arms_per_level" ]; then
    LIMIT=$((LIMIT + 3))
    continue
  fi
  run_grid "$LEVELS" "$LIMIT" || true
  sleep 60
  l1=$(count_level 1); l3=$(count_level 3); lb=$(count_level "$BRANCH")
  if [ "$l1" -ge "$arms_per_level" ] && [ "$l3" -ge "$arms_per_level" ] && [ "$lb" -ge "$arms_per_level" ]; then
  LIMIT=$((LIMIT + 3))
  fi
done

log "=== AFK orchestrator finished ==="
