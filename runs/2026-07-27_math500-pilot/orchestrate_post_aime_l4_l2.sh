#!/usr/bin/env bash
# After AIME n=15 completes: run MATH500 L4 then L2 (n=15 each), detached on pod.
set -euo pipefail

ROOT="/root/j-space-cot-ablation"
RUN_DIR="$ROOT/runs/2026-07-27_math500-pilot"
MATH_OUT="grid_math500_pilot"
AIME_OUT="grid_aime_pilot"
MATH_TRACES="$RUN_DIR/$MATH_OUT/traces.jsonl"
AIME_TRACES="$RUN_DIR/$AIME_OUT/traces.jsonl"
LOG="$RUN_DIR/orchestrate_post_aime.log"
MONITOR="$RUN_DIR/monitor_post_aime.log"
LIMIT=15
TARGET_AIME=$((LIMIT * 6))
TARGET_LEVEL=$((LIMIT * 6))

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

count_aime() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("$AIME_TRACES")
if not p.exists():
    print(0); raise SystemExit
print(sum(1 for line in p.read_text().splitlines()
            if line.strip() and json.loads(line).get("dataset") == "aime"))
PY
}

count_level() {
  local lv="$1"
  python3 - <<PY
import json
from pathlib import Path
p = Path("$MATH_TRACES")
if not p.exists():
    print(0); raise SystemExit
print(sum(1 for line in p.read_text().splitlines()
            if line.strip() and json.loads(line).get("problem_id","").endswith("-L${lv}")))
PY
}

grid_running() {
  pgrep -f "scripts/06_run_grid.py" >/dev/null
}

wait_for_aime() {
  log "Waiting for AIME ${TARGET_AIME}/90 arms"
  while true; do
    local n
    n=$(count_aime)
    local gpu
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] AIME: ${n}/${TARGET_AIME} | GPU ${gpu}" >> "$MONITOR"

    if [ "$n" -ge "$TARGET_AIME" ]; then
      if grid_running; then
        log "AIME traces done (${n}); waiting for grid to exit..."
        sleep 120
        continue
      fi
      log "AIME complete (${n}/${TARGET_AIME})"
      return 0
    fi

    if ! grid_running; then
      log "WARN: grid idle but AIME only ${n}/${TARGET_AIME} — will retry launch if needed"
    fi
    sleep 300
  done
}

wait_for_level() {
  local lv="$1"
  log "Waiting for L${lv} ${TARGET_LEVEL}/90 arms"
  while true; do
    local n
    n=$(count_level "$lv")
    local gpu
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] L${lv}: ${n}/${TARGET_LEVEL} | GPU ${gpu}" >> "$MONITOR"

    if [ "$n" -ge "$TARGET_LEVEL" ]; then
      if grid_running; then
        log "L${lv} traces done (${n}); waiting for grid to exit..."
        sleep 120
        continue
      fi
      log "L${lv} complete (${n}/${TARGET_LEVEL})"
      return 0
    fi

    if ! grid_running; then
      log "WARN: grid idle but L${lv} only ${n}/${TARGET_LEVEL} — retrying launch"
      return 1
    fi
    sleep 300
  done
}

run_math_level() {
  local lv="$1"
  log "START MATH500 L${lv} limit=${LIMIT} out=${MATH_OUT}"
  python -u scripts/06_run_grid.py \
    --datasets math500 --limit "$LIMIT" --levels "$lv" \
    --band-start 27 --band-end 31 --k 10 \
    --random-seeds 0 \
    --early-stop-on-answer \
    --out-name "$MATH_OUT" \
    2>&1 | tee -a "$RUN_DIR/${MATH_OUT}.log"
  local rc=${PIPESTATUS[0]}
  log "END MATH500 L${lv} exit=${rc}"
  return "$rc"
}

run_level_with_retry() {
  local lv="$1"
  if [ "$(count_level "$lv")" -ge "$TARGET_LEVEL" ]; then
    log "L${lv} already complete; skip"
    return 0
  fi
  while [ "$(count_level "$lv")" -lt "$TARGET_LEVEL" ]; do
    if grid_running; then
      log "Grid busy; waiting before L${lv} launch"
      sleep 120
      continue
    fi
    run_math_level "$lv" || true
    if wait_for_level "$lv"; then
      break
    fi
    log "Resuming L${lv} after stall"
  done
}

log "=== Post-AIME orchestrator started (L4 → L2) ==="

wait_for_aime

run_level_with_retry 4
run_level_with_retry 2

log "L4=$(count_level 4)/${TARGET_LEVEL} L2=$(count_level 2)/${TARGET_LEVEL}"
log "=== Post-AIME orchestrator done ==="
