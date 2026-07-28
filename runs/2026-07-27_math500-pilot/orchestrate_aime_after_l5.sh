#!/usr/bin/env bash
# Wait for MATH500 L5 (90 arms), then run AIME n=15 with same science knobs.
set -euo pipefail

ROOT="/root/j-space-cot-ablation"
RUN_DIR="$ROOT/runs/2026-07-27_math500-pilot"
MATH_OUT="grid_math500_pilot"
AIME_OUT="grid_aime_pilot"
TRACES="$RUN_DIR/$MATH_OUT/traces.jsonl"
AIME_TRACES="$RUN_DIR/$AIME_OUT/traces.jsonl"
LOG="$RUN_DIR/orchestrate_aime.log"
LIMIT=15
TARGET_L5=$((LIMIT * 6))   # 90 arms
TARGET_AIME=$((LIMIT * 6)) # 90 arms

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

count_l5() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("$TRACES")
if not p.exists():
    print(0); raise SystemExit
n = sum(1 for line in p.read_text().splitlines()
        if line.strip() and json.loads(line).get("problem_id","").endswith("-L5"))
print(n)
PY
}

count_aime() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("$AIME_TRACES")
if not p.exists():
    print(0); raise SystemExit
n = sum(1 for line in p.read_text().splitlines()
        if line.strip() and json.loads(line).get("dataset") == "aime")
print(n)
PY
}

grid_running() {
  pgrep -f "scripts/06_run_grid.py" >/dev/null
}

run_aime() {
  log "START AIME grid limit=${LIMIT} out=${AIME_OUT}"
  python -u scripts/06_run_grid.py \
    --datasets aime --limit "$LIMIT" \
    --band-start 27 --band-end 31 --k 10 \
    --random-seeds 0 \
    --early-stop-on-answer \
    --out-name "$AIME_OUT" \
    2>&1 | tee -a "$RUN_DIR/${AIME_OUT}.log"
  local rc=${PIPESTATUS[0]}
  log "END AIME grid exit=${rc}"
  return "$rc"
}

log "=== AIME-after-L5 orchestrator started ==="
log "Waiting for L5 ${TARGET_L5}/90 arms in ${MATH_OUT}"

while true; do
  l5=$(count_l5)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] L5: ${l5}/${TARGET_L5} | GPU ${gpu}" >> "$RUN_DIR/monitor_aime.log"

  if [ "$l5" -ge "$TARGET_L5" ]; then
    if grid_running; then
      log "L5 traces complete (${l5}); waiting for grid process to exit..."
      sleep 120
      continue
    fi
    log "L5 complete (${l5}/${TARGET_L5}); grid idle"
    break
  fi

  if ! grid_running; then
    log "WARN: grid not running but L5 only ${l5}/${TARGET_L5} — sleeping (resume L5 manually if needed)"
  fi
  sleep 300
done

aime=$(count_aime)
if [ "$aime" -ge "$TARGET_AIME" ]; then
  log "AIME already complete (${aime}/${TARGET_AIME}); nothing to do"
  exit 0
fi

if grid_running; then
  log "WARN: another grid still running; waiting 5m before AIME launch"
  sleep 300
fi

run_aime || true

aime=$(count_aime)
log "AIME finished: ${aime}/${TARGET_AIME} arms"
log "=== AIME-after-L5 orchestrator done ==="
