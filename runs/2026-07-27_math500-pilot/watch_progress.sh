#!/usr/bin/env bash
# Progress watcher — works without `watch`/curses (TERM=dumb safe).
INTERVAL="${1:-30}"
TRACES="/root/j-space-cot-ablation/runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl"
LOG="/root/j-space-cot-ablation/runs/2026-07-27_math500-pilot/grid_math500_pilot.log"

while true; do
  clear 2>/dev/null || printf '\033[2J\033[H'
  echo "=== $(date -u) === (refresh every ${INTERVAL}s, Ctrl+C to quit)"
  echo
  wc -l "$TRACES"
  python3 - "$TRACES" <<'PY'
import json, sys
from collections import defaultdict
p = sys.argv[1]
by = defaultdict(int)
with open(p) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("problem_id", "")
        for lv in range(1, 6):
            if pid.endswith(f"-L{lv}"):
                by[lv] += 1
                break
for lv in sorted(by):
    print(f"  L{lv}: {by[lv]}/90")
PY
  echo
  echo "--- last 3 log lines ---"
  tail -3 "$LOG"
  echo
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null || true
  echo
  ps aux | grep -E 'orchestrate_afk|06_run_grid' | grep -v grep || echo "(no grid/orchestrator running)"
  sleep "$INTERVAL"
done
