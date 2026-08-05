#!/usr/bin/env bash
# Regenerate all paper figures from committed traces (no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-jspace-cache}"
mkdir -p "$MPLCONFIGDIR"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== backtrack metrics + judge-calibrated plots ==="
python scripts/17_backtrack_metrics.py --per-stratum 30 --no-plot
python scripts/19_backtrack_precision_plot.py
python scripts/20_backtrack_adjusted_plot.py

echo "=== EM / protection / token usage (supplementary + final) ==="
python scripts/21_pilot_em_plots.py
python scripts/22_final_report_plots.py
python scripts/23_backtrack_rate_and_count.py
python scripts/24_gold_logprob_plots.py

echo "Done. Paper figures: report/figures/final/"
