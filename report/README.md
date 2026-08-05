# Report

Paper and figures for the homework write-up.

| File | Role |
| --- | --- |
| [`report_latex.txt`](report_latex.txt) | LaTeX source (compile from this directory so `figures/` paths resolve) |
| [`REFERENCES.md`](REFERENCES.md) | External links (paper, assignment, token-budget note) |

## Figures

| Directory | Contents |
| --- | --- |
| [`figures/final/`](figures/final/) | **Paper figures** — EM overview, CoT-protection, token usage, backtrack, gold logprob |
| [`figures/supplementary/`](figures/supplementary/) | Bootstrap variants, backtrack precision/regex, multihop gate plot, band diagnostic |
| [`figures/archive/`](figures/archive/) | Early per-cell EM plots (July 2026) |

Regenerate everything:

```bash
./scripts/reproduce_report.sh
```

Outputs land in `figures/final/` and `figures/supplementary/`. Judge-calibrated backtrack rates also update `runs/backtrack-analysis/*.json`.

## Figure ↔ script map

| Figure | Script |
| --- | --- |
| `final/em_overview_all.png`, `em_direct_all`, `em_cot_all`, `cot_protection_all`, `token_usage_by_set`, `cap_hit_rate_by_set`, `j_ablation_drops_by_set`, `band_selection` | `scripts/22_final_report_plots.py` |
| `final/backtrack_rate_and_count.png` | `scripts/23_backtrack_rate_and_count.py` |
| `final/gold_logprob_*.png` | `scripts/24_gold_logprob_plots.py` |
| `supplementary/backtrack_*`, `em_*_boot.png` | `scripts/17–21` |
| `supplementary/multihop_em_n93_27_31.png` | `scripts/04_multihop_gate.py` (committed gate run) |
