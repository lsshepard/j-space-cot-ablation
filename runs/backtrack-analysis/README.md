# Backtrack analysis (regex + judge sample)

Branch: `backtrack-analysis`

## Regex headline (all CoT traces)

```bash
python scripts/17_backtrack_metrics.py
```

Outputs:
- `runs/backtrack-analysis/regex_summary.json`
- `runs/backtrack-analysis/judge_sample_jobs.jsonl` (90 spans, 5/cell×condition)
- `runs/backtrack-analysis/hand_label_template.jsonl` (first 50 for hand labels)
- `report/figures/supplementary/backtrack_regex_rates.png`

Current pilot inputs: gsm8k + math500 + aime traces (270 CoT arms; L4 complete n=15).

## Stratified precision test (n=30/stratum, 539 total)

```bash
python scripts/17_backtrack_metrics.py --per-stratum 30
# judge spans → runs/backtrack-analysis/judge_verdicts.jsonl
python scripts/19_backtrack_precision_plot.py
python scripts/20_backtrack_adjusted_plot.py
```

Outputs:
- `runs/backtrack-analysis/precision_by_stratum.json`
- `runs/backtrack-analysis/adjusted_rates_boot.json`
- `report/figures/supplementary/backtrack_precision_by_stratum.png`
- `report/figures/supplementary/backtrack_adjusted_rates_boot.png`

Overall PPV ≈ 0.37 (198/539). L4 resampled after trace completion (shared RNG shifts L5 sample too); 158 new spans judged in `judge_batches_l4/`.


## Judge sample

Fill verdicts with an LLM subagent using each job's frozen `prompt` field. Write JSONL rows:

```json
{"span_id": "...", "is_backtrack": true, "approach_before": "...", "approach_after": "...", "source": "subagent"}
```

Score agreement:

```bash
python scripts/18_backtrack_judge_score.py \
  --jobs runs/backtrack-analysis/judge_sample_jobs.jsonl \
  --verdicts runs/backtrack-analysis/judge_verdicts_batch*.jsonl \
  --hand-labels runs/backtrack-analysis/hand_label_template.jsonl
```

Regex treats every marker span as positive; judge applies the effective-backtrack criterion (§4.4).
