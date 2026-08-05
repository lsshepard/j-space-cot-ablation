# Rubrics

Frozen judge prompts for backtrack analysis (committed before inspecting result traces).

| File | Role |
| --- | --- |
| [`backtrack_judge.md`](backtrack_judge.md) | Criterion for *effective* backtrack (marker + real approach change) |
| [`judge_prompt.md`](judge_prompt.md) | Template assembled into each span job's `prompt` field |

Verdicts: `runs/backtrack-analysis/judge_verdicts.jsonl`. Regenerate jobs with `scripts/17_backtrack_metrics.py`.
