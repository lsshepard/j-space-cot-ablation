# `jspace` package

Core experiment harness. Import via `from jspace.<module> import …` after `pip install -e .`.

| Module | Role |
| --- | --- |
| `config` | `Settings` / `JSPACE_*` env configuration |
| `generate` | Greedy generation, thinking toggle, ablation loop |
| `ablation` | J-lens projection hooks + random control |
| `run_one` | Single problem → `ProblemRecord` |
| `extract` / `graded` | Answer parsing + teacher-forced gold logprob |
| `token_budgets` | Per-dataset caps from calibration |
| `backtrack_metrics` | Regex rates, stratified judge sample |
| `plotting` | Shared matplotlib styling for report figures |

Entry points are the numbered scripts in [`../scripts/`](../scripts/README.md).
