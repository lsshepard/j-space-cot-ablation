# J-space ablation vs CoT externalization

Replication code and results for *Does Chain-of-Thought Protection Against J-Space Ablation Survive Increasing Difficulty?* (CS 2881R Homework Zero, 2026).

Tests whether CoT shields Qwen3-4B from Jacobian-lens J-space ablation as math difficulty rises, following [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/reproduce_report.sh    # figures from committed traces (~1 min, no GPU)
pytest                           # unit tests
```

Paper: [`report.pdf`](report.pdf) · LaTeX source: [`report/report_latex.txt`](report/report_latex.txt) · Figures: [`report/figures/final/`](report/figures/final/) · Pre-registration: [`preregistration.md`](preregistration.md)

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/jspace/`](src/jspace/) | Core library: generation, ablation hooks, scoring, metrics |
| [`scripts/`](scripts/README.md) | Numbered pipeline (`00` calibration → `06` grid → `17–24` analysis/plots) |
| [`runs/`](runs/README.md) | Committed traces, calibration artifacts, RunPod handoffs |
| [`report/`](report/README.md) | LaTeX source, references, paper + supplementary figures |
| [`rubrics/`](rubrics/README.md) | Frozen backtrack-judge prompts |
| [`docs/`](docs/README.md) | Full experiment design spec (pre-implementation) |
| [`tests/`](tests/) | Fast unit tests (+ optional `pytest -m slow` with weights) |

## Main results (committed traces)

| Cell | Traces | Run folder |
| --- | --- | --- |
| GSM8K (n=15) | `runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl` | instrument + easy anchor |
| MATH L1/L3/L4/L5 (n=15 each) | `runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl` | primary difficulty gradient |
| AIME 2024 (n=15) | `runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl` | hard anchor (direct rerun @ 4.1k) |
| Backtrack judge | `runs/backtrack-analysis/` | 539-span stratified sample |

## Frozen science knobs (4B GPU runs)

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B`, bf16, CUDA |
| Band | layers **[27, 31]**, k=10, exclude clean top-10 |
| Token ceiling | 8000; ablated arms ≥ 6× clean p95 |
| Decoding | greedy, seed 0, early-stop on scorable answer |
| Attn | `sdpa` on A40 (eager OOMs on long dual-KV ablation) |

Instrument gate: Anthropic 93-item multihop set (`tests/fixtures/lens-eval-multihop.json`), J below clean and random — see `runs/2026-07-26_band31-check/`.

## Re-running experiments (GPU)

Full grid reproduction requires a GPU, HF access, and the pre-fitted lens from `neuronpedia/jacobian-lens`. See [`scripts/README.md`](scripts/README.md) for the `00`→`06` pipeline and [`runs/2026-07-27_math500-pilot/README.md`](runs/2026-07-27_math500-pilot/README.md) for the A40 runbook.

Set `JSPACE_RUN_DIR` to a dated folder under `runs/` before launching.

## Limitations (disclose in paper)

- Direct arm uses `enable_thinking=False` but prompts still request step-by-step solutions; contrast is thinking-mode off vs on, not answer-only vs CoT.
- Single ablation band (not paper strength sweep); n=15/cell; MATH L2 omitted.
- AIME direct re-budgeted after initial 512-token fallback; CoT arms hit cap on hard cells.
- Number-token J-space loading ≈ 0 on calibration probes.

## Citation

```bibtex
@misc{shepard2026jspacecot,
  author = {Liam Robert Shepard},
  title  = {Does Chain-of-Thought Protection Against J-Space Ablation Survive Increasing Difficulty?},
  year   = {2026},
  howpublished = {\url{https://github.com/lsshepard/j-space-cot-ablation}}
}
```
