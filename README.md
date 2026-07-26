# J-space ablation vs CoT externalization

Homework Zero experiment harness for [CS 2881R](https://boazbk.github.io/mltheoryseminar/hw0-2026/), testing whether CoT protection against J-space ablation shrinks as math difficulty rises ([paper](https://transformer-circuits.pub/2026/workspace/index.html)).

See [preregistration.md](preregistration.md) for hypotheses and disconfirmation conditions (committed before main runs). Design detail: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Single config surface: `jspace.config.Settings` / `JSPACE_*` env vars (`JSPACE_MODEL`, `JSPACE_DEVICE`, `JSPACE_DTYPE`, `JSPACE_BAND_START`, `JSPACE_BAND_END`, `JSPACE_K`, `JSPACE_RUN_DIR`, …).

| Setting | Local (M4) | GPU (authoritative) |
| --- | --- | --- |
| Model | `Qwen/Qwen3-0.6B` (default) | `Qwen/Qwen3-4B` |
| Device / dtype | MPS / fp32 | CUDA / bf16 |
| Attn | `eager` | `eager` |

Local runs validate logic only. Report numbers from the GPU 4B run. The Neuronpedia lens is fit for **Qwen3-4B**; on 0.6B/1.7B the harness uses an identity Jacobian proxy for plumbing only.

## Frozen ablation (instrument-validated on 4B)

| Knob | Value |
| --- | --- |
| Selection | Top-k activated **J-lens token vectors** \(v_t = J^\top u_t\) by lens logit |
| Band | **[27, 33]** (manual lock on late-ramp diagnostic; not auto mid-plateau) |
| `k` | 10 |
| `exclude_topk` | 10 (clean next-token top-10, position-local) |
| Prompt tokens | ablated |
| Control | matched-norm random directions, seeds {0,1,2} |

Instrument gate: exact-match on multihop items the model gets right, with J ≫ random. See `runs/2026-07-26_instrument-run5/HANDOFF.md`.

## Runs layout

All experiment outputs go under [`runs/`](runs/README.md): one dated folder per session (`YYYY-MM-DD_<slug>/`), plus `HANDOFF.md`. Default write path is `runs/active` (`JSPACE_RUN_DIR` to override).

Token budgets: `scripts/00_calibrate_token_budgets.py` → `<run_dir>/calibration/token_budgets.json` (probe ceiling vs final cap; ablated arms ≥ 3× observed clean p95).

## Datasets (homework-fixed)

| Dataset | HF repo | Notes |
| --- | --- | --- |
| GSM8K | `openai/gsm8k` | easy anchor |
| MATH-500 | `HuggingFaceH4/MATH-500` | levels 1–5 = primary axis |
| AIME | `HuggingFaceH4/aime_2024` | **n = 30** (2024 I+II); no silent year mixing |

Target ~50 problems/cell for GSM8K and MATH levels; AIME uses the full set.

## Script order

```bash
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16
# optional: export JSPACE_RUN_DIR=runs/$(date +%F)_my-run

python scripts/00_calibrate_token_budgets.py --datasets gsm8k,math500,aime --problems-per-dataset 5
python scripts/01_harness_smoke.py --limit 5
python scripts/02_lens_sanity.py
python scripts/03_band_and_loading.py --n-prompts 64
# review band plot; override with --band-start/--band-end if needed
python scripts/04_multihop_gate.py --band-start 27 --band-end 33
python scripts/05_timing_probe.py --band-start 27 --band-end 33
python scripts/06_run_grid.py --band-start 27 --band-end 33
python scripts/07_judge_backtracks.py --traces runs/.../traces.jsonl
```

Instrument helpers (optional): `09`–`13` (probes, gold-lp triad, token trace, EM CIs).

## GPU runbook

1. Rent single A100 / L40S; install deps; set `JSPACE_*` as above.
2. Run **00 → 05** (budgets, lens, band, multihop gate, timing). Fail closed on gates.
3. Lock band/k from the 4B diagnostic (local band is throwaway).
4. Run **06** (resumes by default); size GSM8K/MATH cells from the timing probe; AIME = all 30.
5. Offline: **07** + hand-label ~30–50 spans.

## Disclosure checklist (report)

- Single medium-equivalent band (not paper strength sweep); locked `[27,33]` on a late-ramp curve.
- Ablation = J-lens **token** directions (not SVD of J); refill-to-k after exclusion; Gram–Schmidt span projection.
- Selection/holding mechanism is a conjecture beyond the paper’s two-hop evidence.
- Instrument validated on factual multihop (clean-correct EM); number-token loading ≈ 0 — math CoT may differ.
- AIME n=30 + floor risk → widest CIs on the hard anchor.

## Tests

```bash
pytest                 # fast suite (no weights)
pytest -m slow         # optional model-backed (incl. lens agreement)
```

## Lens

Pre-fitted Jacobian lens via `jlens` from `neuronpedia/jacobian-lens`
(`qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`).
