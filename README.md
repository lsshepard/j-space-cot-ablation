# J-space ablation vs CoT externalization

Homework Zero experiment harness for [CS 2881R](https://boazbk.github.io/mltheoryseminar/hw0-2026/), testing whether CoT protection against J-space ablation shrinks as math difficulty rises ([paper](https://transformer-circuits.pub/2026/workspace/index.html)).

See [preregistration.md](preregistration.md) for hypotheses and disconfirmation conditions (committed before main runs).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Single config surface: `jspace.config.Settings` / `JSPACE_*` env vars (`JSPACE_MODEL`, `JSPACE_DEVICE`, `JSPACE_DTYPE`, `JSPACE_BAND_START`, `JSPACE_BAND_END`, `JSPACE_K`, `JSPACE_MAX_NEW_TOKENS` as debug override, …).

**Token budgets (§4.7):** `scripts/00_calibrate_token_budgets.py` measures unablated direct vs CoT trace lengths per dataset, saves `results/calibration/token_budgets.json` (caps = p95 × multiplier, default 2×). All runners resolve per-arm caps from that file unless `JSPACE_MAX_NEW_TOKENS` is set.

| Setting | Local (M4) | GPU (authoritative) |
| --- | --- | --- |
| Model | `Qwen/Qwen3-0.6B` (default) | `Qwen/Qwen3-4B` |
| Device / dtype | MPS / fp32 | CUDA / bf16 |
| Attn | `eager` | `eager` |

Local runs validate logic only. Small models may show no workspace band — expected. Report numbers from the GPU 4B run.

The Neuronpedia lens is fit for **Qwen3-4B** (`d_model=2560`). On 0.6B/1.7B the harness loads it for API smoke but substitutes an **identity Jacobian proxy** for J-ablation plumbing (warning printed once). Authoritative J-ablation requires `JSPACE_MODEL=Qwen/Qwen3-4B`.

## Datasets (homework-fixed)

| Dataset | HF repo | Notes |
| --- | --- | --- |
| GSM8K | `openai/gsm8k` | easy anchor |
| MATH-500 | `HuggingFaceH4/MATH-500` | levels 1–5 = primary axis |
| AIME | `HuggingFaceH4/aime_2024` | **n = 30** (2024 I+II); no silent year mixing |

Target ~50 problems/cell for GSM8K and MATH levels; AIME uses the full set.

## Script order

```bash
python scripts/00_calibrate_token_budgets.py --datasets gsm8k --problems-per-dataset 5
python scripts/01_harness_smoke.py --limit 5
python scripts/08_preview_local.py --limit 3 --calibrate   # local sanity
python scripts/02_lens_sanity.py
python scripts/03_band_and_loading.py
python scripts/04_multihop_gate.py
python scripts/05_timing_probe.py
# after band/k fixed on 4B:
python scripts/06_run_grid.py
# after CoT traces exist:
python scripts/07_judge_backtracks.py --traces results/.../traces.jsonl
```

## GPU runbook

1. Rent single A100 40GB / L40S.
2. Install deps; `export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16`.
3. Run **02 → 05** first (lens, band, multihop gate, timing). Fail closed on gates.
4. Set `--band-start` / `--band-end` from the 4B diagnostic (local band is throwaway).
5. Run **06** (resumes by default; `--fresh` to wipe); size GSM8K/MATH cells from the timing probe; AIME = all 30.
   Pin revisions via `JSPACE_MODEL_REVISION`, `JSPACE_GSM8K_REVISION`, `JSPACE_MATH500_REVISION`, `JSPACE_AIME_REVISION`, `JSPACE_LENS_REVISION` (resolved SHAs are also written to `run_meta.json`).
6. Offline: **07** + hand-label ~30–50 spans; report regex↔judge and hand↔judge agreement.

## Disclosure checklist (report)

- Single medium-equivalent band (not paper strength sweep).
- Selection/holding mechanism is a conjecture beyond the paper’s two-hop evidence.
- AIME n=30 + floor risk → widest CIs on the hard anchor.
- Number-token J-space loading result.
- Lens fit scale from Neuronpedia metadata.

## Tests

```bash
pytest                 # fast suite (no weights)
pytest -m slow         # optional model-backed
```

## Lens

Pre-fitted Jacobian lens via `jlens` from `neuronpedia/jacobian-lens`
(`qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`). Verify revision and CREDIT.md fit scale on first pull.
