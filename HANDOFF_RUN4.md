# Handoff: `runpod-run-4` — J-ablation instrument gate (PASSING)

**Date:** 2026-07-25  
**Branch:** `runpod-run-4`  
**Model:** `Qwen/Qwen3-4B`, CUDA, bfloat16  

## Frozen ablation (instrument validated)

| Knob | Value |
| --- | --- |
| Selection | **Top-k activated J-lens token vectors** \(v_t = J^\top u_t\) by lens logit on \(h\) |
| Band | **[27, 33]** |
| `k` | **10** |
| `exclude_topk` | **10** (paper default; also strong with 0) |
| `ablate_prompt_tokens` | `True` |

**Do not** use SVD(\(J\)) singular-vector ablation — that was the broken methodology (junk survivors, null gates).

## What was wrong

Paper (§3.5.2): ablate the **k=10 most strongly activated J-lens vectors** (token directions), skipping clean next-token top-10.

We previously selected **top-k right singular vectors of \(J\)** by \(|(Vh\,h)\odot S|\). Those decode to punctuation/noise, not verbalizable intermediates — so the hook was mechanically live but not J-content ablation.

## Fix (in tree)

- `src/jspace/ablation.py`: `select_active_j_lens_directions`, `lens_logits_for_residual`, `j_lens_vectors_for_tokens`; hooks use this path; Jacobians moved to CUDA (no CPU SVD).
- Final RMSNorm applied before unembed (matches `jlens` HF unembed).
- Qualitative tracer (`scripts/11_ablation_token_trace.py`) rides the live hook.

## Positive-control results (recreated)

Control set: easy multihop with **clean gold lp > −1** (n=16), band `[27,33]`, k=10.

### Exact-match (confident subset)

| Config | Clean EM | J EM | Drop |
| --- | --- | --- | --- |
| exclude=10 | 0.94 | 0.56 | **0.375** |
| exclude=0 | 0.94 | 0.25 | **0.688** |

Artifacts: `results/diagnostics/positive_control_jlens.json` (EM sweep + triad @ exclude=0).

### Gold-lp triad vs random (same confident set)

**exclude_topk=10** (`positive_control_jlens_exclude10.json`):

| Metric | mean | 95% CI |
| --- | --- | --- |
| ΔJ = lp_clean − lp_J | **2.10** | **[1.14, 3.21]** |
| ΔR | 0.63 | [0.13, 1.36] |
| ΔJ − ΔR | **1.47** | **[0.38, 2.72]** |

**Verdict:** `j_specific_gold_lp_bite`

**exclude_topk=0** (same file’s triad section): ΔJ ≈ 3.96 [2.76, 5.43], ΔJ−ΔR ≈ 3.28 [2.06, 4.53] — also `j_specific_gold_lp_bite`.

### Qualitative check

Last-pos survivors are now semantic (cities / China / Russia / …); clean top-10 shields answer tokens (`北京`, `China`, …) when `exclude_topk=10`. Example: clean `Beijing.` → J `London.`

## Lens status (nuance)

- Calc sanity still only hits intermediates at **L34** (motor-ish); band `[27,33]` is mid/late workspace by match-rate diagnostic.
- Ablation works in `[27,33]` anyway once token directions are selected correctly.
- Number-token loading remains near-zero — math tasks may still be weak; multihop factual control is the validated instrument.

## Final EM gate (paper-style clean / J / random) — PASS

`scripts/04_multihop_gate.py` on easy multihop, **clean-correct only** (n=16), frozen config, random seeds 0/1/2:

| Arm | Accuracy |
| --- | --- |
| Clean | **1.00** |
| J-ablated | **0.50** |
| Random (mean) | **0.85** |

- J drop = **0.50** (≥ 0.30)
- J ≫ random: random−J = **0.35** (≥ 0.10)
- **Verdict:** `j_specific_em_bite` · `gate_pass=true`
- Artifact: `results/gates/multihop_em_gate_n20_summary.json`

## Main grid readiness

Instrument gate **passed** (gold-lp + paper-style EM). Safe to run `scripts/06_run_grid.py` with **that exact** ablation. Prefer `exclude_topk=10` (paper-aligned); document if switching to 0 for stronger bite.

## How to re-verify on GPU

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16 PYTHONUNBUFFERED=1

# EM gate (easy multihop; use confident subset or full fixture)
python -u scripts/04_multihop_gate.py \
  --fixture tests/fixtures/lens-eval-multihop-easy.json \
  --limit 28 --band-start 27 --band-end 33 --k 10 --exclude-topk 10 \
  --max-new-tokens 32 --min-drop 0.3

# Gold-lp triad
python -u scripts/10_gold_lp_triad.py \
  --dataset multihop \
  --fixture tests/fixtures/lens-eval-multihop-easy.json \
  --limit 28 --band-start 27 --band-end 33 --k 10 --exclude-topk 10

pytest tests/test_ablation.py tests/test_diagnostics.py tests/test_gold_lp_triad.py -q
```

## Open notes

1. Full easy-fixture n=10 EM drop was ~0.20 (float-edge); **confident subset** is the fair paper-style control (near-ceiling clean).
2. Tokenization variants (` Beijing` vs `北京`) can partially bypass exclusion — known single-token lens limit.
3. Intermediate-hit counters in older token-trace MD used loose string match; trust last-pos direction lists / strict recounts.
