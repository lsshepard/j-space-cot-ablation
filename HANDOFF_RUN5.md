# Handoff: `runpod-run-5` — instrument confirmed; gsm8k CoT smoke inconclusive

**Date:** 2026-07-26  
**Branch:** `runpod-run-5` (from `runpod-run-4`)  
**Model:** `Qwen/Qwen3-4B`, CUDA A100-80GB, bfloat16  
**HEAD:** see tip of branch (instrument + smoke artifacts committed)

## Bottom line for the next agent

1. **J-ablation instrument works** on multihop factual recall (exact-match), with a hard scope rule: analyze stratified by clean correctness.
2. **Do not start a full gsm8k grid yet** from this handoff alone. A n=6 CoT smoke test showed **zero accuracy separation** between clean / J / random CoT; J only lengthened traces and hit the token cap more.
3. **Exact match is the gate, not the gold-lp triad.** The triad inverts on items the model does not know.
4. Pod ops: serialize GPU jobs if `HF_HOME` is on the RunPod FUSE volume; expect ~50% GPU util from sync-bound ablation.

Push if not already: commits may sit only locally depending whether the last push landed.

---

## Instrument confirmation (do not re-litigate unless code changes)

### Exact-match gate, full 93-item Anthropic multihop set

Artifact: `results/gates/multihop_em_full93_b27-33_*`, CIs in `results/gates/em_gate_ci.json`.

Every arm run on every problem (no `--require-clean-correct` during the run). Stratify at analysis:

| Population | n | clean | J | random | random − J | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| clean **correct** | 34 | 1.000 | 0.529 | 0.765 | **+0.235** | **[+0.049, +0.422]** |
| clean **wrong** | 59 | 0.000 | 0.186 | 0.107 | −0.079 | [−0.198, +0.028] |
| all | 93 | 0.366 | 0.312 | 0.348 | +0.036 | [−0.068, +0.140] |

Mechanism: J removes what the model was about to say. Right → accuracy falls. Confidently wrong → wrong content deleted, correct token sometimes surfaces (J 0.186 vs random 0.107). Pooling cancels. Generic damage cannot track whether the model's own prediction was correct.

Primary band **[27, 33]**, k=10, exclude_topk=10. Band **[27, 31]** is robustness only; same pattern.

Clean-correct strata exclude zero on every EM gate run on this branch (easy-28 and full-93).

### Qualitative + lens agreement

- Trace: `results/diagnostics/ablation_token_traces.{json,md}` — semantic survivors, exclusion fires (~1.37/10 all-pos, ~2.96/10 last-pos). Run-4's committed trace was pre-fix (non-descending `coeff_abs`); deleted and regenerated.
- `tests/test_lens_agreement.py` — our `lens_logits_for_residual` matches `jlens.apply` top-10. `pytest -q`: 73 passed (incl. slow).
- Code: `AblationFactors.unembed_f32` cache; final RMSNorm dtype from norm weight (bitwise-identical readout).

### Gold-lp triad — secondary only

`results/diagnostics/triad_multihop93_b27-33/`, stratified via `scripts/13_triad_by_confidence.py`.

| Stratum | n | ΔJ−ΔR | CI |
| --- | --- | --- | --- |
| all 93 | 93 | −0.504 | [−1.409, +0.426] |
| clean_lp > −1 | 35 | +1.729 | [+0.457, +3.112] |
| not confident | 58 | −1.852 | [−3.081, −0.655] |

Logprob rewards flattening a wrong distribution (−12 → −8) without ever getting the answer right. EM on the same population is near null (−0.079). **Do not gate the experiment on pooled triad.**

gsm8k triad was started and abandoned (~29/64): mean clean gold lp ≈ −7.8, **0 confident items** — direct teacher-forced scoring cannot validate math CoT.

---

## gsm8k CoT smoke (just finished)

**Command:**

```bash
python -u scripts/06_run_grid.py --datasets gsm8k --limit 6 \
  --band-start 27 --band-end 33 --k 10 --random-seeds 0 \
  --max-new-tokens 768 --out-name grid_smoke_gsm8k
```

**Artifacts:** `results/grid_smoke_gsm8k/{traces.jsonl,run_meta.json}`, log `results/grid_smoke_gsm8k.log`.

| Condition | n | accuracy | mean len | cap hits |
| --- | --- | --- | --- | --- |
| direct_clean | 6 | 1.00 | 134 | 0 |
| direct_j_ablated | 6 | 1.00 | 116 | 0 |
| direct_random | 6 | 1.00 | 129 | 0 |
| cot_clean | 6 | 0.83 | 484 | 1 |
| cot_j_ablated | 6 | 0.83 | 638 | 4 |
| cot_random | 6 | 0.83 | 478 | 1 |

Per-problem CoT correctness is **identical** across clean / J / random: `110111`.  
Direct is `111111` on all three arms.

Ablation is live (lengths move; J CoT hits 768-token cap on 4/6 problems) but **does not change EM on this tiny easy sample**. Consistent with weak number-token loading in the band and/or ceiling effects on easy gsm8k.

**Implication:** a full gsm8k grid is high cost (~7–10 h at current util) with a real risk of a null EM result. Next pod should either (a) larger gsm8k pilot with length/cap-hit as co-primary, (b) harder problems (math500), or (c) confirm intermediates load under CoT before burning a night.

---

## Frozen ablation knobs (validated)

| Knob | Value |
| --- | --- |
| Selection | Top-k activated J-lens token vectors (not SVD of J) |
| Band | **[27, 33]** primary |
| k | 10 |
| exclude_topk | 10 |
| ablate_prompt_tokens | True |

---

## Code / scripts on this branch

| Path | Role |
| --- | --- |
| `src/jspace/ablation.py` | fp32 unembed cache; RMSNorm dtype fix |
| `tests/test_lens_agreement.py` | GPU lens-agreement check |
| `scripts/12_em_gate_ci.py` | Bootstrap CI + clean_correct / clean_wrong strata |
| `scripts/13_triad_by_confidence.py` | Stratify triad by clean_lp |

---

## Ops (this RunPod box)

- `HF_HOME` on `/workspace` (MooseFS/FUSE): concurrent HF-cache access can wedge processes in D-state. **One GPU job at a time**, or move cache to local disk.
- Ablation path is sync-bound (~50% util, ~160 W of 400 W). Full CoT grids will be slow until syncs are removed with a bitwise-equivalence check.
- `.venv` was recreated with `--system-site-packages` against system `torch 2.8.0+cu128`.

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16 PYTHONUNBUFFERED=1
```

---

## Suggested next steps (not run)

1. Push `runpod-run-5` if remote is behind.
2. Decide: larger gsm8k pilot vs math500 L1–L2 pilot before full grid.
3. If continuing gsm8k: report stratified clean-correct CoT EM **and** length / cap-hit rates; do not declare null from pooled easy EM alone.
4. Optional later: remove `.tolist()` / Gram–Schmidt host syncs after identity check.

## Reproduce instrument gate

```bash
pytest -q
python -u scripts/04_multihop_gate.py --fixture tests/fixtures/lens-eval-multihop.json \
  --limit 93 --band-start 27 --band-end 33 --k 10 --exclude-topk 10 \
  --max-new-tokens 32 --out-name multihop_em_full93_b27-33
python -u scripts/12_em_gate_ci.py
```
