# Handoff: `runpod-run-5` — J-ablation instrument verification (CONFIRMED, with scope)

**Date:** 2026-07-26
**Branch:** `runpod-run-5` (based on `runpod-run-4`)
**Model:** `Qwen/Qwen3-4B`, CUDA (A100-80GB), bfloat16

**Verdict: the instrument works.** On problems the model can actually answer,
J-ablation destroys the answer significantly more than a norm-matched random
control: **random − J = +0.235, 95% CI [+0.049, +0.422], n=34**. This holds on
every gate run and both bands.

**Scope condition, and it is not optional:** the effect is only measurable on
clean-correct problems. Pooled over all problems the number is +0.036
[−0.068, +0.140]. That is not a weaker version of the same measurement — it is
two opposite effects cancelling, and the grid must be analyzed stratified by
clean correctness or it will report a null that means nothing.

## What J-ablation actually does

The two strata move in opposite directions because they are the same mechanism.
J-ablation removes the content the model was about to emit.

| Population | n | clean | J | random | random − J | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| clean **correct** | 34 | 1.000 | 0.529 | 0.765 | **+0.235** | **[+0.049, +0.422]** |
| clean **wrong** | 59 | 0.000 | 0.186 | 0.107 | −0.079 | [−0.198, +0.028] |
| all | 93 | 0.366 | 0.312 | 0.348 | +0.036 | [−0.068, +0.140] |

Where the model is right, deleting its top J-lens directions removes the
correct answer and accuracy falls. Where the model is confidently wrong, those
same directions carry the *wrong* answer, so deleting them lets the correct
token surface and accuracy rises slightly (J 0.186 vs random 0.107). The pooled
figure is just the weighted sum: 0.235·(34/93) − 0.079·(59/93) = +0.036, which
matches the observed value exactly.

Generic damage cannot produce this pattern. A control that merely degrades the
residual hurts both strata. That the sign tracks whether the model's own
prediction was correct is the strongest evidence here that the ablation is
content-specific.

## Evidence

`results/gates/em_gate_ci.json`, from `scripts/12_em_gate_ci.py` (paired
bootstrap over problems, 10000 resamples, seed 0).

| Run | Band | pooled n | pooled random−J | clean-correct n | clean-correct random−J |
| --- | --- | --- | --- | --- | --- |
| `full93` | [27,33] | 93 | +0.036 [−0.068, +0.140] | 34 | **+0.235 [+0.049, +0.422]** |
| `full` | [27,33] | 28 | +0.143 [−0.036, +0.321] | 15 | **+0.289 [+0.044, +0.556]** |
| `full` | [27,31] | 28 | +0.155 [−0.000, +0.321] | 15 | **+0.311 [+0.111, +0.533]** |
| `clean` | [27,33] | 15 | +0.289 [+0.044, +0.556] | 15 | +0.289 [+0.044, +0.556] |
| `clean` | [27,31] | 15 | +0.311 [+0.111, +0.533] | 15 | +0.311 [+0.111, +0.533] |

The clean-correct stratum excludes zero in all five runs and on both bands.
`[27,33]` remains primary; `[27,31]` is the pre-declared robustness check and is
not promoted despite scoring marginally higher.

Selecting on clean correctness does not bias this contrast. Selection uses only
clean-arm information, and J and random are then scored on the identical item
set, so any regression-to-the-mean from selecting winners hits both arms and
cancels in the difference. What would be circular — selecting on J's outcome,
or dropping items where J did poorly — is not done anywhere.

### Qualitative trace (`results/diagnostics/ablation_token_traces.{json,md}`)

Last-position survivors are semantic. For `super-populous-capital`: ` Москва`,
` Beijing`, ` Mumbai`, ` Tokyo`, `杭州`, `伦敦`, `Putin`. Excluded directions are
the answer tokens the clean top-10 shields (`北京`, `London`, `China`).
Exclusion fires at 1.37/10 directions across all positions and 2.96/10 at the
prediction site, against 0.03/10 in the run-4 artifact. `Beijing.` → `London.`

### Lens readout

`tests/test_lens_agreement.py` asserts `lens_logits_for_residual` matches
`jlens`' own `lens.apply` on the top-10 across band layers and prompt
positions. Confirmed to have teeth: substituting a neighbouring layer's
Jacobian breaks it at every site. `pytest -q`: 73 passed.

## Do not use the gold-lp triad as the instrument gate

The teacher-forced gold-logprob triad reports the *opposite* conclusion and it
is the metric that is wrong, not the ablation.

| Population | n | ΔJ − ΔR | 95% CI |
| --- | --- | --- | --- |
| all 93 | 93 | −0.504 | [−1.409, +0.426] |
| confident (`clean_lp > −1`) | 35 | +1.729 | [+0.457, +3.112] |
| not confident | 58 | **−1.852** | **[−3.081, −0.655]** |

On problems the model cannot answer, J-ablation lifts gold logprob from roughly
−12 to −8 by flattening the distribution. That is a large, *significant* move
in logprob space that never crosses into a correct answer. Exact-match grading
is nearly immune to it: the same population reads −0.079 under EM versus −1.852
under logprob. Logprob rewards a non-event; EM does not.

Both metrics agree on the confident stratum (+1.729 logprob, +0.235 EM), so the
instrument conclusion is unchanged. But **EM is the gate**, and any future
triad number must be read stratified (`scripts/13_triad_by_confidence.py`).

## Corrections to earlier handoffs

**Run-4 reported subset numbers as full-set results.** Its ΔJ−ΔR = 1.47
[0.38, 2.72] was the n=16 confident subset, unlabelled. Reproduced here as
+1.274 [+0.265, +2.479]. Its EM figures (clean 1.00 / J 0.50 / random 0.85)
were the clean-correct subset.

**Run-4's committed evidence was pre-fix.** `ablation_token_traces.json` carried
`coeff_abs` in non-descending order (6.78, 10.11, 13.23, 8.82, ...), impossible
from the `torch.topk` in `select_active_j_lens_directions`. All of
`results/diagnostics/` and `results/gates/` was deleted in `59a0e65` and
regenerated. The new trace has 0/70 blocks out of order.

**An earlier draft of this document called the instrument unconfirmed.** It
weighted the pooled gold-lp triad as the deciding number. That was wrong for
the reason above.

## Provenance

All artifacts on this branch were produced by code at `e6f20d1` or later, which
is run-4's `3582938` plus the two Task 0 changes. Nothing predates the fix.

1. **`AblationFactors.unembed_f32`** — `lens_logits_for_residual` and
   `j_lens_vectors_for_tokens` cast the 151936×2560 bf16 `lm_head` to fp32 on
   every call, rebuilding a ~1.5 GB copy twice per (band layer, position). Now
   materialized once. The final RMSNorm takes its input dtype from its own
   weight rather than from the unembed, so passing the fp32 copy leaves the
   readout **bitwise identical** (verified, max abs difference 0.0).
2. **`tests/test_lens_agreement.py`** — described above.

Analysis scripts added: `12_em_gate_ci.py` (the gate script reports
`j_over_random` without an interval) and `13_triad_by_confidence.py`.

## Before running the grid

1. **Stratify by clean correctness.** Run every arm on every problem — do not
   gate the run itself — then split at analysis time. A pooled accuracy number
   over a set the model half-fails is uninterpretable.
2. **gsm8k is unvalidated.** Everything above is single-forward factual recall.
   A gsm8k gold-lp triad was started and abandoned at 29/64 because it cannot
   work: direct-answer gold logprob averaged −7.79 with **0 of 29 items
   confident**, since Qwen3-4B cannot solve a word problem in one forward pass.
   The open question is whether J-ablation perturbs *CoT generation*, which no
   run here tests. Worth a short generation smoke test before committing to a
   full grid.
3. **Serialize GPU jobs.** `HF_HOME` is on `/workspace`, a RunPod MooseFS/FUSE
   network volume. Two processes touching the HF cache concurrently both wedged
   in uninterruptible disk sleep (`folio_wait_bit_common`) with the GPU at 0%;
   killing one released the other immediately. Run one job at a time, or move
   the cache to local disk (`/` has 69 GB free).
4. **Expect ~50% GPU utilization.** At 161 W of 400 W and 20% memory-controller
   activity, the ablation path is launch- and sync-bound, not compute-bound:
   `select_active_j_lens_directions` does two `.tolist()` device syncs and
   `gram_schmidt` one `float(norm)` host read per direction, roughly a dozen
   syncs per (band layer, position), with no batching across positions. Around
   540 hook calls and ~6500 syncs per 32-token generation. gsm8k CoT runs ~900
   tokens, so this overhead sets the grid's cost. Removing the syncs looks
   worthwhile but must be proven bitwise-identical first — direction selection
   is discrete and any drift silently changes which directions are ablated.

## Reproduce

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16 PYTHONUNBUFFERED=1

pytest -q                                   # 73 passed, includes the lens-agreement check

python -u scripts/11_ablation_token_trace.py --band-start 27 --band-end 33 --k 10 --exclude-topk 10

# the powered gate: every arm on every problem, stratified afterwards
python -u scripts/04_multihop_gate.py --fixture tests/fixtures/lens-eval-multihop.json \
  --limit 93 --band-start 27 --band-end 33 --k 10 --exclude-topk 10 \
  --max-new-tokens 32 --out-name multihop_em_full93_b27-33

python -u scripts/12_em_gate_ci.py
python -u scripts/13_triad_by_confidence.py results/diagnostics/triad_*/gold_lp_triad_multihop.json
```

## Environment note

The `.venv` referenced in the run-4 handoff did not exist. Recreated with
`python -m venv --system-site-packages .venv` against the preinstalled
`torch 2.8.0+cu128`, then `pip install -e ".[dev]"`. `jlens` resolves to
`github.com/anthropics/jacobian-lens@581d398`, lens weights to
`neuronpedia/jacobian-lens` snapshot `a4114d7`.
