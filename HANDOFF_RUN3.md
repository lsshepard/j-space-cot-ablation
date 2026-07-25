# Handoff: `runpod-run-3` — J-ablation instrument gate

**Date:** 2026-07-25  
**Branch:** `runpod-run-3` (from `runpod-run-2`)  
**Model:** `Qwen/Qwen3-4B`, CUDA, bfloat16  
**Frozen ablation under test:** band **[27, 33]**, `k=10`, `exclude_topk=10`, `ablate_prompt_tokens=True`

## Goal

Prove (or falsify) that the **same** J-space ablation used in the main grid is a working instrument: it must produce a **J-specific** behavioral hit vs matched random-direction control. Until that passes, main-experiment nulls are uninterpretable.

## What we built this session

| Piece | Path |
| --- | --- |
| Ablation step diagnostics (`n_active`, `n_survivors`, `‖Δh‖`) | `src/jspace/ablation.py`, wired through `generate.py` |
| Teacher-forced gold LP with clean top-k exclusion + reusable factors | `src/jspace/graded.py` |
| Instrument probes 1–3 (survival, exclude A/B, logprob bite) | `src/jspace/diagnostics/`, `scripts/09_instrument_probes.py` |
| Clean / J / random **gold-lp triad** + bootstrap CIs | `src/jspace/diagnostics/gold_lp_triad.py`, `scripts/10_gold_lp_triad.py` |
| Clean-accuracy screener (find 20–80% sets) | `scripts/10a_screen_clean_acc.py` |
| Easy multihop fixture | `tests/fixtures/lens-eval-multihop-easy.json` |
| Unit tests | `tests/test_diagnostics.py`, `tests/test_gold_lp_triad.py` |

Artifacts (force-added under `results/diagnostics/`):

- `instrument_probes_1_3.json` + `09_instrument.log`
- `gold_lp_triad_multihop.json` + `10_gold_lp_triad_easy.log`
- `screen_clean_easy_multihop.json`, `screen_clean_gsm8k.json`

## Key empirical findings

### 1. Hook is mechanically live (not toothless)

On easy multihop, band [27,33], probe 1:

- mean survivors with exclusion ≈ **9.93 / 10** (exclusion barely filters)
- mean ‖Δh‖ ≈ **69** (large residual edits)
- Verdict: `mechanically_active`

So “exclusion ate the ablation” is **not** the main failure mode.

### 2. Exact-match / EM gate still fails

Easy multihop n=10, band [27,33]: clean acc **0.60**, J-ablated **0.70**, drop **−0.10**.  
`exclude_topk=0` did not produce a large EM collapse either (`no_em_drop_either_way`).

Greedy answer strings often stay put even when soft probs move (e.g. Euro).

### 3. Gold-lp triad on easy multihop (n=28) — not J-specific

Clean direct accuracy on this set: **54%** (in the desired 20–80% band).

| Metric | mean | 95% bootstrap CI |
| --- | --- | --- |
| ΔJ = lp_clean − lp_J | −0.11 | [−0.92, +0.74] |
| ΔR = lp_clean − lp_R | −0.05 | [−0.49, +0.42] |
| ΔJ − ΔR | −0.05 | [−0.58, +0.53] |

**Verdict:** `no_reliable_j_gold_lp_bite` (all CIs include 0).

On **confident** items only (clean gold lp > −1, n=16): both J and random lower P(gold) (CIs > 0), but **J is not above random**.

### 4. Dataset screening

| Set | Clean direct acc | Usable for gold-lp gate? |
| --- | --- | --- |
| easy multihop | **54%** | Yes — completion format; gold LP meaningful |
| hard Anthropic nhop (first ~15) | low clean gold lp (~−6) | No — model doesn’t know answers; J often “helps” |
| GSM8K direct | **97%** | No — ceilinged; gold LP ~−8 even when correct (writes steps before `####`) |

### 5. Band diagnostic context

`results/calibration/band.json` match rates (top-k vs next token) rise sharply late:

- L32 ≈ 0.64, L33 ≈ 0.84, **L34 ≈ 0.98**

Current working band includes through **33**. Extending to **34** may create a larger effect but is near pure next-token territory — **random control becomes load-bearing**, not optional.

## Working hypotheses for the null

1. **Task often doesn’t need J intermediates** — many easy items look like shallow associations; clean isn’t paper-style near-ceiling two-hop.
2. **Metric scores answer mass, not intermediate binding** — teacher-forced P(gold \| prompt) can stay soft-null while workspace hops are damaged.
3. **Late band → generic disruption** — J ≈ random on confident items fits matched-norm damage more than J-content removal.
4. **Model/lens mismatch** — Neuronpedia Qwen3-4B/wikitext lens vs paper’s Claude results; large ‖Δh‖ without selective behavior.
5. **Exclusion denoising** — possible on low-clean items (J raises P(gold)), but survivors≈10 argues against “whitelist ate the gate” as the main story.

## What “pass” must look like before main grid

Freeze one config `(band_start, band_end, k, exclude_topk)` and require on a control set:

1. Clean not floored / not ceilinged.
2. Mean ΔJ > 0 with bootstrap CI excluding 0.
3. Mean (ΔJ − ΔR) > 0 with CI excluding 0 (J ≫ random, 2–3 seeds).

Only then run `06_run_grid.py` with **that exact** ablation.

## Recommended next steps (in order)

**Path A — prove current ablation (preferred first)**

1. Subset where clean gold lp is high **and** J-lens shows the known intermediate in-band.
2. Re-run `scripts/10_gold_lp_triad.py` on that subset, band `[27,33]`, exclude=10.
3. If still null → Path B.

**Path B — change one ablation knob, then re-prove**

Try one change at a time (same triad vs random):

1. `exclude_topk=0`, or
2. band `[27,34]` / `[30,34]` (heavier / later — random mandatory), or
3. larger `k`.

Whatever passes is the **only** ablation for the full study; document deviation from the paper.

**Do not** change task and ablation simultaneously.

## How to resume on GPU

```bash
source .venv/bin/activate
export JSPACE_MODEL=Qwen/Qwen3-4B JSPACE_DEVICE=cuda JSPACE_DTYPE=bfloat16 PYTHONUNBUFFERED=1

# Screen a candidate set (20–80% clean direct)
python -u scripts/10a_screen_clean_acc.py --dataset easy_multihop --limit 28 --max-new-tokens 64

# Gold-lp triad (clean / J / random)
python -u scripts/10_gold_lp_triad.py \
  --dataset multihop \
  --fixture tests/fixtures/lens-eval-multihop-easy.json \
  --limit 28 \
  --band-start 27 --band-end 33 \
  --k 10 --exclude-topk 10 \
  --random-seeds 0,1,2

# Older instrument suite (survival / exclude A/B / bite)
python -u scripts/09_instrument_probes.py \
  --limit 10 --band-start 27 --band-end 33 --k 10 --exclude-topk 10 --max-new-tokens 64

pytest tests/test_diagnostics.py tests/test_gold_lp_triad.py tests/test_graded.py -q
```

## Open decision for the next agent

User interest: try including **layer 34**, but only accept if **J ≫ random**. That is a Path B band change — run the triad at `[27,34]` before freezing anything into the main grid.
