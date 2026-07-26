# Pre-registration — J-Space Ablation vs CoT Externalization

**Dated before any main experimental run.**
**Model:** `Qwen/Qwen3-4B` (36 layers; not Qwen3.5-4B).
**Datasets:** `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `HuggingFaceH4/aime_2024` (n=30).

## Top-level causal claim (§2.1)

As task difficulty increases under J-space ablation, CoT protection against ablation shrinks toward the direct-arm level. That is,

`CoT-protection ≡ (direct_clean − direct_ablated) − (cot_clean − cot_ablated)`

decreases as difficulty rises.

## Mechanism (§2.2) — conjectural

Proposed: J-space mediates **holding** (compensable by CoT writing intermediates to the page) and **selection** (not compensable by CoT). Selection being J-privileged is an **analogical extension** of the paper’s two-hop probe-decomposition / swap results (holding and using an unspoken intermediate), **not** a result the paper tested for per-step reasoning selection under CoT. This conjecture is the most likely place for the mechanism to fail independently of the top-level claim.

Only the backtracking-degradation signature is tested; local-step / global-strategy dissociation is out of scope.

## Primary operational hypothesis (§2.3)

- **H1:** CoT-protection is large and positive on easy problems and shrinks toward zero as difficulty rises (difficulty × condition × ablation interaction with this sign).
- **H0:** CoT-protection is stable across difficulty (no interaction).
- **H1′:** opposite sign — CoT-protection grows with difficulty.

Confirming H1 does **not** confirm the mechanism.

**Primary difficulty axis:** MATH-500 levels 1–5. GSM8K = easy anchor; AIME 2024 = hard anchor. Present side-by-side, not collapsed into one axis.

## Mechanism operational hypothesis (§2.4)

Effective backtracking rate drops under J-ablation more than under random-direction ablation, and the excess drop grows with difficulty.

## Disconfirmation conditions (§2.5)

- H1 is **disconfirmed** if the difficulty × condition × ablation interaction is null or reversed once floored cells are excluded.
- The **mechanism** is disconfirmed if the interaction holds but backtracking shows no J-ablation-specific drop beyond random ablation.
- A result attributable to **broad degradation** (random-direction and/or size-control shape match) rather than J-space-specific effects disconfirms the mechanism regardless of the headline interaction.

## Floor handling (§4.2)

Exclude cells where an arm’s unablated accuracy is **< 10%** from the interaction estimate; report them separately. Graded log-prob of the correct answer is logged from the start.

## Deliberate design deviations / structural limits

1. **Single-band ablation strength (§3.A.5):** one fixed medium-equivalent workspace band for all main runs, not the paper’s light/medium/heavy strength sweep. Stated as a deliberate scope cut.
2. **AIME n ≤ 30 (§4.10):** `HuggingFaceH4/aime_2024` has exactly 30 rows. The hard anchor is both most floor-prone and smallest-n; bootstrap CIs will be widest where H1 is strongest. Disclosed, not “fixed” by silently mixing years.

## Ablation defaults (locked)

- Ablate prompt + generation positions across the band.
- Per-token ablation in **both** CoT and direct arms with per-step clean top-10 exclusion.
- Direction selection: top-k activated **J-lens token vectors** by lens logit (not SVD of \(J\)).
- Band on Qwen3-4B: **[27, 33]** (manual lock after n=64 late-ramp diagnostic).
- Default k=10; sensitivity at k=5 and k=25.
- Random-direction control: matched band/k and matched ‖Δh‖ to the J-ablation
  perturbation at each layer/position (paper matched-norm); seeds {0,1,2}.
- Clean top-10 exclusion is **position-local** (per prefix index), not broadcast
  from the last token.
- Decoding: greedy, fixed seed, one trace per problem.
- Instrument gate: multihop exact-match on clean-correct items with J ≫ random
  (see `runs/2026-07-26_instrument-run5/`).
