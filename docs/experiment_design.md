# Functional Spec — J-Space Ablation vs. Chain-of-Thought Externalization

**Course:** Harvard CS 2881R, Homework Zero (Fall 2026)
**Status:** functional spec for the implementing agent. This describes _what_ to build and _why_, not _how_ to architect it. Make reasonable architectural/implementation choices; where a choice is scientifically load-bearing it is flagged **[LOAD-BEARING]** and must not be defaulted silently.

---

## 1. High-level objective

Investigate the internal↔external reasoning trade-off described in Anthropic's _Verbalizable Representations Form a Global Workspace in Language Models_ (Transformer Circuits, 2026), and test whether it **holds as problem difficulty increases**.

The paper shows that ablating the J-space (a small, verbalizable "global workspace" subframe of the residual stream) collapses internal multi-step reasoning but leaves automatic processing intact — **except** that GSM8K solved with explicit chain-of-thought (CoT) is far more robust to ablation than the same problems solved directly. The interpretation: CoT lets the model _externalize onto the page_ what it would otherwise hold in its internal workspace.

We ask: does that protection persist as problems get harder, or does it break down?

**Model (fixed):** `Qwen/Qwen3-4B` (the original Qwen3, 36 layers — **not** Qwen3.5-4B).
**Datasets (all three required, they are the difficulty axis):** GSM8K, MATH-500, AIME.

---

## 2. Hypotheses

Record all of this in a **pre-registration file, committed/dated BEFORE any main experimental run** (see §6). The pre-registration is itself a graded deliverable; its value is being timestamped before results exist.

### 2.1 Top-level hypothesis (bare causal claim)

As task difficulty increases (under J-space ablation), the CoT protection against ablation shrinks toward the direct-arm level. That is, the quantity

`(direct_clean − direct_ablated) − (cot_clean − cot_ablated)`

decreases as difficulty rises. This is a claim about a cause (rising difficulty) and an effect (shrinking CoT protection). It asserts _nothing_ about why. It can be true while the mechanism in §2.5 is false.

### 2.2 Mechanism (separate, lower-level — can fail independently of §2.1)

Proposed explanation for §2.1. The J-space mediates two functions; CoT can compensate for one but not the other:

- **Holding** — retaining an intermediate across steps. J-space does this internally (fragile, re-derived each forward pass, evicted by new context). CoT substitutes for it _perfectly_: writing an intermediate to the page makes it losslessly persistent and re-readable via attention. **Compensable by CoT.**
- **Selection** — the reasoning that arrives at _what to do next_. We conjecture this is an act of _verbal report_ and therefore privileged to the J-space. **Caveat on the evidence:** the paper's probe-decomposition numbers we lean on (non-J-space component of an intermediate flips answers on ~28% of trials vs ~61% for the J-space component, dropping to ~6% when J-space coordinates are clamped) come from a **two-hop factual-retrieval paradigm** (infer an unspoken intermediate like spider→ant and swap it) — i.e. they test _holding and using an intermediate under swap_, **not** "selecting the next reasoning step." Our holding-vs-selection split, and the claim that selection specifically is J-privileged and CoT-incompressible, is our own **analogical extension** of that result, not something the paper tested. It is the load-bearing conjecture of the mechanism and the most likely place for §2.2 to be wrong. The rhyme-planning and bandit results show _planning/strategy content_ is J-mediated and swap-manipulable, which is suggestive support, but still not a direct test of per-step selection under CoT. **Not compensable by CoT** (conjectured): the page can _store_ past selections but cannot _perform_ the current one, because text is the record of a selection, not the procedure that makes it.

**Why this predicts §2.1:** the per-step selection load is not constant. On GSM8K the next step is dictated by problem structure ("what next" is near-trivial), so load is almost all holding → CoT robust. On harder problems the arithmetic is trivial and "what next" _is_ the difficulty → selection load rises → the page cannot absorb it → CoT protection shrinks.

**Predicted observable signatures of the mechanism:** (i) locally coherent steps, (ii) globally flawed/absent strategy, (iii) backtracking degradation. **Only (iii) is tested** (§2.4, §3.D); (i) and (ii) are out of scope.

### 2.3 Primary operational hypothesis (tests §2.1)

Define **CoT-protection** ≡ `(direct_clean − direct_ablated) − (cot_clean − cot_ablated)` — how much _more_ ablation damages the direct arm than the CoT arm (i.e. how much CoT shields against ablation). This is the §2.1 quantity. Avoid the bare word "gap" — it ambiguously names either this between-arm quantity or a within-arm clean−ablated drop.

**H1:** CoT-protection is large and positive on easy problems and **shrinks toward zero as difficulty rises** (the within-CoT-arm clean−ablated drop grows _toward_ the direct arm's drop). Statistically: a difficulty × condition × ablation interaction with this sign.

**H0 (null):** CoT-protection is **stable** across difficulty (no interaction).

**H1′ (rival direction, not a separate test — just the opposite sign):** CoT-protection **grows** with difficulty (CoT compensates increasingly). A reversed-sign interaction supports this over H1.

Note: H1/H0 test the bare causal claim (§2.1) only. Confirming H1 does **not** confirm the mechanism (§2.2) — the gap could shrink for reasons unrelated to selection/holding (e.g. broad degradation, floor artifacts). The controls (§4.3) and the backtracking test (§2.4) are what connect a confirmed H1 to the mechanism.

### 2.4 Mechanism operational hypothesis (tests §2.2, can fail even if H1 holds)

**Backtracking collapse:** effective backtracking rate drops under J-ablation _more than under random-direction ablation_, and the excess drop grows with difficulty. Backtracking is the least decomposable selection operation (judge current approach wrong → select alternative), so it is the sharpest predicted casualty. This is the _only_ mechanism signature tested; the step/trace dissociation is out of scope.

### 2.5 Pre-committed disconfirmation conditions **[LOAD-BEARING]**

Write these down before running. Example form (finalize exact thresholds during calibration):

- H1 is **disconfirmed** if the difficulty × condition × ablation interaction is null or reversed (gap stable or widening) once floored cells are excluded.
- The **mechanism** is disconfirmed if the interaction holds but backtracking shows no J-ablation-specific drop beyond random ablation.
- A result attributable to **broad degradation** (see controls) rather than J-space-specific effects counts as disconfirming the mechanism regardless of the headline interaction.

---

## 3. High-level experiments

Run in this order. Experiments A and B are **gates**: their failure means downstream nulls are uninterpretable. Do not proceed past a failed gate without fixing it.

### A. Calibration / replication (gate: "the lens works")

Reproduce enough of the paper's method on Qwen3-4B to trust the lens and fix parameters:

1. Load the pre-fitted Jacobian lens (see §5.1). Confirm it loads and applies.
2. **Lens sanity check:** reproduce the arithmetic-intermediate readout, e.g. `calc: ( 4 + 17 ) * 2 + 7 =` — intermediate values (21, then 42, then 49) should surface in the J-lens at successively later layers. If they don't, the lens or its application is wrong; stop and fix.
3. **Band identification:** compute the cheap layer diagnostic (J-lens top-k next-token match by layer: near-zero early, rising through the workspace band, spiking at the final layers). Identify the workspace band start/end **in actual Qwen3-4B layer indices** (translate the paper's 0–100 reindexing to 36 layers once, up front). **[LOAD-BEARING]** — ablating the wrong band gives either a null or generic damage. Implementation: the band-finder should **inspect the diagnostic and auto-select a reasonable band**, but **expose the chosen band (start/end layers) as an explicit, overridable parameter** so it can be reviewed and adjusted after seeing the plot. Always emit the diagnostic plot alongside the auto-selected band — do not hide the selection behind a silent threshold. Note: the small local dev model may show no clean band; the authoritative band comes from the 4B GPU run, so treat any locally-derived band as throwaway.
4. **k selection:** pick the ablation sparsity k. Default to the paper's k=10; sensitivity-check at k=5 and k=25.
5. **Ablation strength — deliberate scope cut, state it.** The paper's flagship ablation result (Fig. 22/24) is _not_ a single ablation — it's a **sweep over ablation strength** (light/medium/heavy, defined by how much of the layer range is included), with the random-direction control run only at _medium_ strength. This spec picks **one fixed band** and treats it as the ablation throughout. That is an acceptable simplification for scope, but it must be **stated explicitly in the report as a deliberate deviation** ("we use a single medium-equivalent band rather than the paper's strength sweep"), because a reviewer familiar with the paper will immediately ask "which strength?" If time permits, running even two strengths (medium + heavy) materially strengthens the result and maps onto the paper's design.
6. **Number-token loading check [LOAD-BEARING]:** measure workspace loading (cosine sim of residual stream with number-token J-lens vectors) on actual math prompts. If numbers are near-absent from the J-space, the ablation may be toothless on math — this must be known before the main runs and reported in Limitations either way. Note Qwen3 tokenization of multi-digit numbers (the lens is single-token by construction).

### B. Positive control (gate: "the ablation does what it should")

Reproduce the paper's multi-hop reasoning ablation result: on a set of ~50 two-hop factual prompts with unspoken intermediates, unablated accuracy should be near-ceiling and J-ablation should drop it toward zero. **If ablation does not tank multi-hop reasoning, the hook/band/k is wrong and every downstream null is meaningless.** This is the instrument-works gate.

### C. Main experiment — the interaction

For each **difficulty cell** (see §4.1) run six conditions per problem:

| #   | Condition             | Thinking | Ablation                         |
| --- | --------------------- | -------- | -------------------------------- |
| 1   | Direct clean          | off      | none                             |
| 2   | CoT clean             | on       | none                             |
| 3   | Direct J-ablated      | off      | J-space, throughout              |
| 4   | CoT J-ablated         | on       | J-space, throughout              |
| 5   | Direct random-control | off      | random-direction, matched band/k |
| 6   | CoT random-control    | on       | random-direction, matched band/k |

Plus a **size control**: run conditions 1–2 (clean, both modes) on a **smaller unablated model** (`Qwen3-1.7B`, optionally `Qwen3-0.6B`) for every cell. This is the "is J-ablation just shrinking the model?" control — see §4.3.

Primary readout: does the clean−ablated accuracy gap differ between CoT and direct, and does that difference change with difficulty (the three-way interaction)? Random-control arms establish the effect is J-space-specific, not generic to removing k directions.

### D. Mechanism — backtracking

For the three CoT conditions (clean / J-ablated / random-ablated), count **effective backtracks** per trace (see §4.4), normalized per trace length, as a function of difficulty. The mechanism-supporting pattern: J-ablated backtracking drops below random-ablated, increasingly with difficulty.

---

## 4. Structured design details

### 4.1 Difficulty axis **[LOAD-BEARING]**

- **Primary interaction test:** MATH-500 **levels 1–5** — same dataset, format, scoring; only difficulty varies. This is the cleanest gradient and carries the main argument.
- **Anchors:** GSM8K = easy anchor and point of contact with the paper's result; AIME = hard extreme.
- Present all as **comparable side-by-side panels** (not collapsed into one axis). Dataset-to-dataset differences confound difficulty with format/domain/tokenization; the within-MATH-levels curve is the clean evidence.

### 4.2 Floor handling **[LOAD-BEARING]**

The dangerous floor is in the **unablated baseline**, not in ablated-going-to-zero. If unablated-direct accuracy on a cell is already near-zero, measured degradation is forced to ~0 there and _mimics H1 confirmation_ (gap collapses for a trivial reason).

- Log **per-cell unablated accuracy** and inspect that column first.
- Pre-commit a threshold (e.g. exclude cells where an arm's unablated accuracy < 10% from the interaction estimate; report them separately).
- Carry a **graded metric from the start** — log-probability of the correct final answer (or a continuous partial-credit score) — so degradation is still measurable once binary accuracy pins. Adding this after discovering a floor means re-running everything.
- Expectation: GSM8K and low MATH levels are non-floored; AIME direct is likely floored (use CoT-only / graded metric there).

### 4.3 Controls

- **Random-direction ablation** (conditions 5–6): at each position, project out k random directions over the matched band, matched norm. Rules out "any k directions hurt." Use **2–3 seeds** so "random didn't hurt" isn't one unlucky draw.
- **Size control** (smaller unablated model): the argument is **shape, not level**. "Ablation just makes it dumber" predicts _flat, proportional_ degradation across tasks. H1 predicts _selective_ degradation (automatic/easy spared, flexible/hard destroyed). Compare the _per-task/per-difficulty profile_ of J-ablated-4B against smaller-unablated-model; the smaller model is the template for "flat and proportional." This is why per-cell breakdown matters — an averaged number can't show "selective."

### 4.4 Backtracking metric

- **Effective backtrack [LOAD-BEARING]:** a discourse marker of reversal ("wait," "actually," "let me reconsider," "alternatively," "that's wrong," etc.) **followed by an actual change in approach**. Marker-only counts do not qualify — ablation can inflate degenerate "wait, wait" loops.
- Freeze the definition as a **judge rubric in a committed `.md` file BEFORE looking at result data.**
- Method: regex for the headline number + **LLM-judge on a sample** to validate. Report **regex↔judge agreement rate** — that agreement _is_ the validity evidence.
- Normalize per trace length (ablated traces run longer; raw counts rise mechanically).
- **Judge implementation (scope-appropriate).** A full frontier-API judge (temp 0, pinned version) is the gold standard but overkill for this assignment. Instead, invoke an **LLM subagent per candidate span with an identical, frozen prompt and rubric across all examples** — consistency across items is the property that matters, and a fixed prompt provides it, so the subagent is a legitimate judge. To keep it defensible despite weaker reproducibility than a pinned API call: (a) the judge operates on **spans, not whole traces**, so multiple backtracks in one trace are counted individually; (b) it emits **structured output** (`{is_backtrack: bool, approach_before: str, approach_after: str}`) so the "effective" criterion — reversal marker _plus a real approach change_ — is enforced by schema rather than by prose vibes; (c) **all verdicts are saved to a file in the repo**, shifting the reproducibility claim from "re-run for identical labels" to "here are the actual labels, audit them"; and (d) a **small hand-labeled sample (~30–50 spans)** is checked against the judge and the agreement rate reported, alongside the regex↔judge agreement. This combination (frozen rubric + per-span structured judgments + saved auditable verdicts + a hand-checked calibration sample) is an acceptable scope tradeoff for a short assignment and reports its own reliability.

### 4.5 "Direct" enforcement **[LOAD-BEARING]**

"Direct" must genuinely suppress reasoning, or the CoT/direct contrast (the axis the hypothesis rides on) is contaminated.

- Primary: `tokenizer.apply_chat_template(..., enable_thinking=False)`.
- Verify, don't trust: log direct-arm token counts and check answer bodies for step-by-step content. Flag/exclude leaked traces. A length threshold catches most leakage.

### 4.6 Ablation mechanics

- **Ablate throughout generation, in BOTH arms** (not answer-only): the hook fires at every generated token position across the band, in the direct arm as well as the CoT arm. The direct arm is shorter but still autoregressive — do NOT leave direct-mode generation as a single forward pass; conditions 3 and 5 require the same per-token ablation as conditions 4 and 6. Rationale: selection happens at every generated token in either mode. (Answer-only ablation is a _different_ experiment testing pure retrieval-from-page — out of scope unless time permits, in which case the difference between the two decomposes holding vs. selection and is a bonus result.)
- **Positions:** ablate generation positions across the band. Decide and state whether prompt tokens are also ablated. **[decide, then fix]**
- **Clean-pass top-10 exclusion [LOAD-BEARING]:** to target internal reasoning rather than report, do not ablate any token that appears in the clean forward pass's top-10 at that position. During generation this MUST be computed **per-step on the ablated run's own prefix**: at each generated position, run one clean forward pass (no ablation) on the current prefix to get that position's top-10, then run the ablated pass to produce the token. This is **~2× forward passes per generated token** — accept that cost. Do NOT pre-generate a full clean trace and replay its top-10s: the ablated trace diverges after the first differing token and the alignment becomes meaningless (no error is raised — it silently produces garbage exclusions).

### 4.7 Sampling & termination

- **Decoding:** greedy, fixed seed, one trace per problem. Uncertainty comes from **bootstrap over the problem set**, not within-problem sampling. This is a legitimate simplification; the homework asks for uncertainty to be _shown_, not for any specific scheme. The real limiter on trustworthiness is **problems-per-cell**, not traces-per-problem.
- **Token budget [LOAD-BEARING]:** set `max_new_tokens` sized from _unablated_ trace lengths (measure in calibration, set a generous multiple — e.g. a few K for MATH, more for AIME), well under the 32K context window. A hung ablated trace will otherwise generate to context limit and burn GPU budget across hundreds of runs.
- **Termination logging:** log cap-hit / non-termination rate **separately per condition**. A cap-hit is ambiguous (genuine non-convergence vs. slow-but-fine); do NOT silently fold cap-hits into "wrong answer." If ablated traces hit the cap far more than unablated, that difference is _a result_ (ablation breaks convergence), not bookkeeping. If even unablated traces hit the cap, the cap is too tight — you created the failure.
- **Answer-probability capture (build in from the start):** the generation loop must record, per generated answer token, the log-prob the model assigned to it — enough to later recover the **log-prob of the correct answer**. These values exist during the forward pass and are discarded unless deliberately logged; a loop that keeps only the final extracted text throws them away, and recovering them means re-running everything. This is a _logging requirement_, not a metric choice — the exact graded metric (§4.2) is decided later, but the raw material must be captured now.

### 4.8 Answer extraction

Three separate parsers; extraction failures masquerade as wrong answers and may differ by condition (ablated traces format worse), biasing the gap.

**Dataset repos are fixed by the homework page — hardcode them, do not treat as an open decision:** `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `HuggingFaceH4/aime_2024`. (The last is 2024 AIME I+II, 30 rows — this is the exact AIME set to state in the report.)

- GSM8K: `#### N` integer match.
- MATH-500: `\boxed{}` extraction + symbolic equivalence via `math_verify` / standard MATH `is_equiv` — **do not hand-roll equivalence checking.**
- AIME: integer 0–999 match.
- **Log extraction-failure rate per condition.** If it differs across arms, separate it from real accuracy loss.

### 4.9 Prompts & reproducibility

- **Frozen prompt templates**, identical across conditions except the thinking toggle. One system prompt + one user format per dataset. Any cross-condition variation confounds condition with prompt.
- Record HF **revision + split** for every dataset (repos are already fixed, see §4.8) and pin **repo + revision** for both models.
- Log per-problem: condition, seed, raw trace, extracted answer, correctness, graded score, trace length, termination flag, extraction-success flag, backtrack count.

### 4.10 Cost sizing

Before full runs, do a **timing probe on ~5 problems** covering the most expensive condition (CoT J-ablated with 2× passes on a long AIME trace). Size problems-per-cell to GPU budget from that measurement rather than discovering cost mid-run. Start ~50 problems/cell **for GSM8K and MATH levels**; expand where CIs are too wide to conclude.

**AIME is the exception — n is capped by the dataset.** `HuggingFaceH4/aime_2024` has **only 30 rows** (AIME I + II, 2024). So the AIME cell is **n ≤ 30**, full stop — there is no "start ~50" headroom for the hard anchor. Note the compounding problem: the hard anchor is _both_ the most likely to floor (§4.2) _and_ the smallest-n cell, so its bootstrap CI will be widest exactly where H1 makes its strongest claim. This is a structural limitation to disclose, not fix. (Pulling in AIME 2025 to reach n≤~60 is possible but must be explicitly justified and disclosed, given the homework's insistence on stating the exact AIME set used — do not silently mix years.)

---

## 5. What to write vs. what to outsource

### 5.1 Outsource (do NOT reimplement)

- **Jacobian lens itself:** use Anthropic's `jlens` library (`github.com/anthropics/jacobian-lens`, Apache-2.0). Load the **pre-fitted** lens from `neuronpedia/jacobian-lens`. Confirmed path (a third-party project, J-Wash, loads exactly this for `Qwen/Qwen3-4B`):
  - repo: `neuronpedia/jacobian-lens`
  - file: `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`
  - API: `jlens.from_hf(hf, tok)` → `jlens.JacobianLens.from_pretrained(repo, filename=...)` → `lens.apply(model, prompt, positions=...)`. Lens holds one `[d_model, d_model]` matrix per layer.

  **Two things to verify on first pull, don't assume:** (a) **revision** — the Anthropic walkthrough uses `revision="qwen-n1000"` for the _Qwen3.5_ files, but the confirmed `qwen3-4b` filename has **no `_n1000` suffix**, so it may sit on a different revision (try default/main first). (b) **fit scale** — because the filename lacks the `_n1000` tag, do **not** assume it was fit at the paper's n=1000; read the folder's metadata/CREDIT.md to confirm the prompt count, since fit scale bears on lens quality. If it's fit at low n, note it in Limitations or fit your own at n≈1000 (the library's `jlens.fit(model, prompts=...)` — the repo says ~100 prompts is usable, quality saturates by ~1000).

  **Do NOT derive or fit the Jacobian math from scratch** unless the pre-fitted lens proves inadequate. Double-check you grab **`qwen3-4b`**, not **`qwen3.5-4b`** — different model, different architecture, incompatible Jacobians.

- **Model + datasets:** HuggingFace `transformers` + `datasets`.
- **MATH equivalence checking:** `math_verify` or standard `is_equiv`.
- **Chat template / thinking toggle:** Qwen's tokenizer handles it; read the model card for exact usage.
- **Bootstrap CIs, McNemar (if used):** standard libraries.

### 5.2 Write yourself (the actual deliverable)

- **The ablation forward-hook**: projection-out of top-k active J-lens directions across the band, with the per-step clean-pass top-10 exclusion (§4.6). This is the core novel code.
- **The random-direction control** ablation variant (matched band/k/norm, seeded).
- **The 6-condition × difficulty experiment driver**: generation in each condition, per-problem logging.
- **The three answer-extraction parsers** and per-condition failure logging.
- **The band/k calibration diagnostic** and number-token loading check.
- **The metric layer:** accuracy, graded log-prob metric, effective-backtrack counter (regex + judge harness), termination/extraction/length logging, bootstrap CIs, plots.
- **The pre-registration file** (§6) and the frozen backtrack-judge rubric (§4.4).

### 5.3 Environment

- **Dev locally** on Qwen3-0.6B/1.7B on the M4 (MPS, fp32) — debug the _entire_ pipeline (lens load, hook, exclusion, generation both modes, parsers, metrics, plots). Parameterize `MODEL_NAME` / `LENS_FILE` / device+dtype in ONE place.
- **Run for real** on a rented GPU (single A100 40GB / L40S, ~$1–2/hr; ~20–40 GPU-hrs incl. false starts; within the $100 reimbursement). bf16, `attn_implementation="eager"` (needed for clean autograd/hook behavior).
- **Trust local for logic, GPU for numbers.** MPS-fp32 and CUDA-bf16 differ numerically; all _reported_ results come from the GPU run. The small dev model may show no workspace effect at all — that's expected, not a bug; locally you're only verifying the code runs (use the multi-hop positive control to confirm the hook bites).
- Bonus: the smaller dev model **doubles as the size control** (§4.3).

---

## 6. Deliverables & ordering

1. **Pre-registration file** — top-level causal claim (§2.1), mechanism (§2.2), primary interaction + H0/H1/H1′ (§2.3), mechanism prediction (§2.4), disconfirmation conditions (§2.5). **Committed and dated before any main run.**
2. **Frozen backtrack-judge rubric** (`.md`), committed before looking at result data.
3. **GitHub repo:** short README, scripts/commands run, exact experimental settings, pinned dataset/model revisions, enough to reproduce.
4. **~2-page report**, sections mandated by the homework:
   1. Hypothesis (recorded before experiments)
   2. Experiment design (conditions, comparisons, controls, metrics — enough for confirming _or_ disconfirming evidence to be visible)
   3. Experimental details (implementation + reproduction info)
   4. Experimental results (plots, tables, uncertainty, failures, missing runs)
   5. Analysis (what results say about the hypothesis, alternative explanations remaining, what to test next)

### Report must explicitly address (rubric-critical)

- Controls separating J-space ablation from **broad degradation** (random-direction + size control, framed as shape-not-level).
- **Negative and inconclusive results** reported as clearly as positive ones. "Inconclusive, CI spans zero, here's why and what more compute resolves" is fully creditable.
- **Limitations:** (a) J-lens is single-token — a threat to the whole premise on _math_; the paper itself found number words had the lowest workspace loading, possibly a lens artifact. (b) Number-token loading result from §3.A.5. (c) **Faithfulness caveat** — backtracking-in-text ≠ backtracking-in-mechanism; the metric measures the trace, the claim is about selection. (d) **Family-confound** (independent Praxagent audit): a geometric workspace band does not guarantee behavioral J-dependence — Qwen may resolve concepts without behavioral dependence on the band. (e) "Direct" mode is a trained-in but non-neutral condition on a hybrid-reasoning model. (f) Qwen3's RL-trained long-CoT (backtracking, self-verification) may not be the same object as the paper's prompted CoT on Sonnet.

---

## 7. Ordered build path (risk front-loaded)

1. Harness on 5 GSM8K problems: load model, generate both modes, extract, score. _Proves harness before interpretability._
2. Load pre-fitted lens; lens sanity check (`calc:` intermediates). **Gate.**
3. Band diagnostic + k + number-token loading check → fix band/k. **[LOAD-BEARING]**
4. Ablation hook + per-step top-10 exclusion; random-control variant.
5. Multi-hop positive control — ablation must tank two-hop reasoning. **Gate.**
6. Timing probe (5 problems, most expensive condition) → size problems-per-cell.
7. Full 6-condition × difficulty grid + size control; all logging.
8. Metrics, bootstrap CIs, backtrack judge, plots, write-up.

Steps 2–5 hold the risk; 6–8 are mechanical once the hook is validated. If time-boxed: logit-lens ablation as a fallback for the lens (needs no Jacobian — just `W_U` on intermediate activations; the paper notes it captures much of the workspace structure with lower reliability) + steps 1,2,3,4,5,7 is a complete, defensible study.

---

## 8. Implementation lock-ins (post-instrument, Qwen3-4B)

Recorded after the instrument gate; does not change the hypotheses in §2.

- **Ablation selection:** top-k activated **J-lens token vectors** \(v_t = J^\top u_t\) ranked by lens logit on \(h\), not right singular vectors of \(J\). Survivors after clean top-10 exclusion are Gram–Schmidt’d and the span is projected out; the ranking walks past excluded tokens to refill up to \(k\) (slightly stronger than dropping to \(<k\)).
- **Band:** layers **[27, 33]** on Qwen3-4B (36 layers). The match-rate diagnostic is a **late ramp**, not a mid-network plateau; auto-select preferred [30, 33]; human lock includes onset. Prompt tokens are ablated.
- **Instrument success:** exact-match on multihop items that are clean-correct, with J accuracy clearly below matched-norm random. Pooled EM / gold-lp over items the model does not know can null or invert — do not use those as the sole gate.
- **Artifacts:** session outputs live under `runs/<YYYY-MM-DD_slug>/` with a `HANDOFF.md`; see `runs/README.md`.
- **Math caveat unchanged:** number-token J-space loading remains near zero; a null on GSM8K/MATH CoT EM is compatible with a working factual instrument.
