"""Qualitative per-direction ablation traces from the live ablation hook path."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from jspace.ablation import (
    AblationConfig,
    AblationFactors,
    AblationHookState,
    AblationStepDiag,
    ablation_hooks,
    build_ablation_factors,
    clean_topk_by_position,
)
from jspace.data import Problem
from jspace.generate import build_chat_text, generate_with_ablation
from jspace.load import LoadedModel


@dataclass(frozen=True)
class DirectionTrace:
    """Decoded view of one DirectionExclusionInfo from the live hook."""

    rank: int
    coeff_abs: float
    top_token_id: int
    top_token: str
    excluded: bool
    matches_intermediate: bool = False
    matches_gold: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PositionLayerTrace:
    layer_idx: int
    abs_pos: int
    prompt_token: str
    n_active: int
    n_survivors: int
    n_excluded: int
    delta_h_norm: float
    directions: list[DirectionTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "abs_pos": self.abs_pos,
            "prompt_token": self.prompt_token,
            "n_active": self.n_active,
            "n_survivors": self.n_survivors,
            "n_excluded": self.n_excluded,
            "delta_h_norm": self.delta_h_norm,
            "directions": [d.to_dict() for d in self.directions],
        }


@dataclass
class IntermediateHit:
    """Where a known intermediate appears in clean top-k or active J-dirs."""

    kind: str  # clean_topk | active_excluded | active_survivor
    layer_idx: int | None
    abs_pos: int
    prompt_token: str
    decoded_token: str
    intermediate: str
    coeff_abs: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenFreq:
    token: str
    count: int
    frac: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProblemTokenTrace:
    problem_id: str
    prompt: str
    gold_answer: str
    intermediates: list[str]
    prompt_token_count: int
    band_start: int
    band_end: int
    k: int
    exclude_topk: int
    clean_completion: str
    jspace_completion: str
    # Full direction dump: last prompt position × band layers (prediction site).
    last_position_traces: list[PositionLayerTrace]
    # Clean top-k at last position (decoded).
    last_position_clean_topk: list[str]
    # Aggregate over all prompt positions × band layers.
    mean_n_active: float
    mean_n_survivors: float
    mean_n_excluded: float
    mean_survivor_fraction: float
    mean_delta_h_norm: float
    top_survivor_tokens: list[TokenFreq]
    top_excluded_tokens: list[TokenFreq]
    intermediate_hits: list[IntermediateHit]
    n_steps: int
    n_steps_with_exclusion: int
    n_intermediate_survivor_hits: int
    n_intermediate_excluded_hits: int
    n_intermediate_clean_topk_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "prompt": self.prompt,
            "gold_answer": self.gold_answer,
            "intermediates": self.intermediates,
            "prompt_token_count": self.prompt_token_count,
            "band_start": self.band_start,
            "band_end": self.band_end,
            "k": self.k,
            "exclude_topk": self.exclude_topk,
            "clean_completion": self.clean_completion,
            "jspace_completion": self.jspace_completion,
            "last_position_traces": [t.to_dict() for t in self.last_position_traces],
            "last_position_clean_topk": self.last_position_clean_topk,
            "mean_n_active": self.mean_n_active,
            "mean_n_survivors": self.mean_n_survivors,
            "mean_n_excluded": self.mean_n_excluded,
            "mean_survivor_fraction": self.mean_survivor_fraction,
            "mean_delta_h_norm": self.mean_delta_h_norm,
            "top_survivor_tokens": [t.to_dict() for t in self.top_survivor_tokens],
            "top_excluded_tokens": [t.to_dict() for t in self.top_excluded_tokens],
            "intermediate_hits": [h.to_dict() for h in self.intermediate_hits],
            "n_steps": self.n_steps,
            "n_steps_with_exclusion": self.n_steps_with_exclusion,
            "n_intermediate_survivor_hits": self.n_intermediate_survivor_hits,
            "n_intermediate_excluded_hits": self.n_intermediate_excluded_hits,
            "n_intermediate_clean_topk_hits": self.n_intermediate_clean_topk_hits,
        }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _matches_any(decoded: str, needles: Sequence[str]) -> str | None:
    d = _norm(decoded)
    if not d:
        return None
    for needle in needles:
        n = _norm(needle)
        if not n:
            continue
        if d == n or n in d or d in n:
            return needle
    return None


def _token_freqs(counter: Counter[str], *, top_n: int = 25) -> list[TokenFreq]:
    total = sum(counter.values()) or 1
    return [
        TokenFreq(token=tok, count=cnt, frac=cnt / total)
        for tok, cnt in counter.most_common(top_n)
    ]


def _direction_traces(
    step: AblationStepDiag,
    *,
    tokenizer: Any,
    intermediates: Sequence[str],
    gold: str,
) -> list[DirectionTrace]:
    out: list[DirectionTrace] = []
    for info in step.directions:
        tok = _decode_token(tokenizer, info.top_token_id)
        out.append(
            DirectionTrace(
                rank=info.rank,
                coeff_abs=info.coeff_abs,
                top_token_id=info.top_token_id,
                top_token=tok,
                excluded=info.excluded,
                matches_intermediate=_matches_any(tok, intermediates) is not None,
                matches_gold=_matches_any(tok, [gold]) is not None,
            )
        )
    return out


def _summarize_diag_steps(
    steps: Sequence[AblationStepDiag],
    *,
    tokenizer: Any,
    prompt_token_strs: Sequence[str],
    excluded_by_position: Sequence[set[int]],
    intermediates: Sequence[str],
    gold: str,
    last_pos: int,
) -> tuple[
    list[PositionLayerTrace],
    list[str],
    list[IntermediateHit],
    Counter[str],
    Counter[str],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """Decode/summarize hook diag_steps; no ablation math here."""
    last_position_traces: list[PositionLayerTrace] = []
    survivor_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()
    hits: list[IntermediateHit] = []
    n_active_list: list[int] = []
    n_surv_list: list[int] = []
    n_excl_list: list[int] = []
    delta_list: list[float] = []
    n_steps_with_exclusion = 0

    for abs_pos, excl in enumerate(excluded_by_position):
        ptok = prompt_token_strs[abs_pos] if abs_pos < len(prompt_token_strs) else ""
        for tid in excl:
            tok = _decode_token(tokenizer, tid)
            matched = _matches_any(tok, intermediates)
            if matched is not None:
                hits.append(
                    IntermediateHit(
                        kind="clean_topk",
                        layer_idx=None,
                        abs_pos=abs_pos,
                        prompt_token=ptok,
                        decoded_token=tok,
                        intermediate=matched,
                    )
                )

    last_clean_topk = sorted(
        _decode_token(tokenizer, tid) for tid in excluded_by_position[last_pos]
    )

    for step in steps:
        dirs = _direction_traces(
            step, tokenizer=tokenizer, intermediates=intermediates, gold=gold
        )
        n_excl = sum(1 for d in dirs if d.excluded)
        ptok = (
            prompt_token_strs[step.abs_pos]
            if 0 <= step.abs_pos < len(prompt_token_strs)
            else ""
        )
        plt = PositionLayerTrace(
            layer_idx=step.layer_idx,
            abs_pos=step.abs_pos,
            prompt_token=ptok,
            n_active=step.n_active,
            n_survivors=step.n_survivors,
            n_excluded=n_excl,
            delta_h_norm=step.delta_h_norm,
            directions=dirs,
        )
        n_active_list.append(step.n_active)
        n_surv_list.append(step.n_survivors)
        n_excl_list.append(n_excl)
        delta_list.append(step.delta_h_norm)
        if n_excl > 0:
            n_steps_with_exclusion += 1

        for d in dirs:
            if d.excluded:
                excluded_counter[d.top_token] += 1
                if d.matches_intermediate:
                    matched = _matches_any(d.top_token, intermediates) or "?"
                    hits.append(
                        IntermediateHit(
                            kind="active_excluded",
                            layer_idx=step.layer_idx,
                            abs_pos=step.abs_pos,
                            prompt_token=ptok,
                            decoded_token=d.top_token,
                            intermediate=matched,
                            coeff_abs=d.coeff_abs,
                        )
                    )
            else:
                survivor_counter[d.top_token] += 1
                if d.matches_intermediate:
                    matched = _matches_any(d.top_token, intermediates) or "?"
                    hits.append(
                        IntermediateHit(
                            kind="active_survivor",
                            layer_idx=step.layer_idx,
                            abs_pos=step.abs_pos,
                            prompt_token=ptok,
                            decoded_token=d.top_token,
                            intermediate=matched,
                            coeff_abs=d.coeff_abs,
                        )
                    )

        if step.abs_pos == last_pos:
            last_position_traces.append(plt)

    last_position_traces.sort(key=lambda t: t.layer_idx)
    return (
        last_position_traces,
        last_clean_topk,
        hits,
        survivor_counter,
        excluded_counter,
        n_active_list,
        n_surv_list,
        n_excl_list,
        delta_list,
        n_steps_with_exclusion,
    )


@torch.inference_mode()
def collect_ablation_diag_steps(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    ablation: AblationConfig,
    factors: AblationFactors | None = None,
) -> tuple[list[AblationStepDiag], list[set[int]]]:
    """
    Same clean-exclude → ablated-forward path as graded/generate, with diag on.

    Returns (diag_steps from the live hook, clean top-k exclusion sets).
    """
    if ablation.kind == "none":
        raise ValueError("collect_ablation_diag_steps requires an ablating config")
    cfg = AblationConfig(
        kind=ablation.kind,
        band_start=ablation.band_start,
        band_end=ablation.band_end,
        k=ablation.k,
        seed=ablation.seed,
        exclude_topk=ablation.exclude_topk,
        ablate_prompt_tokens=ablation.ablate_prompt_tokens,
        collect_diag=True,
    )
    if factors is None:
        factors = build_ablation_factors(loaded.hf_model, loaded.lens, cfg)

    prompt_len = int(input_ids.shape[-1])
    state = AblationHookState(prompt_token_count=prompt_len, collect_diag=True)
    # Paper §4.6 / graded.py: exclusion from the clean forward at this prefix.
    clean_out = loaded.hf_model(input_ids=input_ids, attention_mask=attention_mask)
    state.excluded_by_position = clean_topk_by_position(
        clean_out.logits, cfg.exclude_topk
    )
    state.past_token_count = 0
    with ablation_hooks(loaded.hf_model, loaded.lens, cfg, state, factors=factors):
        loaded.hf_model(input_ids=input_ids, attention_mask=attention_mask)
    return list(state.diag_steps), list(state.excluded_by_position)


@torch.inference_mode()
def trace_problem_tokens(
    loaded: LoadedModel,
    problem: Problem,
    *,
    band_start: int,
    band_end: int,
    k: int = 10,
    exclude_topk: int = 10,
    max_new_tokens: int = 16,
    seed: int = 0,
    factors: AblationFactors | None = None,
) -> ProblemTokenTrace:
    """
    Run the live jspace ablation once and decode its per-direction diag.

    Selection / exclusion / projection are entirely inside ablation_hooks;
    this function only decodes token ids and aggregates for qualitative review.
    """
    intermediates = list((problem.meta or {}).get("intermediates") or [])
    text = build_chat_text(
        loaded.tokenizer,
        problem.dataset,
        problem.prompt,
        enable_thinking=False,
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(loaded.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(loaded.device)

    prompt_len = int(input_ids.shape[-1])
    prompt_token_strs = [
        _decode_token(loaded.tokenizer, int(tid)) for tid in input_ids[0].tolist()
    ]
    last_pos = prompt_len - 1

    cfg = AblationConfig(
        kind="jspace",
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        ablate_prompt_tokens=True,
        collect_diag=True,
    )
    if factors is None:
        factors = build_ablation_factors(loaded.hf_model, loaded.lens, cfg)

    diag_steps, excluded_by_pos = collect_ablation_diag_steps(
        loaded,
        input_ids,
        attention_mask,
        ablation=cfg,
        factors=factors,
    )

    (
        last_position_traces,
        last_clean_topk,
        hits,
        survivor_counter,
        excluded_counter,
        n_active_list,
        n_surv_list,
        n_excl_list,
        delta_list,
        n_steps_with_exclusion,
    ) = _summarize_diag_steps(
        diag_steps,
        tokenizer=loaded.tokenizer,
        prompt_token_strs=prompt_token_strs,
        excluded_by_position=excluded_by_pos,
        intermediates=intermediates,
        gold=problem.gold_answer,
        last_pos=last_pos,
    )

    clean_gen = generate_with_ablation(
        loaded,
        problem.dataset,
        problem.prompt,
        enable_thinking=False,
        max_new_tokens=max_new_tokens,
        seed=seed,
        ablation=AblationConfig(kind="none"),
    )
    j_gen = generate_with_ablation(
        loaded,
        problem.dataset,
        problem.prompt,
        enable_thinking=False,
        max_new_tokens=max_new_tokens,
        seed=seed,
        ablation=AblationConfig(
            kind="jspace",
            band_start=band_start,
            band_end=band_end,
            k=k,
            exclude_topk=exclude_topk,
            ablate_prompt_tokens=True,
        ),
        use_kv_cache=True,
    )

    n = len(n_active_list) or 1
    surv_frac = [s / max(a, 1) for a, s in zip(n_active_list, n_surv_list)]
    n_int_surv = sum(1 for h in hits if h.kind == "active_survivor")
    n_int_excl = sum(1 for h in hits if h.kind == "active_excluded")
    n_int_clean = sum(1 for h in hits if h.kind == "clean_topk")

    return ProblemTokenTrace(
        problem_id=problem.problem_id,
        prompt=problem.prompt,
        gold_answer=problem.gold_answer,
        intermediates=intermediates,
        prompt_token_count=prompt_len,
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        clean_completion=clean_gen.text.strip(),
        jspace_completion=j_gen.text.strip(),
        last_position_traces=last_position_traces,
        last_position_clean_topk=last_clean_topk,
        mean_n_active=float(sum(n_active_list) / n),
        mean_n_survivors=float(sum(n_surv_list) / n),
        mean_n_excluded=float(sum(n_excl_list) / n),
        mean_survivor_fraction=float(sum(surv_frac) / n),
        mean_delta_h_norm=float(sum(delta_list) / n),
        top_survivor_tokens=_token_freqs(survivor_counter),
        top_excluded_tokens=_token_freqs(excluded_counter),
        intermediate_hits=hits,
        n_steps=len(n_active_list),
        n_steps_with_exclusion=n_steps_with_exclusion,
        n_intermediate_survivor_hits=n_int_surv,
        n_intermediate_excluded_hits=n_int_excl,
        n_intermediate_clean_topk_hits=n_int_clean,
    )


def format_problem_trace_report(trace: ProblemTokenTrace) -> str:
    """Human-readable qualitative report for one problem."""
    lines: list[str] = []
    lines.append(f"### {trace.problem_id}")
    lines.append(f"prompt: {trace.prompt}")
    lines.append(f"gold: {trace.gold_answer!r}  intermediates: {trace.intermediates}")
    lines.append(f"clean→ {trace.clean_completion!r}")
    lines.append(f"J-abl→ {trace.jspace_completion!r}")
    lines.append(
        f"aggregates: survivors={trace.mean_n_survivors:.2f}/{trace.mean_n_active:.2f} "
        f"(frac={trace.mean_survivor_fraction:.3f})  "
        f"excluded_dirs/step={trace.mean_n_excluded:.2f}  "
        f"‖Δh‖={trace.mean_delta_h_norm:.2f}  "
        f"steps_with_any_excl={trace.n_steps_with_exclusion}/{trace.n_steps}"
    )
    lines.append(
        f"intermediate hits: survivor={trace.n_intermediate_survivor_hits} "
        f"excluded={trace.n_intermediate_excluded_hits} "
        f"clean_topk={trace.n_intermediate_clean_topk_hits}"
    )
    lines.append(f"last-pos clean top-{trace.exclude_topk}: {trace.last_position_clean_topk}")
    lines.append("last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):")
    for plt in trace.last_position_traces:
        dirs = ", ".join(
            f"{d.rank}:{d.top_token!r}"
            f"{'[X]' if d.excluded else '[S]'}"
            f"{'*' if d.matches_intermediate else ''}"
            f"{'G' if d.matches_gold else ''}"
            f"(c={d.coeff_abs:.2f})"
            for d in plt.directions
        )
        lines.append(
            f"  L{plt.layer_idx} surv={plt.n_survivors}/{plt.n_active} "
            f"‖Δh‖={plt.delta_h_norm:.2f} | {dirs}"
        )
    if trace.top_survivor_tokens:
        top_s = ", ".join(
            f"{t.token!r}:{t.count}" for t in trace.top_survivor_tokens[:10]
        )
        lines.append(f"top survivors (all pos): {top_s}")
    if trace.top_excluded_tokens:
        top_e = ", ".join(
            f"{t.token!r}:{t.count}" for t in trace.top_excluded_tokens[:10]
        )
        lines.append(f"top excluded (all pos): {top_e}")
    if trace.intermediate_hits:
        lines.append("intermediate hit detail (first 20):")
        for h in trace.intermediate_hits[:20]:
            layer = f"L{h.layer_idx}" if h.layer_idx is not None else "clean"
            coeff = f" c={h.coeff_abs:.2f}" if h.coeff_abs is not None else ""
            lines.append(
                f"  {h.kind} {layer} pos={h.abs_pos} "
                f"@ {h.prompt_token!r} → {h.decoded_token!r} "
                f"(~{h.intermediate}){coeff}"
            )
    return "\n".join(lines)


def run_token_traces(
    loaded: LoadedModel,
    problems: list[Problem],
    *,
    band_start: int,
    band_end: int,
    k: int = 10,
    exclude_topk: int = 10,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> list[ProblemTokenTrace]:
    cfg = AblationConfig(
        kind="jspace",
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        ablate_prompt_tokens=True,
        collect_diag=True,
    )
    factors = build_ablation_factors(loaded.hf_model, loaded.lens, cfg)
    traces: list[ProblemTokenTrace] = []
    for i, problem in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] tracing {problem.problem_id}", flush=True)
        tr = trace_problem_tokens(
            loaded,
            problem,
            band_start=band_start,
            band_end=band_end,
            k=k,
            exclude_topk=exclude_topk,
            max_new_tokens=max_new_tokens,
            seed=seed,
            factors=factors,
        )
        print(format_problem_trace_report(tr), flush=True)
        print(flush=True)
        traces.append(tr)
    return traces


def summarize_traces(traces: Sequence[ProblemTokenTrace]) -> dict[str, Any]:
    """Cross-problem qualitative aggregates."""
    if not traces:
        return {"n": 0}
    return {
        "n": len(traces),
        "mean_survivor_fraction": float(
            sum(t.mean_survivor_fraction for t in traces) / len(traces)
        ),
        "mean_n_excluded": float(sum(t.mean_n_excluded for t in traces) / len(traces)),
        "mean_delta_h_norm": float(sum(t.mean_delta_h_norm for t in traces) / len(traces)),
        "problems_with_intermediate_survivor": sum(
            1 for t in traces if t.n_intermediate_survivor_hits > 0
        ),
        "problems_with_intermediate_excluded": sum(
            1 for t in traces if t.n_intermediate_excluded_hits > 0
        ),
        "problems_with_intermediate_in_clean_topk": sum(
            1 for t in traces if t.n_intermediate_clean_topk_hits > 0
        ),
        "problems_where_completion_changed": sum(
            1 for t in traces if t.clean_completion != t.jspace_completion
        ),
        "per_problem": [
            {
                "problem_id": t.problem_id,
                "gold": t.gold_answer,
                "intermediates": t.intermediates,
                "clean_completion": t.clean_completion,
                "jspace_completion": t.jspace_completion,
                "mean_survivor_fraction": t.mean_survivor_fraction,
                "mean_n_excluded": t.mean_n_excluded,
                "mean_delta_h_norm": t.mean_delta_h_norm,
                "n_intermediate_survivor_hits": t.n_intermediate_survivor_hits,
                "n_intermediate_excluded_hits": t.n_intermediate_excluded_hits,
                "n_intermediate_clean_topk_hits": t.n_intermediate_clean_topk_hits,
                "last_pos_survivor_tokens": [
                    d.top_token
                    for plt in t.last_position_traces
                    for d in plt.directions
                    if not d.excluded
                ],
                "last_pos_excluded_tokens": [
                    d.top_token
                    for plt in t.last_position_traces
                    for d in plt.directions
                    if d.excluded
                ],
                "last_pos_clean_topk": t.last_position_clean_topk,
            }
            for t in traces
        ],
    }
