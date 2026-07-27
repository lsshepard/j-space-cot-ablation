"""Shared single-problem run → ProblemRecord helper."""

from __future__ import annotations

from jspace.ablation import AblationConfig
from jspace.backtrack import normalized_backtrack_rate, regex_backtrack_count
from jspace.config import Settings
from jspace.data import Problem
from jspace.extract import extract_answer
from jspace.generate import generate
from jspace.graded import teacher_forced_gold_logprob
from jspace.load import LoadedModel
from jspace.metrics import score_prediction
from jspace.records import ProblemRecord
from jspace.token_budgets import TokenBudgetProfile, resolve_max_new_tokens


def condition_name(enable_thinking: bool, ablation_kind: str) -> str:
    mode = "cot" if enable_thinking else "direct"
    if ablation_kind == "none":
        return f"{mode}_clean"
    if ablation_kind == "jspace":
        return f"{mode}_j_ablated"
    if ablation_kind == "random":
        return f"{mode}_random"
    raise ValueError(ablation_kind)


def run_problem(
    loaded: LoadedModel,
    problem: Problem,
    *,
    settings: Settings,
    enable_thinking: bool,
    ablation: AblationConfig,
    seed: int,
    max_new_tokens: int | None = None,
    token_profile: TokenBudgetProfile | None = None,
    local_fast: bool = False,
) -> ProblemRecord:
    budget = max_new_tokens
    if budget is None:
        budget = resolve_max_new_tokens(
            settings,
            problem.dataset,
            enable_thinking=enable_thinking,
            profile=token_profile,
            ablation_kind=ablation.kind,
            local_fast=local_fast,
        )

    result = generate(
        loaded,
        problem.dataset,
        problem.prompt,
        enable_thinking=enable_thinking,
        max_new_tokens=budget,
        seed=seed,
        ablation=ablation,
        early_stop_on_answer=settings.early_stop_on_answer,
    )
    extraction = extract_answer(problem.dataset, result.text)
    correct = score_prediction(problem.dataset, extraction.answer, problem.gold_answer)
    graded = None
    if result.prompt_input_ids is not None:
        graded = teacher_forced_gold_logprob(
            loaded,
            result.prompt_input_ids,
            problem.dataset,
            problem.gold_answer,
            enable_thinking=enable_thinking,
            generated_token_ids=result.token_ids,
            ablation=ablation if ablation.kind != "none" else None,
            attention_mask=result.prompt_attention_mask,
        )
    bt_count = regex_backtrack_count(result.text) if enable_thinking else None
    bt_rate = (
        normalized_backtrack_rate(bt_count, len(result.token_ids))
        if bt_count is not None
        else None
    )
    return ProblemRecord(
        problem_id=problem.problem_id,
        dataset=problem.dataset,
        difficulty=problem.difficulty,
        condition=condition_name(enable_thinking, ablation.kind),
        seed=seed,
        model_name=loaded.model_name,
        enable_thinking=enable_thinking,
        ablation=ablation.kind,
        raw_trace=result.text,
        extracted_answer=extraction.answer,
        gold_answer=problem.gold_answer,
        correct=correct,
        graded_logprob=graded,
        trace_length_tokens=len(result.token_ids),
        hit_token_cap=result.hit_token_cap,
        extraction_success=extraction.success,
        backtrack_count=bt_count,
        backtrack_rate=bt_rate,
        direct_leak_flag=result.direct_leak_flag,
        token_logprobs=result.token_logprobs,
        extra={
            "hook_call_count": result.hook_call_count,
            "max_new_tokens": budget,
            "local_fast": local_fast and ablation.kind != "none",
            "early_stopped": result.early_stopped,
            "early_stop_on_answer": settings.early_stop_on_answer,
            # Source problem text (not only restated inside raw_trace).
            "prompt": problem.prompt,
        },
    )
