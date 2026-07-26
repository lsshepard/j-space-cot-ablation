"""Run instrument probes 1–3 on a multihop fixture slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jspace.ablation import AblationConfig, AblationStepDiag
from jspace.config import Settings
from jspace.data import Problem
from jspace.diagnostics.exclude_ab import (
    ExcludeABSummary,
    exclude_ab_verdict,
    summarize_exclude_ab,
)
from jspace.diagnostics.logprob_bite import (
    LogprobBitePair,
    LogprobBiteSummary,
    logprob_bite_verdict,
    summarize_logprob_bite,
)
from jspace.diagnostics.survival import (
    SurvivalSummary,
    summarize_survival,
    survival_verdict,
)
from jspace.extract import extract_answer
from jspace.generate import GenerationResult, generate
from jspace.graded import teacher_forced_gold_logprob
from jspace.load import LoadedModel
from jspace.metrics import mean_logprob, score_prediction
from jspace.token_budgets import resolve_max_new_tokens


@dataclass
class ArmProbe:
    kind: str
    exclude_topk: int
    answer: str | None
    correct: bool | None
    mean_gen_logprob: float | None
    gold_logprob: float | None
    hook_call_count: int
    survival: SurvivalSummary
    diag_steps: list[AblationStepDiag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "exclude_topk": self.exclude_topk,
            "answer": self.answer,
            "correct": self.correct,
            "mean_gen_logprob": self.mean_gen_logprob,
            "gold_logprob": self.gold_logprob,
            "hook_call_count": self.hook_call_count,
            "survival": self.survival.to_dict(),
            "n_diag_steps": len(self.diag_steps),
        }


@dataclass
class ProblemProbe:
    problem_id: str
    gold_answer: str
    clean: ArmProbe
    jspace_exclude: ArmProbe
    jspace_no_exclude: ArmProbe

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "gold_answer": self.gold_answer,
            "clean": self.clean.to_dict(),
            "jspace_exclude": self.jspace_exclude.to_dict(),
            "jspace_no_exclude": self.jspace_no_exclude.to_dict(),
        }


@dataclass
class InstrumentSuiteResult:
    band_start: int
    band_end: int
    k: int
    n: int
    problems: list[ProblemProbe]
    survival_exclude: SurvivalSummary
    survival_no_exclude: SurvivalSummary
    exclude_ab: ExcludeABSummary
    no_exclude_ab: ExcludeABSummary
    logprob_bite_exclude: LogprobBiteSummary
    logprob_bite_no_exclude: LogprobBiteSummary
    verdicts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_start": self.band_start,
            "band_end": self.band_end,
            "k": self.k,
            "n": self.n,
            "survival_exclude": self.survival_exclude.to_dict(),
            "survival_no_exclude": self.survival_no_exclude.to_dict(),
            "exclude_ab": self.exclude_ab.to_dict(),
            "no_exclude_ab": self.no_exclude_ab.to_dict(),
            "logprob_bite_exclude": self.logprob_bite_exclude.to_dict(),
            "logprob_bite_no_exclude": self.logprob_bite_no_exclude.to_dict(),
            "verdicts": self.verdicts,
            "problems": [p.to_dict() for p in self.problems],
        }


def _score_arm(
    loaded: LoadedModel,
    problem: Problem,
    result: GenerationResult,
    *,
    ablation: AblationConfig,
) -> tuple[str | None, bool | None, float | None]:
    extraction = extract_answer(problem.dataset, result.text)
    correct = score_prediction(
        problem.dataset, extraction.answer, problem.gold_answer
    )
    gold_lp = None
    if result.prompt_input_ids is not None:
        gold_lp = teacher_forced_gold_logprob(
            loaded,
            result.prompt_input_ids,
            problem.dataset,
            problem.gold_answer,
            enable_thinking=False,
            generated_token_ids=None,
            ablation=ablation if ablation.kind != "none" else None,
            attention_mask=result.prompt_attention_mask,
        )
    return extraction.answer, correct, gold_lp


def _run_arm(
    loaded: LoadedModel,
    problem: Problem,
    *,
    ablation: AblationConfig,
    max_new_tokens: int,
    seed: int,
) -> ArmProbe:
    result = generate(
        loaded,
        problem.dataset,
        problem.prompt,
        enable_thinking=False,
        max_new_tokens=max_new_tokens,
        seed=seed,
        ablation=ablation,
    )
    answer, correct, gold_lp = _score_arm(loaded, problem, result, ablation=ablation)
    survival = summarize_survival(result.diag_steps)
    return ArmProbe(
        kind=ablation.kind,
        exclude_topk=ablation.exclude_topk,
        answer=answer,
        correct=correct,
        mean_gen_logprob=mean_logprob(result.token_logprobs),
        gold_logprob=gold_lp,
        hook_call_count=result.hook_call_count,
        survival=survival,
        diag_steps=list(result.diag_steps),
    )


def _bite_pairs(
    problems: list[ProblemProbe],
    *,
    ablated_attr: str,
) -> list[LogprobBitePair]:
    pairs: list[LogprobBitePair] = []
    for item in problems:
        abl: ArmProbe = getattr(item, ablated_attr)
        answers_match = (
            item.clean.answer is not None
            and abl.answer is not None
            and item.clean.answer.strip().lower() == abl.answer.strip().lower()
        )
        pairs.append(
            LogprobBitePair(
                problem_id=item.problem_id,
                answers_match=answers_match,
                clean_mean_gen_lp=item.clean.mean_gen_logprob,
                ablated_mean_gen_lp=abl.mean_gen_logprob,
                clean_gold_lp=item.clean.gold_logprob,
                ablated_gold_lp=abl.gold_logprob,
            )
        )
    return pairs


def run_instrument_suite(
    loaded: LoadedModel,
    problems: list[Problem],
    *,
    settings: Settings,
    band_start: int,
    band_end: int,
    k: int = 10,
    exclude_topk: int = 10,
    seed: int | None = None,
    max_new_tokens: int | None = None,
) -> InstrumentSuiteResult:
    """
    Probes 1–3 on the same problems:
      1) survival + ‖Δh‖ under paper exclusion
      2) EM A/B exclude_topk vs 0
      3) gold / matched-gen logprob bite
    """
    run_seed = settings.seed if seed is None else seed
    budget = max_new_tokens
    if budget is None:
        budget = resolve_max_new_tokens(
            settings,
            problems[0].dataset if problems else "multihop",
            enable_thinking=False,
            ablation_kind="jspace",
        )

    probes: list[ProblemProbe] = []
    for i, problem in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] {problem.problem_id}", flush=True)
        clean = _run_arm(
            loaded,
            problem,
            ablation=AblationConfig(kind="none"),
            max_new_tokens=budget,
            seed=run_seed,
        )
        j_ex = _run_arm(
            loaded,
            problem,
            ablation=AblationConfig(
                kind="jspace",
                band_start=band_start,
                band_end=band_end,
                k=k,
                exclude_topk=exclude_topk,
                ablate_prompt_tokens=True,
                collect_diag=True,
            ),
            max_new_tokens=budget,
            seed=run_seed,
        )
        j_no = _run_arm(
            loaded,
            problem,
            ablation=AblationConfig(
                kind="jspace",
                band_start=band_start,
                band_end=band_end,
                k=k,
                exclude_topk=0,
                ablate_prompt_tokens=True,
                collect_diag=True,
            ),
            max_new_tokens=budget,
            seed=run_seed,
        )
        probes.append(
            ProblemProbe(
                problem_id=problem.problem_id,
                gold_answer=problem.gold_answer,
                clean=clean,
                jspace_exclude=j_ex,
                jspace_no_exclude=j_no,
            )
        )
        print(
            f"{problem.problem_id}: clean={clean.answer!r} "
            f"ex{exclude_topk}={j_ex.answer!r} ex0={j_no.answer!r} "
            f"surv_mean={j_ex.survival.mean_n_survivors:.2f} "
            f"Δh={j_ex.survival.mean_delta_h_norm:.4g}",
            flush=True,
        )

    all_ex_steps = [s for p in probes for s in p.jspace_exclude.diag_steps]
    all_no_steps = [s for p in probes for s in p.jspace_no_exclude.diag_steps]
    survival_ex = summarize_survival(all_ex_steps)
    survival_no = summarize_survival(all_no_steps)

    clean_flags = [bool(p.clean.correct) for p in probes]
    ex_flags = [bool(p.jspace_exclude.correct) for p in probes]
    no_flags = [bool(p.jspace_no_exclude.correct) for p in probes]
    ab_ex = summarize_exclude_ab(clean_flags, ex_flags, exclude_topk=exclude_topk)
    ab_no = summarize_exclude_ab(clean_flags, no_flags, exclude_topk=0)

    bite_ex = summarize_logprob_bite(_bite_pairs(probes, ablated_attr="jspace_exclude"))
    bite_no = summarize_logprob_bite(
        _bite_pairs(probes, ablated_attr="jspace_no_exclude")
    )

    verdicts = {
        "survival_exclude": survival_verdict(survival_ex),
        "survival_no_exclude": survival_verdict(survival_no),
        "exclude_ab": exclude_ab_verdict(ab_ex, ab_no),
        "logprob_bite_exclude": logprob_bite_verdict(bite_ex),
        "logprob_bite_no_exclude": logprob_bite_verdict(bite_no),
    }
    return InstrumentSuiteResult(
        band_start=band_start,
        band_end=band_end,
        k=k,
        n=len(probes),
        problems=probes,
        survival_exclude=survival_ex,
        survival_no_exclude=survival_no,
        exclude_ab=ab_ex,
        no_exclude_ab=ab_no,
        logprob_bite_exclude=bite_ex,
        logprob_bite_no_exclude=bite_no,
        verdicts=verdicts,
    )
