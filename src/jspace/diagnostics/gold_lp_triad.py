"""Clean / J / random teacher-forced gold logprob triad (instrument gate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from jspace.ablation import AblationConfig, AblationFactors, build_ablation_factors
from jspace.config import Settings
from jspace.data import Problem
from jspace.generate import build_chat_text
from jspace.graded import teacher_forced_gold_logprob
from jspace.load import LoadedModel
from jspace.metrics import bootstrap_ci


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lo: float
    hi: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def excludes_zero_positive(self) -> bool:
        return self.lo > 0.0


@dataclass
class GoldLpTriadRow:
    problem_id: str
    gold_answer: str
    clean_lp: float
    jspace_lp: float
    random_lps: dict[int, float] = field(default_factory=dict)

    @property
    def random_lp_mean(self) -> float:
        return float(sum(self.random_lps.values()) / len(self.random_lps))

    @property
    def delta_j(self) -> float:
        """clean − J; positive ⇒ J lowers P(gold)."""
        return float(self.clean_lp - self.jspace_lp)

    @property
    def delta_r(self) -> float:
        """clean − random_mean; positive ⇒ random lowers P(gold)."""
        return float(self.clean_lp - self.random_lp_mean)

    @property
    def delta_j_minus_r(self) -> float:
        """(clean−J) − (clean−R) = R − J; positive ⇒ J hurts gold more than random."""
        return float(self.random_lp_mean - self.jspace_lp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "gold_answer": self.gold_answer,
            "clean_lp": self.clean_lp,
            "jspace_lp": self.jspace_lp,
            "random_lps": {str(k): v for k, v in self.random_lps.items()},
            "random_lp_mean": self.random_lp_mean,
            "delta_j": self.delta_j,
            "delta_r": self.delta_r,
            "delta_j_minus_r": self.delta_j_minus_r,
        }


@dataclass
class GoldLpTriadResult:
    band_start: int
    band_end: int
    k: int
    exclude_topk: int
    n: int
    random_seeds: list[int]
    rows: list[GoldLpTriadRow]
    delta_j: MeanCI
    delta_r: MeanCI
    delta_j_minus_r: MeanCI
    mean_clean_lp: float
    mean_jspace_lp: float
    mean_random_lp: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_start": self.band_start,
            "band_end": self.band_end,
            "k": self.k,
            "exclude_topk": self.exclude_topk,
            "n": self.n,
            "random_seeds": self.random_seeds,
            "mean_clean_lp": self.mean_clean_lp,
            "mean_jspace_lp": self.mean_jspace_lp,
            "mean_random_lp": self.mean_random_lp,
            "delta_j": self.delta_j.to_dict(),
            "delta_r": self.delta_r.to_dict(),
            "delta_j_minus_r": self.delta_j_minus_r.to_dict(),
            "verdict": self.verdict,
            "rows": [r.to_dict() for r in self.rows],
        }


def summarize_gold_lp_triad(
    rows: Sequence[GoldLpTriadRow],
    *,
    band_start: int,
    band_end: int,
    k: int,
    exclude_topk: int,
    random_seeds: Sequence[int],
    n_boot: int = 1000,
    boot_seed: int = 0,
) -> GoldLpTriadResult:
    if not rows:
        nan_ci = MeanCI(float("nan"), float("nan"), float("nan"))
        return GoldLpTriadResult(
            band_start=band_start,
            band_end=band_end,
            k=k,
            exclude_topk=exclude_topk,
            n=0,
            random_seeds=list(random_seeds),
            rows=[],
            delta_j=nan_ci,
            delta_r=nan_ci,
            delta_j_minus_r=nan_ci,
            mean_clean_lp=float("nan"),
            mean_jspace_lp=float("nan"),
            mean_random_lp=float("nan"),
            verdict="no_items",
        )

    dj = [r.delta_j for r in rows]
    dr = [r.delta_r for r in rows]
    djr = [r.delta_j_minus_r for r in rows]
    mj, jlo, jhi = bootstrap_ci(dj, n_boot=n_boot, seed=boot_seed)
    mr, rlo, rhi = bootstrap_ci(dr, n_boot=n_boot, seed=boot_seed + 1)
    mjr, jrlo, jrhi = bootstrap_ci(djr, n_boot=n_boot, seed=boot_seed + 2)
    delta_j = MeanCI(mj, jlo, jhi)
    delta_r = MeanCI(mr, rlo, rhi)
    delta_j_minus_r = MeanCI(mjr, jrlo, jrhi)
    return GoldLpTriadResult(
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        n=len(rows),
        random_seeds=list(random_seeds),
        rows=list(rows),
        delta_j=delta_j,
        delta_r=delta_r,
        delta_j_minus_r=delta_j_minus_r,
        mean_clean_lp=float(sum(r.clean_lp for r in rows) / len(rows)),
        mean_jspace_lp=float(sum(r.jspace_lp for r in rows) / len(rows)),
        mean_random_lp=float(sum(r.random_lp_mean for r in rows) / len(rows)),
        verdict=gold_lp_triad_verdict(delta_j, delta_j_minus_r),
    )


def gold_lp_triad_verdict(delta_j: MeanCI, delta_j_minus_r: MeanCI) -> str:
    """
    Pass: J lowers P(gold) and more than random (both bootstrap CIs > 0).
    """
    j_bite = delta_j.excludes_zero_positive
    j_over_r = delta_j_minus_r.excludes_zero_positive
    if j_bite and j_over_r:
        return "j_specific_gold_lp_bite"
    if j_bite:
        return "j_gold_lp_bite_not_above_random"
    if delta_j.hi < 0:
        return "j_raises_gold_lp"
    return "no_reliable_j_gold_lp_bite"


def _prompt_tensors(loaded: LoadedModel, problem: Problem) -> tuple[Any, Any]:
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
    return input_ids, attention_mask


def _score(
    loaded: LoadedModel,
    problem: Problem,
    prompt_ids: Any,
    attention_mask: Any,
    *,
    ablation: AblationConfig,
    factors: AblationFactors | None,
) -> float:
    lp = teacher_forced_gold_logprob(
        loaded,
        prompt_ids,
        problem.dataset,
        problem.gold_answer,
        enable_thinking=False,
        generated_token_ids=None,
        ablation=ablation if ablation.kind != "none" else None,
        attention_mask=attention_mask,
        factors=factors,
    )
    if lp is None:
        raise RuntimeError(f"empty gold tokenization for {problem.problem_id}")
    return float(lp)


def run_gold_lp_triad(
    loaded: LoadedModel,
    problems: list[Problem],
    *,
    settings: Settings,
    band_start: int,
    band_end: int,
    k: int = 10,
    exclude_topk: int = 10,
    random_seeds: Sequence[int] | None = None,
    n_boot: int = 1000,
) -> GoldLpTriadResult:
    """
    Teacher-forced P(gold) under clean / J / random (matched band/k/exclusion).

    No generation — only graded gold logprobs on the direct chat prefix.
    """
    seeds = list(random_seeds if random_seeds is not None else settings.random_ablation_seeds)
    base = dict(
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        ablate_prompt_tokens=True,
    )
    j_cfg = AblationConfig(kind="jspace", **base)
    # Jacobians/SVD depend on band+k, not on ablation kind or random seed.
    factors = build_ablation_factors(loaded.hf_model, loaded.lens, j_cfg)

    rows: list[GoldLpTriadRow] = []
    for i, problem in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] {problem.problem_id}", flush=True)
        prompt_ids, attn = _prompt_tensors(loaded, problem)
        clean_lp = _score(
            loaded,
            problem,
            prompt_ids,
            attn,
            ablation=AblationConfig(kind="none"),
            factors=None,
        )
        j_lp = _score(
            loaded,
            problem,
            prompt_ids,
            attn,
            ablation=j_cfg,
            factors=factors,
        )
        random_lps: dict[int, float] = {}
        for seed in seeds:
            r_cfg = AblationConfig(kind="random", seed=seed, **base)
            random_lps[seed] = _score(
                loaded,
                problem,
                prompt_ids,
                attn,
                ablation=r_cfg,
                factors=factors,
            )
        row = GoldLpTriadRow(
            problem_id=problem.problem_id,
            gold_answer=problem.gold_answer,
            clean_lp=clean_lp,
            jspace_lp=j_lp,
            random_lps=random_lps,
        )
        print(
            f"  gold={problem.gold_answer!r} "
            f"clean={clean_lp:.3f} J={j_lp:.3f} R={row.random_lp_mean:.3f} "
            f"ΔJ={row.delta_j:+.3f} ΔR={row.delta_r:+.3f} ΔJ−ΔR={row.delta_j_minus_r:+.3f}",
            flush=True,
        )
        rows.append(row)

    return summarize_gold_lp_triad(
        rows,
        band_start=band_start,
        band_end=band_end,
        k=k,
        exclude_topk=exclude_topk,
        random_seeds=seeds,
        n_boot=n_boot,
        boot_seed=settings.seed,
    )
