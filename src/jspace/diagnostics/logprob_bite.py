"""Gold / confidence logprob bite summaries (instrument probe 3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LogprobBitePair:
    """Paired clean vs ablated logprobs for one problem."""

    problem_id: str
    answers_match: bool
    clean_mean_gen_lp: float | None
    ablated_mean_gen_lp: float | None
    clean_gold_lp: float | None
    ablated_gold_lp: float | None

    def gold_delta(self) -> float | None:
        """clean − ablated; positive ⇒ ablation lowers P(gold)."""
        if self.clean_gold_lp is None or self.ablated_gold_lp is None:
            return None
        return float(self.clean_gold_lp - self.ablated_gold_lp)

    def matched_gen_delta(self) -> float | None:
        if not self.answers_match:
            return None
        if self.clean_mean_gen_lp is None or self.ablated_mean_gen_lp is None:
            return None
        return float(self.clean_mean_gen_lp - self.ablated_mean_gen_lp)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold_delta"] = self.gold_delta()
        d["matched_gen_delta"] = self.matched_gen_delta()
        return d


@dataclass(frozen=True)
class LogprobBiteSummary:
    n_pairs: int
    n_with_gold: int
    n_matched_answers: int
    mean_gold_lp_delta: float | None
    fraction_gold_bite: float | None
    mean_matched_gen_lp_delta: float | None
    fraction_matched_gen_bite: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def summarize_logprob_bite(pairs: Sequence[LogprobBitePair]) -> LogprobBiteSummary:
    """Aggregate whether ablation reliably lowers gold / matched-gen logprob."""
    gold_deltas = [d for p in pairs if (d := p.gold_delta()) is not None]
    matched_gen = [d for p in pairs if (d := p.matched_gen_delta()) is not None]
    n_matched = sum(1 for p in pairs if p.answers_match)

    return LogprobBiteSummary(
        n_pairs=len(pairs),
        n_with_gold=len(gold_deltas),
        n_matched_answers=n_matched,
        mean_gold_lp_delta=_mean(gold_deltas),
        fraction_gold_bite=(
            float(sum(1 for d in gold_deltas if d > 0) / len(gold_deltas))
            if gold_deltas
            else None
        ),
        mean_matched_gen_lp_delta=_mean(matched_gen),
        fraction_matched_gen_bite=(
            float(sum(1 for d in matched_gen if d > 0) / len(matched_gen))
            if matched_gen
            else None
        ),
    )


def logprob_bite_verdict(summary: LogprobBiteSummary, *, min_mean_delta: float = 0.05) -> str:
    """Short interpretability label for probe 3."""
    if summary.n_with_gold == 0:
        return "no_gold_logprobs"
    mean_d = summary.mean_gold_lp_delta
    frac = summary.fraction_gold_bite
    assert mean_d is not None and frac is not None
    if mean_d >= min_mean_delta and frac >= 0.6:
        return "gold_lp_bite_confirmed"
    if mean_d <= -min_mean_delta:
        return "gold_lp_improved_under_ablation"
    return "gold_lp_bite_weak_or_absent"
