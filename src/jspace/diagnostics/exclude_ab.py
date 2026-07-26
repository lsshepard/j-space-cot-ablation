"""exclude_topk on/off accuracy A/B (instrument probe 2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from jspace.metrics import accuracy


@dataclass(frozen=True)
class ExcludeABSummary:
    exclude_topk: int
    n: int
    clean_accuracy: float
    ablated_accuracy: float
    drop: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_exclude_ab(
    clean_correct: Sequence[bool],
    ablated_correct: Sequence[bool],
    *,
    exclude_topk: int,
) -> ExcludeABSummary:
    if len(clean_correct) != len(ablated_correct):
        raise ValueError("clean/ablated flag lists must be same length")
    clean_acc = accuracy(list(clean_correct))
    abl_acc = accuracy(list(ablated_correct))
    return ExcludeABSummary(
        exclude_topk=exclude_topk,
        n=len(clean_correct),
        clean_accuracy=clean_acc,
        ablated_accuracy=abl_acc,
        drop=float(clean_acc - abl_acc),
    )


def exclude_ab_verdict(
    with_exclusion: ExcludeABSummary,
    without_exclusion: ExcludeABSummary,
    *,
    drop_threshold: float = 0.2,
) -> str:
    """
    Compare paper shield (exclude_topk>0) vs shield-off (exclude_topk=0).

    Big drop only when exclusion is off → plumbing works; shield is the story.
    No drop either way → deeper band/J/hook failure.
    """
    drop_on = with_exclusion.drop
    drop_off = without_exclusion.drop
    strong_off = drop_off >= drop_threshold
    strong_on = drop_on >= drop_threshold
    weak_on = drop_on < drop_threshold / 2
    if strong_off and weak_on:
        return "shield_explains_null_gate"
    if (not strong_off) and (not strong_on):
        return "no_em_drop_either_way"
    if strong_on:
        return "em_drop_with_exclusion"
    return "mixed_em_pattern"
