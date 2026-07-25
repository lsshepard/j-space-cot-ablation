"""Unit tests for instrument-probe summarizers (no GPU)."""

from jspace.ablation import AblationStepDiag
from jspace.diagnostics.exclude_ab import exclude_ab_verdict, summarize_exclude_ab
from jspace.diagnostics.logprob_bite import (
    LogprobBitePair,
    logprob_bite_verdict,
    summarize_logprob_bite,
)
from jspace.diagnostics.survival import summarize_survival, survival_verdict


def _step(*, active: int, survivors: int, delta: float) -> AblationStepDiag:
    return AblationStepDiag(
        layer_idx=27,
        abs_pos=0,
        n_active=active,
        n_survivors=survivors,
        delta_h_norm=delta,
    )


def test_summarize_survival_empty():
    s = summarize_survival([])
    assert s.n_steps == 0
    assert survival_verdict(s) == "no_edits_recorded"


def test_summarize_survival_toothless():
    steps = [_step(active=10, survivors=0, delta=0.0) for _ in range(5)]
    s = summarize_survival(steps)
    assert s.n_steps == 5
    assert s.mean_n_survivors == 0.0
    assert s.fraction_zero_survivors == 1.0
    assert s.fraction_near_zero_delta == 1.0
    assert survival_verdict(s) == "toothless_exclusion_or_null_edit"


def test_summarize_survival_active():
    steps = [
        _step(active=10, survivors=4, delta=1.5),
        _step(active=10, survivors=6, delta=2.0),
    ]
    s = summarize_survival(steps)
    assert s.mean_n_active == 10.0
    assert s.mean_n_survivors == 5.0
    assert abs(s.mean_survivor_fraction - 0.5) < 1e-9
    assert s.mean_delta_h_norm == 1.75
    assert survival_verdict(s) == "mechanically_active"


def test_summarize_exclude_ab_drop():
    clean = [True, True, True, False]
    abl = [True, False, False, False]
    summary = summarize_exclude_ab(clean, abl, exclude_topk=10)
    assert summary.n == 4
    assert abs(summary.clean_accuracy - 0.75) < 1e-9
    assert abs(summary.ablated_accuracy - 0.25) < 1e-9
    assert abs(summary.drop - 0.5) < 1e-9


def test_exclude_ab_verdict_shield():
    with_ex = summarize_exclude_ab(
        [True, True, True, True, True],
        [True, True, True, True, True],
        exclude_topk=10,
    )
    without = summarize_exclude_ab(
        [True, True, True, True, True],
        [False, False, False, False, False],
        exclude_topk=0,
    )
    assert with_ex.drop == 0.0
    assert without.drop == 1.0
    assert exclude_ab_verdict(with_ex, without) == "shield_explains_null_gate"


def test_exclude_ab_verdict_no_drop():
    with_ex = summarize_exclude_ab(
        [True, True, False, False],
        [True, True, False, True],
        exclude_topk=10,
    )
    without = summarize_exclude_ab(
        [True, True, False, False],
        [True, True, True, False],
        exclude_topk=0,
    )
    assert with_ex.drop < 0.2
    assert without.drop < 0.2
    assert exclude_ab_verdict(with_ex, without) == "no_em_drop_either_way"


def test_logprob_bite_gold_confirmed():
    pairs = [
        LogprobBitePair("a", True, -0.1, -0.8, -0.05, -0.6),
        LogprobBitePair("b", True, -0.2, -0.9, -0.1, -0.7),
        LogprobBitePair("c", False, -0.3, -1.0, -0.2, -0.9),
    ]
    summary = summarize_logprob_bite(pairs)
    assert summary.n_pairs == 3
    assert summary.n_matched_answers == 2
    assert summary.n_with_gold == 3
    assert summary.mean_gold_lp_delta is not None
    assert summary.mean_gold_lp_delta > 0.4
    assert summary.fraction_gold_bite == 1.0
    assert summary.mean_matched_gen_lp_delta is not None
    assert abs(summary.mean_matched_gen_lp_delta - 0.7) < 1e-9
    assert logprob_bite_verdict(summary) == "gold_lp_bite_confirmed"


def test_logprob_bite_improved_under_ablation():
    pairs = [
        LogprobBitePair("a", True, -0.5, -0.1, -0.8, -0.1),
        LogprobBitePair("b", True, -0.4, -0.05, -0.7, -0.05),
    ]
    summary = summarize_logprob_bite(pairs)
    assert logprob_bite_verdict(summary) == "gold_lp_improved_under_ablation"
