"""Unit tests for gold-lp triad summarizer (no GPU)."""

from jspace.diagnostics.gold_lp_triad import (
    GoldLpTriadRow,
    MeanCI,
    gold_lp_triad_verdict,
    summarize_gold_lp_triad,
)


def _row(pid: str, clean: float, j: float, r: float) -> GoldLpTriadRow:
    return GoldLpTriadRow(
        problem_id=pid,
        gold_answer="x",
        clean_lp=clean,
        jspace_lp=j,
        random_lps={0: r, 1: r, 2: r},
    )


def test_summarize_j_specific_bite():
    # J hurts a lot; random barely hurts.
    rows = [
        _row("a", -0.1, -1.0, -0.15),
        _row("b", -0.2, -1.2, -0.25),
        _row("c", -0.05, -0.9, -0.1),
        _row("d", -0.3, -1.5, -0.35),
        _row("e", -0.1, -1.1, -0.2),
    ]
    result = summarize_gold_lp_triad(
        rows,
        band_start=27,
        band_end=31,
        k=10,
        exclude_topk=10,
        random_seeds=(0, 1, 2),
        n_boot=200,
        boot_seed=0,
    )
    assert result.n == 5
    assert result.delta_j.mean > 0.7
    assert result.delta_j_minus_r.mean > 0.5
    assert result.verdict == "j_specific_gold_lp_bite"


def test_verdict_not_above_random():
    dj = MeanCI(mean=0.5, lo=0.2, hi=0.8)
    djr = MeanCI(mean=0.05, lo=-0.1, hi=0.2)
    assert gold_lp_triad_verdict(dj, djr) == "j_gold_lp_bite_not_above_random"


def test_verdict_no_bite():
    dj = MeanCI(mean=0.02, lo=-0.1, hi=0.15)
    djr = MeanCI(mean=-0.01, lo=-0.2, hi=0.1)
    assert gold_lp_triad_verdict(dj, djr) == "no_reliable_j_gold_lp_bite"


def test_verdict_raises():
    dj = MeanCI(mean=-0.4, lo=-0.6, hi=-0.1)
    djr = MeanCI(mean=-0.2, lo=-0.4, hi=0.0)
    assert gold_lp_triad_verdict(dj, djr) == "j_raises_gold_lp"
