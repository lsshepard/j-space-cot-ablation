"""Unit tests for qualitative token-trace helpers (no GPU)."""

from jspace.diagnostics.token_trace import (
    DirectionTrace,
    IntermediateHit,
    PositionLayerTrace,
    ProblemTokenTrace,
    TokenFreq,
    _matches_any,
    _norm,
    format_problem_trace_report,
    summarize_traces,
)


def test_norm_and_match():
    assert _norm(" Brazil!") == "brazil"
    assert _matches_any(" Brazil", ["brazil", "China"]) == "brazil"
    assert _matches_any("iron", ["Fe", "iron"]) == "iron"
    assert _matches_any("xyz", ["brazil"]) is None


def _minimal_trace(
    pid: str,
    *,
    surv_hits: int = 0,
    excl_hits: int = 0,
    clean_hits: int = 0,
    clean_c: str = "A",
    j_c: str = "A",
) -> ProblemTokenTrace:
    hits = []
    for _ in range(surv_hits):
        hits.append(
            IntermediateHit(
                kind="active_survivor",
                layer_idx=27,
                abs_pos=1,
                prompt_token="x",
                decoded_token="Brazil",
                intermediate="Brazil",
                coeff_abs=1.0,
            )
        )
    for _ in range(excl_hits):
        hits.append(
            IntermediateHit(
                kind="active_excluded",
                layer_idx=27,
                abs_pos=1,
                prompt_token="x",
                decoded_token="Brazil",
                intermediate="Brazil",
                coeff_abs=1.0,
            )
        )
    for _ in range(clean_hits):
        hits.append(
            IntermediateHit(
                kind="clean_topk",
                layer_idx=None,
                abs_pos=1,
                prompt_token="x",
                decoded_token="Brazil",
                intermediate="Brazil",
            )
        )
    last = PositionLayerTrace(
        layer_idx=27,
        abs_pos=10,
        prompt_token=" ",
        n_active=10,
        n_survivors=9,
        n_excluded=1,
        delta_h_norm=12.0,
        directions=[
            DirectionTrace(
                rank=0,
                coeff_abs=3.0,
                top_token_id=1,
                top_token="Brazil",
                excluded=False,
                matches_intermediate=True,
            ),
            DirectionTrace(
                rank=1,
                coeff_abs=2.0,
                top_token_id=2,
                top_token="the",
                excluded=True,
            ),
        ],
    )
    return ProblemTokenTrace(
        problem_id=pid,
        prompt="Fact: ...",
        gold_answer="Portuguese",
        intermediates=["Brazil"],
        prompt_token_count=11,
        band_start=27,
        band_end=33,
        k=10,
        exclude_topk=10,
        clean_completion=clean_c,
        jspace_completion=j_c,
        last_position_traces=[last],
        last_position_clean_topk=["the", "a"],
        mean_n_active=10.0,
        mean_n_survivors=9.5,
        mean_n_excluded=0.5,
        mean_survivor_fraction=0.95,
        mean_delta_h_norm=20.0,
        top_survivor_tokens=[TokenFreq(token="Brazil", count=3, frac=0.3)],
        top_excluded_tokens=[TokenFreq(token="the", count=2, frac=0.2)],
        intermediate_hits=hits,
        n_steps=70,
        n_steps_with_exclusion=5,
        n_intermediate_survivor_hits=surv_hits,
        n_intermediate_excluded_hits=excl_hits,
        n_intermediate_clean_topk_hits=clean_hits,
    )


def test_summarize_traces():
    traces = [
        _minimal_trace("a", surv_hits=2, clean_hits=1, clean_c="X", j_c="Y"),
        _minimal_trace("b", excl_hits=1, clean_c="X", j_c="X"),
    ]
    summary = summarize_traces(traces)
    assert summary["n"] == 2
    assert summary["problems_with_intermediate_survivor"] == 1
    assert summary["problems_with_intermediate_excluded"] == 1
    assert summary["problems_with_intermediate_in_clean_topk"] == 1
    assert summary["problems_where_completion_changed"] == 1
    assert abs(summary["mean_survivor_fraction"] - 0.95) < 1e-9


def test_format_report_contains_markers():
    tr = _minimal_trace("amazon-language", surv_hits=1)
    report = format_problem_trace_report(tr)
    assert "amazon-language" in report
    assert "[S]" in report
    assert "[X]" in report
    assert "Brazil" in report
