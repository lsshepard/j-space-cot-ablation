from jspace.backtrack import BacktrackSpan
from jspace.backtrack_metrics import (
    AdjustedTraceRate,
    adjusted_trace_rates,
    bootstrap_adjusted_means,
    cot_rows,
    precision_by_stratum,
    spans_from_rows,
    stratified_span_sample,
    summarize_regex_backtracks,
    wilson_ci,
)


def _cot_row(**kwargs):
    base = {
        "problem_id": "p1",
        "dataset": "gsm8k",
        "condition": "cot_clean",
        "enable_thinking": True,
        "raw_trace": "First try. Wait, actually multiply instead.",
        "trace_length_tokens": 100,
        "backtrack_count": 1,
        "backtrack_rate": 0.01,
        "extra": {"cell": "gsm8k"},
    }
    base.update(kwargs)
    return base


def test_cot_rows_filters_direct():
    rows = [_cot_row(), _cot_row(condition="direct_clean", enable_thinking=False)]
    assert len(cot_rows(rows)) == 1


def test_summarize_groups_by_cell_and_condition():
    rows = [
        _cot_row(condition="cot_clean", backtrack_rate=0.02),
        _cot_row(condition="cot_j_ablated", backtrack_rate=0.01),
    ]
    summary = summarize_regex_backtracks(rows)
    assert summary["n_cot_traces"] == 2
    assert "gsm8k::cot_clean" in summary["strata"]
    assert summary["strata"]["gsm8k::cot_clean"]["mean_rate"] == 0.02


def test_stratified_span_sample_caps_per_stratum():
    spans = [
        BacktrackSpan("s1", "t", 0, 5, "wait", "wait a"),
        BacktrackSpan("s2", "t", 6, 10, "wait", "wait b"),
        BacktrackSpan("s3", "t2", 0, 5, "actually", "actually c"),
    ]
    strata = {"s1": "a", "s2": "a", "s3": "b"}
    sample = stratified_span_sample(spans, strata, per_stratum=1, seed=0)
    assert len(sample) == 2
    assert {s.span_id for s in sample} <= {"s1", "s2", "s3"}


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(3, 10)
    assert 0.0 <= lo <= 0.3 <= hi <= 1.0


def test_bootstrap_adjusted_means_recovers_point():
    traces = [
        AdjustedTraceRate("p1", "cot_clean", "gsm8k", 0.01, 0.004),
        AdjustedTraceRate("p2", "cot_clean", "gsm8k", 0.02, 0.008),
    ]
    out = bootstrap_adjusted_means(traces, n_boot=500, seed=0)
    row = out["gsm8k::cot_clean"]
    assert row["n_traces"] == 2
    assert abs(row["mean_adjusted_rate"] - 0.006) < 1e-9
    assert row["ci_lo"] <= row["mean_adjusted_rate"] <= row["ci_hi"]


def test_adjusted_trace_rates_scales_by_ppv():
    rows = [_cot_row(backtrack_rate=0.02)]
    precision = {"gsm8k::cot_clean": {"ppv": 0.5, "n": 10, "positives": 5}}
    adjusted = adjusted_trace_rates(rows, precision)
    assert len(adjusted) == 1
    assert adjusted[0].adjusted_rate == 0.01


def test_precision_by_stratum_groups_jobs():
    jobs = [
        {"span_id": "a", "stratum": "gsm8k::cot_clean"},
        {"span_id": "b", "stratum": "gsm8k::cot_clean"},
    ]
    verdicts = {"a": True, "b": False}
    out = precision_by_stratum(jobs, verdicts)
    assert out["gsm8k::cot_clean"]["n"] == 2
    assert out["gsm8k::cot_clean"]["positives"] == 1
    assert out["gsm8k::cot_clean"]["ppv"] == 0.5


def test_spans_from_rows_attaches_strata():
    rows = [_cot_row()]
    spans, strata = spans_from_rows(rows)
    assert spans
    assert all(strata[s.span_id].endswith("cot_clean") for s in spans)
