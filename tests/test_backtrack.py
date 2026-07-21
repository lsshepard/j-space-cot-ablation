from jspace.backtrack import (
    find_candidate_spans,
    normalized_backtrack_rate,
    regex_backtrack_count,
)


def test_finds_marker_spans():
    trace = "I add 3. Wait, actually I should multiply instead. So 3*4=12."
    spans = find_candidate_spans(trace, trace_id="t1")
    assert len(spans) >= 1
    assert any("wait" in s.marker.lower() or "actually" in s.marker.lower() for s in spans)


def test_degenerate_wait_loop_merges():
    trace = "wait, wait, wait I am stuck"
    count = regex_backtrack_count(trace)
    assert count == 1


def test_normalized_rate():
    assert normalized_backtrack_rate(2, 100) == 0.02
    assert normalized_backtrack_rate(1, 0) == 0.0
