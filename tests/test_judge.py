import json
from pathlib import Path

from jspace.backtrack import BacktrackSpan
from jspace.judge import (
    agreement_rate,
    assemble_judge_prompt,
    parse_verdict,
    write_span_jobs,
    write_verdicts,
    JudgeVerdict,
)


def test_assemble_prompt_inserts_span(tmp_path: Path):
    template = "HEADER\n{{SPAN}}\nFOOTER"
    prompt = assemble_judge_prompt("hello span", template)
    assert "hello span" in prompt
    assert "{{SPAN}}" not in prompt


def test_parse_verdict_json():
    raw = '{"is_backtrack": true, "approach_before": "add", "approach_after": "multiply"}'
    v = parse_verdict("s1", raw)
    assert v.is_backtrack is True
    assert v.approach_before == "add"


def test_parse_verdict_rejects_missing():
    try:
        parse_verdict("s1", '{"is_backtrack": true}')
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_agreement_and_jsonl_roundtrip(tmp_path: Path):
    spans = [
        BacktrackSpan("s1", "t", 0, 10, "wait", "wait, redo"),
        BacktrackSpan("s2", "t", 11, 20, "actually", "actually same plan"),
    ]
    jobs = tmp_path / "jobs.jsonl"
    write_span_jobs(jobs, spans, template="P {{SPAN}}")
    assert jobs.exists()
    line = jobs.read_text(encoding="utf-8").splitlines()[0]
    assert "wait, redo" in line

    verdicts = [
        JudgeVerdict("s1", True, "a", "b"),
        JudgeVerdict("s2", False, "a", "a"),
    ]
    vpath = tmp_path / "verdicts.jsonl"
    write_verdicts(vpath, verdicts)
    loaded = [json.loads(x) for x in vpath.read_text().splitlines()]
    assert loaded[0]["is_backtrack"] is True

    a = {"s1": True, "s2": True}
    b = {"s1": True, "s2": False}
    assert agreement_rate(a, b) == 0.5
