from pathlib import Path

from jspace.records import ProblemRecord, read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path: Path):
    rec = ProblemRecord(
        problem_id="p1",
        dataset="gsm8k",
        difficulty="easy",
        condition="cot_clean",
        seed=0,
        model_name="x",
        enable_thinking=True,
        ablation="none",
        raw_trace="hi",
        extracted_answer="1",
        gold_answer="1",
        correct=True,
        graded_logprob=-0.1,
        trace_length_tokens=3,
        hit_token_cap=False,
        extraction_success=True,
        token_logprobs=[-0.1, -0.2],
    )
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [rec])
    rows = read_jsonl(path)
    assert rows[0]["problem_id"] == "p1"
    assert rows[0]["token_logprobs"] == [-0.1, -0.2]
