import json
from pathlib import Path

from jspace.records import completed_run_keys, run_key


def test_completed_run_keys_resume(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    rows = [
        {
            "problem_id": "p0",
            "condition": "direct_clean",
            "seed": 0,
            "extra": {"random_seed": 0},
        },
        {
            "problem_id": "p0",
            "condition": "cot_random",
            "seed": 0,
            "extra": {"random_seed": 2},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    done = completed_run_keys(path)
    assert run_key("p0", "direct_clean", 0) in done
    assert run_key("p0", "cot_random", 2) in done
    assert run_key("p0", "cot_random", 0) not in done
    assert completed_run_keys(tmp_path / "missing.jsonl") == set()
