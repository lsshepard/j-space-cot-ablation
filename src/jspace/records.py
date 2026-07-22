"""Per-problem logging schema (§4.9)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ProblemRecord:
    problem_id: str
    dataset: str
    difficulty: str | int | None
    condition: str
    seed: int
    model_name: str
    enable_thinking: bool
    ablation: str
    raw_trace: str
    extracted_answer: str | None
    gold_answer: str
    correct: bool | None
    graded_logprob: float | None
    trace_length_tokens: int
    hit_token_cap: bool
    extraction_success: bool
    backtrack_count: int | None = None
    backtrack_rate: float | None = None
    direct_leak_flag: bool = False
    token_logprobs: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: Path, records: Iterable[ProblemRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: ProblemRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_run_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_key(problem_id: str, condition: str, random_seed: int) -> tuple[str, str, int]:
    return (problem_id, condition, random_seed)


def completed_run_keys(path: Path) -> set[tuple[str, str, int]]:
    """Keys already present in a traces JSONL (for resume)."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    for row in read_jsonl(path):
        extra = row.get("extra") or {}
        rseed = extra.get("random_seed", row.get("seed", 0))
        done.add(run_key(row["problem_id"], row["condition"], int(rseed)))
    return done
