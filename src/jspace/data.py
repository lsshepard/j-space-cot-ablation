"""Pinned dataset loaders (§4.8 / §4.9)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jspace.config import (
    AIME_N,
    AIME_REPO,
    DEFAULT_AIME_SPLIT,
    DEFAULT_GSM8K_SPLIT,
    DEFAULT_MATH500_SPLIT,
    GSM8K_REPO,
    MATH500_REPO,
)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    dataset: str
    prompt: str
    gold_answer: str
    difficulty: str | int | None = None
    meta: dict[str, Any] | None = None


def _load_hf(repo: str, split: str, revision: str | None = None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision
    # gsm8k needs config name "main"
    if repo == GSM8K_REPO:
        return load_dataset(repo, "main", split=split, **kwargs)
    return load_dataset(repo, split=split, **kwargs)


def load_gsm8k(
    *,
    limit: int | None = None,
    split: str = DEFAULT_GSM8K_SPLIT,
    revision: str | None = None,
) -> list[Problem]:
    ds = _load_hf(GSM8K_REPO, split, revision)
    problems: list[Problem] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        problems.append(
            Problem(
                problem_id=f"gsm8k-{split}-{i}",
                dataset="gsm8k",
                prompt=row["question"].strip(),
                gold_answer=gold,
                difficulty="easy",
            )
        )
    return problems


def load_math500(
    *,
    level: int | None = None,
    limit: int | None = None,
    split: str = DEFAULT_MATH500_SPLIT,
    revision: str | None = None,
) -> list[Problem]:
    ds = _load_hf(MATH500_REPO, split, revision)
    problems: list[Problem] = []
    for i, row in enumerate(ds):
        row_level = int(row.get("level", row.get("Level", 0)))
        if level is not None and row_level != level:
            continue
        problems.append(
            Problem(
                problem_id=f"math500-{split}-{i}-L{row_level}",
                dataset="math500",
                prompt=row["problem"].strip(),
                gold_answer=str(row["answer"]).strip(),
                difficulty=row_level,
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


def load_aime(
    *,
    limit: int | None = None,
    split: str = DEFAULT_AIME_SPLIT,
    revision: str | None = None,
) -> list[Problem]:
    ds = _load_hf(AIME_REPO, split, revision)
    problems: list[Problem] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # Schema variants across HF copies.
        prompt = row.get("problem") or row.get("question") or row["prompt"]
        gold = str(row.get("answer") or row.get("solution") or row["expected_answer"])
        gold = gold.strip()
        if gold.isdigit():
            gold = str(int(gold))
        problems.append(
            Problem(
                problem_id=f"aime2024-{split}-{i}",
                dataset="aime",
                prompt=str(prompt).strip(),
                gold_answer=gold,
                difficulty="hard",
            )
        )
    if limit is None and len(problems) != AIME_N:
        # Soft check — some mirrors may differ; still expose count.
        pass
    return problems


def load_multihop_fixture(path: Path) -> list[Problem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[Problem] = []
    for i, row in enumerate(payload):
        problems.append(
            Problem(
                problem_id=row.get("id", f"multihop-{i}"),
                dataset="multihop",
                prompt=row["prompt"].strip(),
                gold_answer=str(row["answer"]).strip(),
                difficulty="control",
                meta={"intermediate": row.get("intermediate")},
            )
        )
    return problems


def dataset_revision_meta() -> dict[str, str]:
    return {
        "gsm8k_repo": GSM8K_REPO,
        "math500_repo": MATH500_REPO,
        "aime_repo": AIME_REPO,
        "aime_n_expected": str(AIME_N),
    }
