"""Frozen-prompt span judge I/O and agreement helpers (§4.4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from jspace.backtrack import BacktrackSpan
from jspace.config import RUBRICS_DIR


JUDGE_PROMPT_PATH = RUBRICS_DIR / "judge_prompt.md"


@dataclass(frozen=True)
class JudgeVerdict:
    span_id: str
    is_backtrack: bool
    approach_before: str
    approach_after: str
    source: str = "subagent"


def load_judge_prompt_template(path: Path | None = None) -> str:
    return (path or JUDGE_PROMPT_PATH).read_text(encoding="utf-8")


def assemble_judge_prompt(span_text: str, template: str | None = None) -> str:
    tmpl = template if template is not None else load_judge_prompt_template()
    if "{{SPAN}}" not in tmpl:
        raise ValueError("judge prompt template missing {{SPAN}} placeholder")
    return tmpl.replace("{{SPAN}}", span_text)


def parse_verdict(span_id: str, raw: str | dict[str, Any], *, source: str = "subagent") -> JudgeVerdict:
    if isinstance(raw, str):
        text = raw.strip()
        # Allow fenced JSON.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        payload = json.loads(text)
    else:
        payload = raw

    required = {"is_backtrack", "approach_before", "approach_after"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"verdict missing keys {sorted(missing)}")
    if not isinstance(payload["is_backtrack"], bool):
        raise ValueError("is_backtrack must be bool")
    return JudgeVerdict(
        span_id=span_id,
        is_backtrack=bool(payload["is_backtrack"]),
        approach_before=str(payload["approach_before"]),
        approach_after=str(payload["approach_after"]),
        source=source,
    )


def write_span_jobs(path: Path, spans: Iterable[BacktrackSpan], template: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpl = template if template is not None else load_judge_prompt_template()
    with path.open("w", encoding="utf-8") as f:
        for span in spans:
            job = {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "marker": span.marker,
                "text": span.text,
                "prompt": assemble_judge_prompt(span.text, tmpl),
            }
            f.write(json.dumps(job, ensure_ascii=False) + "\n")


def write_verdicts(path: Path, verdicts: Iterable[JudgeVerdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for verdict in verdicts:
            f.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")


def read_verdicts(path: Path) -> list[JudgeVerdict]:
    rows: list[JudgeVerdict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                JudgeVerdict(
                    span_id=payload["span_id"],
                    is_backtrack=bool(payload["is_backtrack"]),
                    approach_before=str(payload["approach_before"]),
                    approach_after=str(payload["approach_after"]),
                    source=str(payload.get("source", "subagent")),
                )
            )
    return rows


def agreement_rate(a: dict[str, bool], b: dict[str, bool]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    agree = sum(1 for k in keys if a[k] == b[k])
    return agree / len(keys)


def regex_labels_for_spans(spans: Iterable[BacktrackSpan]) -> dict[str, bool]:
    """Regex headline treats every candidate span as a positive candidate."""
    return {span.span_id: True for span in spans}
