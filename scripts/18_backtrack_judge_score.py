#!/usr/bin/env python3
"""Merge span verdicts and report regex↔judge (and optional hand↔judge) agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.backtrack import BacktrackSpan
from jspace.judge import (
    agreement_rate,
    read_verdicts,
    regex_labels_for_spans,
    write_verdicts,
)
from jspace.records import read_jsonl


def load_jobs(path: Path) -> list[dict]:
    return read_jsonl(path)


def spans_for_jobs(jobs: list[dict]) -> list[BacktrackSpan]:
    return [
        BacktrackSpan(
            span_id=str(job["span_id"]),
            trace_id=str(job.get("trace_id", "")),
            start=0,
            end=len(job.get("text", "")),
            marker=str(job.get("marker", "")),
            text=str(job.get("text", "")),
        )
        for job in jobs
    ]


def merge_verdict_paths(paths: list[Path]):
    merged = []
    seen: set[str] = set()
    for path in paths:
        for verdict in read_verdicts(path):
            if verdict.span_id in seen:
                continue
            seen.add(verdict.span_id)
            merged.append(verdict)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument(
        "--verdicts",
        type=Path,
        nargs="+",
        required=True,
        help="one or more verdict JSONL files to merge",
    )
    parser.add_argument("--hand-labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    jobs = load_jobs(args.jobs)
    spans = spans_for_jobs(jobs)
    job_ids = {str(job["span_id"]) for job in jobs}
    verdicts = merge_verdict_paths(args.verdicts)
    verdicts = [v for v in verdicts if v.span_id in job_ids]

    judge_labels = {v.span_id: v.is_backtrack for v in verdicts}
    regex_labels = regex_labels_for_spans(spans)
    report = {
        "n_jobs": len(jobs),
        "n_verdicts": len(verdicts),
        "regex_judge_agreement": agreement_rate(regex_labels, judge_labels),
        "judge_positive_rate": (
            sum(judge_labels.values()) / len(judge_labels) if judge_labels else float("nan")
        ),
    }

    if args.hand_labels is not None:
        hand = {}
        for row in read_jsonl(args.hand_labels):
            if row.get("is_backtrack") is None:
                continue
            hand[row["span_id"]] = bool(row["is_backtrack"])
        report["hand_judge_agreement"] = agreement_rate(hand, judge_labels)
        report["n_hand_labels"] = len(hand)

    out = args.out or args.jobs.parent / "judge_agreement.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")

    merged_path = args.jobs.parent / "judge_verdicts.jsonl"
    write_verdicts(merged_path, verdicts)
    print(f"merged verdicts → {merged_path}")


if __name__ == "__main__":
    main()
