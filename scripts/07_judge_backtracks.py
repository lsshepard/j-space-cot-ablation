#!/usr/bin/env python3
"""Emit backtrack span jobs; collect verdicts; report agreement (§4.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.backtrack import find_candidate_spans
from jspace.config import load_settings
from jspace.judge import (
    agreement_rate,
    read_verdicts,
    regex_labels_for_spans,
    write_span_jobs,
)
from jspace.records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="existing verdicts JSONL to score; if omitted, only emit span jobs",
    )
    parser.add_argument(
        "--hand-labels",
        type=Path,
        default=None,
        help="optional hand labels JSONL: {span_id, is_backtrack}",
    )
    parser.add_argument("--cot-only", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    rows = read_jsonl(args.traces)
    spans = []
    for row in rows:
        if args.cot_only and not row.get("enable_thinking", True):
            continue
        if row.get("ablation") not in {None, "none", "jspace", "random"}:
            continue
        # Prefer CoT arms for mechanism metric.
        if not row.get("enable_thinking"):
            continue
        trace_id = f"{row['problem_id']}::{row['condition']}"
        spans.extend(find_candidate_spans(row["raw_trace"], trace_id=trace_id))

    out_dir = settings.results_dir / "judge"
    jobs_path = out_dir / "span_jobs.jsonl"
    write_span_jobs(jobs_path, spans)
    print(f"wrote {len(spans)} span jobs → {jobs_path}")
    print(
        "Fill verdicts with an LLM subagent using each job's frozen `prompt` field; "
        "write JSONL with parseable {span_id,is_backtrack,approach_before,approach_after}."
    )

    template_hand = out_dir / "hand_label_template.jsonl"
    with template_hand.open("w", encoding="utf-8") as f:
        for span in spans[:50]:
            f.write(
                json.dumps(
                    {
                        "span_id": span.span_id,
                        "text": span.text,
                        "is_backtrack": None,
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote hand-label template (up to 50) → {template_hand}")

    if args.verdicts is None:
        return

    verdicts = read_verdicts(args.verdicts)
    # Also accept raw model strings via a sidecar if needed — parse_verdict available.
    judge_labels = {v.span_id: v.is_backtrack for v in verdicts}
    regex_labels = regex_labels_for_spans(spans)
    regex_judge = agreement_rate(regex_labels, judge_labels)

    report = {
        "n_spans": len(spans),
        "n_verdicts": len(verdicts),
        "regex_judge_agreement": regex_judge,
        "note": (
            "Regex labels all marker spans True; judge applies effective-backtrack "
            "criterion. Hand↔judge agreement reported when --hand-labels provided."
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

    report_path = out_dir / "agreement.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
