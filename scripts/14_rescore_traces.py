#!/usr/bin/env python3
"""Recompute ``correct`` on an existing traces.jsonl (no regeneration)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.data import load_aime, load_gsm8k, load_math500
from jspace.metrics import score_prediction


def _prompt_index() -> dict[str, str]:
    """problem_id → source prompt (for backfilling older rows)."""
    out: dict[str, str] = {}
    for p in load_gsm8k(limit=None):
        out[p.problem_id] = p.prompt
    for level in range(1, 6):
        for p in load_math500(level=level, limit=None):
            out[p.problem_id] = p.prompt
    for p in load_aime(limit=None):
        out[p.problem_id] = p.prompt
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="path to traces.jsonl")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="rewrite the file in place (default: print summary only)",
    )
    parser.add_argument(
        "--backfill-prompt",
        action="store_true",
        help="fill extra.prompt from the dataset when missing",
    )
    args = parser.parse_args()

    prompts = _prompt_index() if args.backfill_prompt else {}
    rows: list[dict] = []
    flipped = 0
    true_pos = 0
    prompts_filled = 0
    for line in args.traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        old = bool(row.get("correct"))
        new = score_prediction(
            row["dataset"], row.get("extracted_answer"), row["gold_answer"]
        )
        if old != new:
            flipped += 1
        if new:
            true_pos += 1
        row["correct"] = new
        if args.backfill_prompt:
            extra = dict(row.get("extra") or {})
            if not extra.get("prompt"):
                prompt = prompts.get(row["problem_id"])
                if prompt is not None:
                    extra["prompt"] = prompt
                    row["extra"] = extra
                    prompts_filled += 1
        rows.append(row)

    print(
        f"n={len(rows)} flipped={flipped} correct_now={true_pos} "
        f"em={true_pos / len(rows) if rows else float('nan'):.3f} "
        f"prompts_filled={prompts_filled}"
    )
    if args.inplace:
        tmp = args.traces.with_suffix(args.traces.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(args.traces)
        print(f"rewrote {args.traces}")


if __name__ == "__main__":
    main()
