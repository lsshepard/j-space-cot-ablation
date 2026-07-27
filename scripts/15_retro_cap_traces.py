#!/usr/bin/env python3
"""Counterfactual token ceiling: truncate long traces and rescore in place.

For rows with ``trace_length_tokens > --cap``, retokenize ``raw_trace``, keep the
first ``cap`` tokens, re-extract the answer, and rescore. Sets ``hit_token_cap``
and records originals under ``extra.retro_*``. Does **not** blanket-mark long
traces as incorrect (answer may already be present before the cap).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.extract import extract_answer
from jspace.metrics import score_prediction


def _truncate_row(row: dict, cap: int, tokenizer) -> tuple[dict, bool]:
    """Return (row, changed)."""
    old_len = int(row.get("trace_length_tokens") or 0)
    if old_len <= cap:
        return row, False

    text = row.get("raw_trace") or ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    trunc_ids = ids[:cap]
    new_text = tokenizer.decode(trunc_ids, skip_special_tokens=True)
    extraction = extract_answer(row["dataset"], new_text)
    new_correct = score_prediction(
        row["dataset"], extraction.answer, row["gold_answer"]
    )

    extra = dict(row.get("extra") or {})
    extra["retro_cap_tokens"] = cap
    extra["original_trace_length_tokens"] = old_len
    extra["original_correct"] = bool(row.get("correct"))
    extra["original_extracted_answer"] = row.get("extracted_answer")
    extra["original_hit_token_cap"] = bool(row.get("hit_token_cap"))
    extra["original_raw_trace_chars"] = len(text)

    logprobs = row.get("token_logprobs")
    if isinstance(logprobs, list) and len(logprobs) > cap:
        row["token_logprobs"] = logprobs[:cap]

    row["raw_trace"] = new_text
    row["trace_length_tokens"] = len(trunc_ids)
    row["extracted_answer"] = extraction.answer
    row["extraction_success"] = bool(extraction.success)
    row["correct"] = new_correct
    row["hit_token_cap"] = True
    row["extra"] = extra
    return row, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="rewrite traces (writes .bak first)",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-4B",
        help="HF tokenizer id used to retokenize raw_trace",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rows: list[dict] = []
    changed = 0
    flipped_correct = 0
    for line in args.traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        old_correct = bool(row.get("correct"))
        row, did = _truncate_row(row, args.cap, tokenizer)
        if did:
            changed += 1
            if old_correct != bool(row.get("correct")):
                flipped_correct += 1
                print(
                    f"FLIP {row['problem_id']} {row['condition']}: "
                    f"correct {old_correct} -> {row['correct']} "
                    f"(orig_len={row['extra']['original_trace_length_tokens']})"
                )
            else:
                print(
                    f"CAP  {row['problem_id']} {row['condition']}: "
                    f"len {row['extra']['original_trace_length_tokens']} -> "
                    f"{row['trace_length_tokens']} correct={row['correct']}"
                )
        rows.append(row)

    print(
        f"n={len(rows)} capped={changed} correct_flips={flipped_correct} "
        f"cap={args.cap}"
    )
    if args.inplace:
        bak = args.traces.with_suffix(args.traces.suffix + ".pre_retro_cap.bak")
        if not bak.exists():
            shutil.copy2(args.traces, bak)
            print(f"backup {bak}")
        tmp = args.traces.with_suffix(args.traces.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(args.traces)
        print(f"wrote {args.traces}")


if __name__ == "__main__":
    main()
