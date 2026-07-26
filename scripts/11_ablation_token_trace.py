#!/usr/bin/env python3
"""Qualitative ablation token traces: excluded vs survivor J-decoded tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings
from jspace.data import load_multihop_fixture
from jspace.diagnostics.token_trace import (
    format_problem_trace_report,
    run_token_traces,
    summarize_traces,
)
from jspace.load import load_model_and_lens

# Confident easy-multihop items spanning strong J bite → null → J "helps"
# (from results/diagnostics/gold_lp_triad_multihop.json, clean_lp ≳ −0.5).
DEFAULT_IDS = (
    "super-populous-capital",
    "amazon-language",
    "super-smallest-continent",
    "atomic-26-symbol",
    "colosseum-currency",
    "planet-3-moons",
    "carnival-ocean",
    "atomic-29-symbol",
    "paper-continent",
    "spider-legs",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "lens-eval-multihop-easy.json",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=",".join(DEFAULT_IDS),
        help="comma-separated problem ids (empty = all / --limit)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--band-start", type=int, default=27)
    parser.add_argument("--band-end", type=int, default=33)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exclude-topk", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <run_dir>/diagnostics/",
    )
    args = parser.parse_args()

    settings = load_settings()
    loaded = load_model_and_lens(settings)
    problems = load_multihop_fixture(args.fixture)
    by_id = {p.problem_id: p for p in problems}

    id_list = [x.strip() for x in args.ids.split(",") if x.strip()]
    if id_list:
        missing = [i for i in id_list if i not in by_id]
        if missing:
            raise SystemExit(f"unknown problem ids: {missing}")
        selected = [by_id[i] for i in id_list]
    else:
        selected = problems
        if args.limit is not None:
            selected = selected[: args.limit]

    print(
        f"token traces n={len(selected)} band=[{args.band_start},{args.band_end}] "
        f"k={args.k} exclude_topk={args.exclude_topk}",
        flush=True,
    )
    traces = run_token_traces(
        loaded,
        selected,
        band_start=args.band_start,
        band_end=args.band_end,
        k=args.k,
        exclude_topk=args.exclude_topk,
        max_new_tokens=args.max_new_tokens,
        seed=settings.seed,
    )

    out_dir = args.out_dir or (settings.results_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_traces(traces)
    payload = {
        "band_start": args.band_start,
        "band_end": args.band_end,
        "k": args.k,
        "exclude_topk": args.exclude_topk,
        "fixture": str(args.fixture),
        "ids": [t.problem_id for t in traces],
        "summary": summary,
        "traces": [t.to_dict() for t in traces],
    }
    out_json = out_dir / "ablation_token_traces.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Ablation token-trace qualitative report",
        "",
        f"band=[{args.band_start},{args.band_end}] k={args.k} "
        f"exclude_topk={args.exclude_topk} n={len(traces)}",
        "",
        "## Cross-problem summary",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Per-problem traces",
        "",
    ]
    for tr in traces:
        report_lines.append(format_problem_trace_report(tr))
        report_lines.append("")
    out_md = out_dir / "ablation_token_traces.md"
    out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
