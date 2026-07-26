#!/usr/bin/env python3
"""Screen clean direct accuracy to find a 20–80% instrument set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import load_settings
from jspace.data import load_gsm8k, load_math500, load_multihop_fixture
from jspace.load import load_model_and_lens
from jspace.run_one import run_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("easy_multihop", "gsm8k", "math500"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--level", type=int, default=None, help="MATH-500 level filter")
    parser.add_argument("--thinking", action="store_true", help="CoT instead of direct")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)
    loaded = load_model_and_lens(settings)

    if args.dataset == "easy_multihop":
        problems = load_multihop_fixture(
            ROOT / "tests" / "fixtures" / "lens-eval-multihop-easy.json"
        )
    elif args.dataset == "gsm8k":
        problems = load_gsm8k(limit=args.limit)
    else:
        problems = load_math500(level=args.level, limit=args.limit)
    problems = problems[: args.limit]

    flags: list[bool] = []
    gold_lps: list[float] = []
    for i, problem in enumerate(problems):
        rec = run_problem(
            loaded,
            problem,
            settings=settings,
            enable_thinking=args.thinking,
            ablation=AblationConfig(kind="none"),
            seed=settings.seed,
        )
        ok = bool(rec.correct)
        flags.append(ok)
        if rec.graded_logprob is not None:
            gold_lps.append(float(rec.graded_logprob))
        print(
            f"[{i + 1}/{len(problems)}] {problem.problem_id} "
            f"correct={ok} ans={rec.extracted_answer!r} gold={problem.gold_answer!r} "
            f"lp={rec.graded_logprob}",
            flush=True,
        )

    acc = sum(flags) / len(flags) if flags else float("nan")
    mean_lp = sum(gold_lps) / len(gold_lps) if gold_lps else float("nan")
    summary = {
        "dataset": args.dataset,
        "thinking": args.thinking,
        "level": args.level,
        "n": len(flags),
        "clean_accuracy": acc,
        "mean_clean_gold_lp": mean_lp,
        "in_target_band_20_80": 0.2 <= acc <= 0.8,
    }
    out = settings.results_dir / "diagnostics" / f"screen_clean_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
