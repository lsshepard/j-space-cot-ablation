#!/usr/bin/env python3
"""Harness smoke: load model, generate both modes on GSM8K, extract, score (§7.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import load_settings
from jspace.data import load_gsm8k
from jspace.load import load_hf_model
from jspace.records import write_jsonl
from jspace.run_one import run_problem
from jspace.token_budgets import (
    calibration_path,
    calibrate_token_budgets,
    load_token_budgets,
    resolve_max_new_tokens,
    save_token_budgets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="re-measure unablated lengths and write token_budgets.json",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="debug override: same cap for every arm",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)

    loaded = load_hf_model(settings)
    profile = load_token_budgets(calibration_path(settings))

    if args.calibrate or profile is None:
        profile = calibrate_token_budgets(
            loaded,
            settings,
            datasets=("gsm8k",),
            problems_per_dataset=min(args.limit, 5),
        )
        save_token_budgets(calibration_path(settings), profile)
        print(f"calibrated token budgets → {calibration_path(settings)}")

    problems = load_gsm8k(limit=args.limit)
    records = []
    for problem in problems:
        for thinking in (False, True):
            budget = resolve_max_new_tokens(
                settings, problem.dataset, enable_thinking=thinking, profile=profile
            )
            rec = run_problem(
                loaded,
                problem,
                settings=settings,
                enable_thinking=thinking,
                ablation=AblationConfig(kind="none"),
                seed=settings.seed,
                token_profile=profile,
            )
            records.append(rec)
            print(
                f"{rec.problem_id} thinking={thinking} correct={rec.correct} "
                f"extract_ok={rec.extraction_success} cap={rec.hit_token_cap} "
                f"len={rec.trace_length_tokens}/{budget}"
            )

    out = settings.results_dir / "harness" / "smoke.jsonl"
    write_jsonl(out, records)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
