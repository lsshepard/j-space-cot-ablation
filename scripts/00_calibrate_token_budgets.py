#!/usr/bin/env python3
"""Calibrate per-dataset direct/CoT token caps from unablated traces (§4.7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings
from jspace.load import load_hf_model
from jspace.token_budgets import (
    calibration_path,
    calibrate_token_budgets,
    load_token_budgets,
    save_token_budgets,
    summarize_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="gsm8k",
        help="comma-separated: gsm8k,math500,aime",
    )
    parser.add_argument("--problems-per-dataset", type=int, default=5)
    parser.add_argument("--multiplier", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--ceiling", type=int, default=None)
    parser.add_argument(
        "--probe-ceiling",
        type=int,
        default=None,
        help="max_new_tokens while measuring lengths (default: settings probe ceiling)",
    )
    args = parser.parse_args()

    settings = load_settings()
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    loaded = load_hf_model(settings)

    profile = calibrate_token_budgets(
        loaded,
        settings,
        datasets=datasets,
        problems_per_dataset=args.problems_per_dataset,
        percentile=args.percentile,
        multiplier=args.multiplier,
        ceiling=args.ceiling,
        probe_ceiling=args.probe_ceiling,
    )
    out = calibration_path(settings)
    save_token_budgets(out, profile)
    summary = summarize_profile(profile)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
