#!/usr/bin/env python3
"""Clean / J / random teacher-forced gold logprob triad (n≈30 instrument gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings
from jspace.data import load_gsm8k, load_math500, load_multihop_fixture
from jspace.diagnostics.gold_lp_triad import run_gold_lp_triad
from jspace.load import load_model_and_lens


def _load_problems(args: argparse.Namespace):
    if args.dataset == "multihop":
        problems = load_multihop_fixture(args.fixture)
    elif args.dataset == "gsm8k":
        problems = load_gsm8k(limit=args.limit)
    elif args.dataset == "math500":
        problems = load_math500(level=args.level, limit=args.limit)
    else:
        raise ValueError(args.dataset)
    if args.limit is not None:
        problems = problems[: args.limit]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("multihop", "gsm8k", "math500"),
        default="gsm8k",
        help="prefer a set where clean direct accuracy is ~20–80%",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "lens-eval-multihop-easy.json",
        help="multihop fixture path (when --dataset multihop)",
    )
    parser.add_argument("--level", type=int, default=None, help="MATH-500 level filter")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--band-start", type=int, default=27)
    parser.add_argument("--band-end", type=int, default=31)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exclude-topk", type=int, default=10)
    parser.add_argument(
        "--random-seeds",
        type=str,
        default="0,1,2",
        help="comma-separated random-ablation seeds",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <run_dir>/diagnostics/",
    )
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.random_seeds.split(",") if x.strip())
    settings = load_settings()
    loaded = load_model_and_lens(settings)
    problems = _load_problems(args)

    print(
        f"gold-lp triad dataset={args.dataset} n={len(problems)} "
        f"band=[{args.band_start},{args.band_end}] "
        f"k={args.k} exclude_topk={args.exclude_topk} random_seeds={seeds}",
        flush=True,
    )
    result = run_gold_lp_triad(
        loaded,
        problems,
        settings=settings,
        band_start=args.band_start,
        band_end=args.band_end,
        k=args.k,
        exclude_topk=args.exclude_topk,
        random_seeds=seeds,
        n_boot=args.n_boot,
    )

    out_dir = args.out_dir or (settings.results_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gold_lp_triad_{args.dataset}.json"
    payload = result.to_dict()
    payload["dataset"] = args.dataset
    payload["level"] = args.level
    if args.dataset == "multihop":
        payload["fixture"] = str(args.fixture)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    slim = {
        "dataset": args.dataset,
        "level": args.level,
        "n": result.n,
        "band_start": result.band_start,
        "band_end": result.band_end,
        "k": result.k,
        "exclude_topk": result.exclude_topk,
        "random_seeds": result.random_seeds,
        "mean_clean_lp": result.mean_clean_lp,
        "mean_jspace_lp": result.mean_jspace_lp,
        "mean_random_lp": result.mean_random_lp,
        "delta_j": result.delta_j.to_dict(),
        "delta_r": result.delta_r.to_dict(),
        "delta_j_minus_r": result.delta_j_minus_r.to_dict(),
        "verdict": result.verdict,
    }
    print(json.dumps(slim, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
