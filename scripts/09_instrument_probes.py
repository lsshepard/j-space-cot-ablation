#!/usr/bin/env python3
"""Instrument probes 1–3: survival/‖Δh‖, exclude_topk A/B, gold logprob bite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings
from jspace.data import load_multihop_fixture
from jspace.diagnostics import run_instrument_suite
from jspace.load import load_model_and_lens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "lens-eval-multihop-easy.json",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--band-start", type=int, default=27)
    parser.add_argument("--band-end", type=int, default=31)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exclude-topk", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <run_dir>/diagnostics/",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)

    loaded = load_model_and_lens(settings)
    problems = load_multihop_fixture(args.fixture)
    if args.limit is not None:
        problems = problems[: args.limit]

    print(
        f"instrument suite n={len(problems)} band=[{args.band_start},{args.band_end}] "
        f"k={args.k} exclude_topk={args.exclude_topk} vs 0"
    )
    result = run_instrument_suite(
        loaded,
        problems,
        settings=settings,
        band_start=args.band_start,
        band_end=args.band_end,
        k=args.k,
        exclude_topk=args.exclude_topk,
        max_new_tokens=args.max_new_tokens,
    )

    out_dir = args.out_dir or (settings.results_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "instrument_probes_1_3.json"
    payload = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    slim = {
        "n": result.n,
        "band_start": result.band_start,
        "band_end": result.band_end,
        "k": result.k,
        "verdicts": result.verdicts,
        "survival_exclude": result.survival_exclude.to_dict(),
        "survival_no_exclude": result.survival_no_exclude.to_dict(),
        "exclude_ab": result.exclude_ab.to_dict(),
        "no_exclude_ab": result.no_exclude_ab.to_dict(),
        "logprob_bite_exclude": result.logprob_bite_exclude.to_dict(),
        "logprob_bite_no_exclude": result.logprob_bite_no_exclude.to_dict(),
    }
    print(json.dumps(slim, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
