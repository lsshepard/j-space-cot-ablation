#!/usr/bin/env python3
"""Timing probe on expensive condition: CoT + J-ablation (§4.10 / §7.6)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import AIME_N, load_settings
from jspace.data import load_aime, load_gsm8k, load_math500
from jspace.load import load_model_and_lens
from jspace.run_one import run_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "aime"], default="aime")
    parser.add_argument("--band-start", type=int, default=None)
    parser.add_argument("--band-end", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)
    loaded = load_model_and_lens(settings)

    band_path = settings.results_dir / "calibration" / "band.json"
    if band_path.exists():
        band = json.loads(band_path.read_text(encoding="utf-8"))
        default_start, default_end = band["band_start"], band["band_end"]
    else:
        default_start, default_end = loaded.n_layers // 3, (2 * loaded.n_layers) // 3

    band_start = (
        args.band_start
        if args.band_start is not None
        else (settings.band_start if settings.band_start is not None else default_start)
    )
    band_end = (
        args.band_end
        if args.band_end is not None
        else (settings.band_end if settings.band_end is not None else default_end)
    )

    if args.dataset == "gsm8k":
        problems = load_gsm8k(limit=args.limit)
    elif args.dataset == "math500":
        problems = load_math500(limit=args.limit)
    else:
        problems = load_aime(limit=args.limit)

    abl = AblationConfig(
        kind="jspace",
        band_start=band_start,
        band_end=band_end,
        k=settings.k,
        ablate_prompt_tokens=True,
    )

    timings = []
    for problem in problems:
        t0 = time.perf_counter()
        rec = run_problem(
            loaded,
            problem,
            settings=settings,
            enable_thinking=True,
            ablation=abl,
            seed=settings.seed,
        )
        elapsed = time.perf_counter() - t0
        timings.append(
            {
                "problem_id": problem.problem_id,
                "seconds": elapsed,
                "tokens": rec.trace_length_tokens,
                "hit_cap": rec.hit_token_cap,
            }
        )
        print(
            f"{problem.problem_id}: {elapsed:.1f}s tokens={rec.trace_length_tokens} "
            f"hooks={rec.extra.get('hook_call_count')}"
        )

    mean_s = sum(t["seconds"] for t in timings) / max(len(timings), 1)
    # 6 conditions × cells: GSM8K + 5 MATH levels + AIME ≈ 7 cells; plus size control.
    conditions = 6
    cells_main = 1 + 5 + 1
    sec_per_problem_all_conds = mean_s * conditions
    recommend_gsm_math = settings.problems_per_cell
    hours_gsm_math = (
        sec_per_problem_all_conds * recommend_gsm_math * (1 + 5) / 3600.0
    )
    hours_aime = sec_per_problem_all_conds * AIME_N / 3600.0

    summary = {
        "dataset": args.dataset,
        "n_probe": len(timings),
        "mean_seconds_cot_j_ablated": mean_s,
        "recommended_problems_per_cell_gsm_math": recommend_gsm_math,
        "aime_n": AIME_N,
        "est_gpu_hours_gsm_math_6conds": hours_gsm_math,
        "est_gpu_hours_aime_6conds": hours_aime,
        "band_start": band_start,
        "band_end": band_end,
        "note": "Estimates exclude size-control and calibration overhead.",
        "timings": timings,
    }
    out = settings.results_dir / "calibration" / "timing_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "timings"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
