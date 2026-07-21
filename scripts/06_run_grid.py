#!/usr/bin/env python3
"""Full 6-condition × difficulty grid + optional size-control clean arms (§7.7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import (
    AIME_N,
    AIME_REPO,
    GSM8K_REPO,
    MATH500_REPO,
    SIZE_CONTROL_MODEL,
    load_settings,
)
from jspace.data import dataset_revision_meta, load_aime, load_gsm8k, load_math500
from jspace.load import load_hf_model, load_model_and_lens
from jspace.records import append_jsonl, write_run_meta
from jspace.run_one import run_problem
from jspace.token_budgets import calibration_path, load_token_budgets, summarize_profile


CONDITIONS = [
    ("direct_clean", False, "none", None),
    ("cot_clean", True, "none", None),
    ("direct_j_ablated", False, "jspace", None),
    ("cot_j_ablated", True, "jspace", None),
    ("direct_random", False, "random", 0),
    ("cot_random", True, "random", 0),
]


def load_cell(dataset: str, level: int | None, limit: int | None):
    if dataset == "gsm8k":
        return load_gsm8k(limit=limit)
    if dataset == "math500":
        return load_math500(level=level, limit=limit)
    if dataset == "aime":
        # AIME is capped at dataset size (§4.10).
        return load_aime(limit=None if limit is None else min(limit, AIME_N))
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="gsm8k,math500,aime",
        help="comma list: gsm8k,math500,aime",
    )
    parser.add_argument("--limit", type=int, default=None, help="override per-cell limit")
    parser.add_argument("--band-start", type=int, default=None)
    parser.add_argument("--band-end", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--size-control", action="store_true")
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--out-name", default="grid")
    args = parser.parse_args()

    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)
    k = args.k if args.k is not None else settings.k
    random_seeds = tuple(int(x) for x in args.random_seeds.split(",") if x.strip())

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

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cells: list[tuple[str, int | None, int | None]] = []
    for ds in datasets:
        if ds == "math500":
            for level in range(1, 6):
                lim = args.limit if args.limit is not None else settings.problems_per_cell
                cells.append((ds, level, lim))
        elif ds == "aime":
            cells.append((ds, None, args.limit))  # None → full 30
        else:
            lim = args.limit if args.limit is not None else settings.problems_per_cell
            cells.append((ds, None, lim))

    out_dir = settings.results_dir / args.out_name
    traces_path = out_dir / "traces.jsonl"
    if traces_path.exists():
        traces_path.unlink()

    token_profile = load_token_budgets(calibration_path(settings))
    meta = {
        "model_name": loaded.model_name,
        "model_revision": loaded.model_revision,
        "device": str(loaded.device),
        "dtype": str(loaded.dtype),
        "band_start": band_start,
        "band_end": band_end,
        "strength_label": "medium_equivalent",
        "k": k,
        "token_budgets": summarize_profile(token_profile) if token_profile else None,
        "datasets": datasets,
        "repos": {
            "gsm8k": GSM8K_REPO,
            "math500": MATH500_REPO,
            "aime": AIME_REPO,
            "aime_n": AIME_N,
        },
        "data_meta": dataset_revision_meta(),
        "lens_meta": loaded.lens_meta,
        "random_seeds": list(random_seeds),
    }
    write_run_meta(out_dir / "run_meta.json", meta)

    for dataset, level, limit in cells:
        problems = load_cell(dataset, level, limit)
        cell_name = f"{dataset}" + (f"_L{level}" if level is not None else "")
        print(f"=== cell {cell_name} n={len(problems)} ===")
        for problem in problems:
            for _cname, thinking, kind, _seed0 in CONDITIONS:
                seeds = random_seeds if kind == "random" else (settings.seed,)
                for rseed in seeds:
                    abl = AblationConfig(
                        kind=kind,  # type: ignore[arg-type]
                        band_start=band_start,
                        band_end=band_end,
                        k=k,
                        seed=rseed,
                        ablate_prompt_tokens=True,
                    )
                    rec = run_problem(
                        loaded,
                        problem,
                        settings=settings,
                        enable_thinking=thinking,
                        ablation=abl,
                        seed=settings.seed,
                        token_profile=token_profile,
                    )
                    rec.extra["cell"] = cell_name
                    rec.extra["random_seed"] = rseed
                    append_jsonl(traces_path, rec)
                    print(
                        f"{problem.problem_id} {rec.condition} rseed={rseed} "
                        f"correct={rec.correct} len={rec.trace_length_tokens}"
                    )

    if args.size_control:
        size_settings = settings.with_overrides(model_name=SIZE_CONTROL_MODEL)
        size_loaded = load_hf_model(size_settings)
        for dataset, level, limit in cells:
            problems = load_cell(dataset, level, limit)
            for problem in problems:
                for thinking in (False, True):
                    rec = run_problem(
                        size_loaded,
                        problem,
                        settings=size_settings,
                        enable_thinking=thinking,
                        ablation=AblationConfig(kind="none"),
                        seed=settings.seed,
                    )
                    rec.extra["cell"] = f"{dataset}" + (
                        f"_L{level}" if level is not None else ""
                    )
                    rec.extra["size_control"] = True
                    append_jsonl(out_dir / "size_control.jsonl", rec)

    print(f"wrote {traces_path}")


if __name__ == "__main__":
    main()
