#!/usr/bin/env python3
"""Multi-hop positive control: clean / J / random EM gate (paper §3.5.2 / §7.5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import load_settings
from jspace.data import load_multihop_fixture
from jspace.load import load_model_and_lens
from jspace.metrics import accuracy
from jspace.records import write_jsonl
from jspace.run_one import run_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "lens-eval-multihop-easy.json",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="comma-separated problem ids (overrides --limit order)",
    )
    parser.add_argument(
        "--require-clean-correct",
        action="store_true",
        help="eval J/random only on items clean got right (near-ceiling set)",
    )
    parser.add_argument("--band-start", type=int, default=27)
    parser.add_argument("--band-end", type=int, default=31)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exclude-topk", type=int, default=10)
    parser.add_argument("--random-seeds", type=str, default="0,1,2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-j-drop", type=float, default=0.3)
    parser.add_argument(
        "--min-j-over-random",
        type=float,
        default=0.1,
        help="minimum (random_acc − J_acc); J must hurt more than random",
    )
    parser.add_argument("--out-name", type=str, default="multihop_em_gate")
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.random_seeds.split(",") if x.strip())
    settings = load_settings()
    if args.max_new_tokens is not None:
        settings = settings.with_overrides(max_new_tokens_override=args.max_new_tokens)
    loaded = load_model_and_lens(settings)

    problems = load_multihop_fixture(args.fixture)
    id_list = [x.strip() for x in args.ids.split(",") if x.strip()]
    if id_list:
        by_id = {p.problem_id: p for p in problems}
        missing = [i for i in id_list if i not in by_id]
        if missing:
            raise SystemExit(f"unknown ids: {missing}")
        problems = [by_id[i] for i in id_list]
    elif args.limit is not None:
        problems = problems[: args.limit]

    base = dict(
        band_start=args.band_start,
        band_end=args.band_end,
        k=args.k,
        exclude_topk=args.exclude_topk,
        ablate_prompt_tokens=True,
    )

    print(
        f"EM gate n={len(problems)} band=[{args.band_start},{args.band_end}] "
        f"k={args.k} exclude_topk={args.exclude_topk} random_seeds={seeds}",
        flush=True,
    )

    records = []
    clean_by_id: dict[str, bool] = {}

    for problem in problems:
        rec = run_problem(
            loaded,
            problem,
            settings=settings,
            enable_thinking=False,
            ablation=AblationConfig(kind="none", **base),
            seed=settings.seed,
        )
        rec.extra["ablation_seed"] = None
        records.append(rec)
        clean_by_id[problem.problem_id] = bool(rec.correct)
        print(
            f"{problem.problem_id} clean correct={rec.correct} "
            f"ans={rec.extracted_answer!r}",
            flush=True,
        )

    if args.require_clean_correct:
        eval_problems = [p for p in problems if clean_by_id[p.problem_id]]
        if len(eval_problems) < 5:
            raise SystemExit(
                f"only {len(eval_problems)} clean-correct items; need a better set"
            )
        print(
            f"restricting J/random eval to clean-correct n={len(eval_problems)}",
            flush=True,
        )
    else:
        eval_problems = list(problems)

    j_flags: list[bool] = []
    random_flags_by_seed: dict[int, list[bool]] = {s: [] for s in seeds}

    for problem in eval_problems:
        j_rec = run_problem(
            loaded,
            problem,
            settings=settings,
            enable_thinking=False,
            ablation=AblationConfig(kind="jspace", **base),
            seed=settings.seed,
        )
        j_rec.extra["ablation_seed"] = None
        records.append(j_rec)
        j_flags.append(bool(j_rec.correct))
        print(
            f"{problem.problem_id} jspace correct={j_rec.correct} "
            f"ans={j_rec.extracted_answer!r}",
            flush=True,
        )
        for seed in seeds:
            r_rec = run_problem(
                loaded,
                problem,
                settings=settings,
                enable_thinking=False,
                ablation=AblationConfig(kind="random", seed=seed, **base),
                seed=settings.seed,
            )
            r_rec.extra["ablation_seed"] = seed
            records.append(r_rec)
            random_flags_by_seed[seed].append(bool(r_rec.correct))
            print(
                f"{problem.problem_id} random seed={seed} correct={r_rec.correct} "
                f"ans={r_rec.extracted_answer!r}",
                flush=True,
            )

    clean_flags = [clean_by_id[p.problem_id] for p in eval_problems]
    clean_acc = accuracy(clean_flags)
    j_acc = accuracy(j_flags)
    per_seed_random_acc = {
        str(s): accuracy(flags) for s, flags in random_flags_by_seed.items()
    }
    random_acc = float(
        sum(per_seed_random_acc.values()) / max(len(per_seed_random_acc), 1)
    )
    j_drop = clean_acc - j_acc
    r_drop = clean_acc - random_acc
    j_over_r = random_acc - j_acc

    gate_pass = (
        clean_acc >= 0.7
        and j_drop + 1e-9 >= args.min_j_drop
        and j_over_r + 1e-9 >= args.min_j_over_random
    )
    if gate_pass:
        verdict = "j_specific_em_bite"
    elif j_drop + 1e-9 >= args.min_j_drop:
        verdict = "j_em_drop_not_above_random"
    else:
        verdict = "no_reliable_j_em_drop"

    summary = {
        "n_requested": len(problems),
        "n_eval": len(eval_problems),
        "require_clean_correct": bool(args.require_clean_correct),
        "eval_ids": [p.problem_id for p in eval_problems],
        "band_start": args.band_start,
        "band_end": args.band_end,
        "k": args.k,
        "exclude_topk": args.exclude_topk,
        "random_seeds": list(seeds),
        "clean_accuracy": clean_acc,
        "j_ablated_accuracy": j_acc,
        "random_accuracy_mean": random_acc,
        "random_accuracy_by_seed": per_seed_random_acc,
        "j_drop": j_drop,
        "random_drop": r_drop,
        "j_over_random": j_over_r,
        "min_j_drop": args.min_j_drop,
        "min_j_over_random": args.min_j_over_random,
        "verdict": verdict,
        "gate_pass": gate_pass,
    }

    out_dir = settings.results_dir / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{args.out_name}.jsonl", records)
    out_path = out_dir / f"{args.out_name}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)

    if not gate_pass:
        print(
            "GATE FAIL: need high clean EM, J drop ≥ threshold, and J ≫ random.",
            flush=True,
        )
        if "4B" in settings.model_name:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
