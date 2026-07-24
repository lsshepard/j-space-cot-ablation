#!/usr/bin/env python3
"""Multi-hop positive control: J-ablation should tank two-hop accuracy (§7.5)."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "lens-eval-multihop.json",
        help="multihop JSON fixture (homemade or Anthropic lens-eval-multihop)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--band-start", type=int, default=None)
    parser.add_argument("--band-end", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="debug override: same cap for every arm")
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.3,
        help="minimum clean-ablated accuracy drop required to pass gate",
    )
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

    band_start = args.band_start if args.band_start is not None else (
        settings.band_start if settings.band_start is not None else default_start
    )
    band_end = args.band_end if args.band_end is not None else (
        settings.band_end if settings.band_end is not None else default_end
    )
    k = args.k if args.k is not None else settings.k

    problems = load_multihop_fixture(args.fixture)
    if args.limit is not None:
        problems = problems[: args.limit]

    records = []
    for problem in problems:
        for kind in ("none", "jspace"):
            abl = AblationConfig(
                kind=kind,  # type: ignore[arg-type]
                band_start=band_start,
                band_end=band_end,
                k=k,
                ablate_prompt_tokens=True,
            )
            rec = run_problem(
                loaded,
                problem,
                settings=settings,
                enable_thinking=False,
                ablation=abl,
                seed=settings.seed,
            )
            records.append(rec)
            print(
                f"{problem.problem_id} {kind} correct={rec.correct} "
                f"ans={rec.extracted_answer!r}"
            )

    clean = [r.correct for r in records if r.ablation == "none" and r.correct is not None]
    ablated = [
        r.correct for r in records if r.ablation == "jspace" and r.correct is not None
    ]
    clean_acc = accuracy(clean)
    abl_acc = accuracy(ablated)
    drop = clean_acc - abl_acc
    summary = {
        "clean_accuracy": clean_acc,
        "j_ablated_accuracy": abl_acc,
        "drop": drop,
        "band_start": band_start,
        "band_end": band_end,
        "k": k,
        "n": len(problems),
        "gate_pass": drop >= args.min_drop,
    }
    out_dir = settings.results_dir / "gates"
    write_jsonl(out_dir / "multihop.jsonl", records)
    (out_dir / "multihop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if not summary["gate_pass"]:
        # Soft on tiny/dev models; hard messaging always.
        print(
            "GATE FAIL MESSAGE: ablation did not tank multi-hop enough — "
            "hook/band/k may be wrong; downstream nulls uninterpretable."
        )
        if "4B" in settings.model_name:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
