#!/usr/bin/env python3
"""Stratify a gold-lp triad by clean confidence.

The pooled triad mean is not interpretable on a set the model often gets wrong.
Where the gold answer is already near-certain, J-ablation removes it and P(gold)
falls; where the model is confidently wrong, the top J-lens directions carry the
wrong answer, so removing them flattens the distribution and P(gold) *rises*.
The two populations carry opposite signs and cancel when pooled, so report them
apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings
from jspace.diagnostics.gold_lp_triad import (
    GoldLpTriadRow,
    summarize_gold_lp_triad,
)

# Pre-declared in the runpod-run-4 handoff: "clean gold lp > -1".
CONFIDENT_CLEAN_LP = -1.0


def rows_from_payload(payload: dict) -> list[GoldLpTriadRow]:
    return [
        GoldLpTriadRow(
            problem_id=r["problem_id"],
            gold_answer=r["gold_answer"],
            clean_lp=r["clean_lp"],
            jspace_lp=r["jspace_lp"],
            random_lps={int(k): v for k, v in r["random_lps"].items()},
        )
        for r in payload["rows"]
    ]


def stratum(
    rows: list[GoldLpTriadRow],
    payload: dict,
    *,
    n_boot: int,
) -> dict:
    res = summarize_gold_lp_triad(
        rows,
        band_start=payload["band_start"],
        band_end=payload["band_end"],
        k=payload["k"],
        exclude_topk=payload["exclude_topk"],
        random_seeds=payload["random_seeds"],
        n_boot=n_boot,
        boot_seed=0,
    )
    jr = res.delta_j_minus_r
    return {
        "n": res.n,
        "mean_clean_lp": res.mean_clean_lp,
        "delta_j": res.delta_j.to_dict(),
        "delta_j_minus_r": jr.to_dict(),
        "excludes_zero_positive": jr.lo > 0.0,
        "excludes_zero_negative": jr.hi < 0.0,
        "verdict": res.verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "triads",
        nargs="+",
        type=Path,
        help="gold_lp_triad_*.json files to stratify",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=CONFIDENT_CLEAN_LP)
    parser.add_argument("--out-name", type=str, default="triad_by_confidence")
    args = parser.parse_args()

    out: dict[str, dict] = {}
    for path in args.triads:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = rows_from_payload(payload)
        conf = [r for r in rows if r.clean_lp > args.threshold]
        rest = [r for r in rows if r.clean_lp <= args.threshold]
        entry = {"all": stratum(rows, payload, n_boot=args.n_boot)}
        if len(conf) >= 3:
            entry["confident"] = stratum(conf, payload, n_boot=args.n_boot)
        if len(rest) >= 3:
            entry["not_confident"] = stratum(rest, payload, n_boot=args.n_boot)
        key = f"{payload.get('dataset', 'unknown')}:{path.parent.name}"
        out[key] = entry

        print(f"\n=== {key}  (confident = clean_lp > {args.threshold}) ===", flush=True)
        for label, st in entry.items():
            jr = st["delta_j_minus_r"]
            if st["excludes_zero_positive"]:
                tag = "EXCLUDES 0 (J bites)"
            elif st["excludes_zero_negative"]:
                tag = "EXCLUDES 0 (J helps)"
            else:
                tag = "includes 0"
            print(
                f"  {label:<15} n={st['n']:<4} clean_lp={st['mean_clean_lp']:+7.3f} "
                f"dJ-dR={jr['mean']:+.3f} [{jr['lo']:+.3f}, {jr['hi']:+.3f}]  {tag}",
                flush=True,
            )

    dest = load_settings().results_dir / "diagnostics" / f"{args.out_name}.json"
    dest.write_text(
        json.dumps(
            {"threshold_clean_lp": args.threshold, "n_boot": args.n_boot, "strata": out},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
