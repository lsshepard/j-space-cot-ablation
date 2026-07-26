#!/usr/bin/env python3
"""Paired bootstrap CI on (random_mean − J) exact-match accuracy per gate run.

04_multihop_gate.py reports j_over_random as a point estimate. At n≈28 that
number needs an interval before it can carry a pass/fail, so recompute it from
the committed per-problem records.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.config import load_settings


def per_problem_flags(path: Path) -> dict[str, dict[str, bool]]:
    """{problem_id: {clean|j|r<seed>: correct}} from a gate jsonl."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec["ablation"]
        if kind == "none":
            key = "clean"
        elif kind == "jspace":
            key = "j"
        else:
            key = f"r{rec['extra'].get('ablation_seed')}"
        out[rec["problem_id"]][key] = bool(rec["correct"])
    return dict(out)


def bootstrap_j_over_random(
    flags: dict[str, dict[str, bool]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Resample problems (J and random share the item, so pair them)."""
    diffs: list[float] = []
    for pid in sorted(flags):
        row = flags[pid]
        if "j" not in row:
            continue  # not in the eval subset
        rkeys = [k for k in row if k.startswith("r")]
        random_mean = sum(row[k] for k in rkeys) / len(rkeys)
        diffs.append(random_mean - float(row["j"]))

    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return {
        "n": n,
        "j_over_random": sum(diffs) / n,
        "lo": means[int(0.025 * n_boot)],
        "hi": means[int(0.975 * n_boot)],
    }


def arm_accuracies(flags: dict[str, dict[str, bool]], ids: list[str]) -> dict[str, float]:
    def mean(fn) -> float:
        return sum(fn(flags[i]) for i in ids) / len(ids)

    return {
        "clean_accuracy": mean(lambda row: float(row["clean"])),
        "j_accuracy": mean(lambda row: float(row["j"])),
        "random_accuracy": mean(
            lambda row: sum(row[k] for k in row if k.startswith("r"))
            / len([k for k in row if k.startswith("r")])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", type=str, default="em_gate_ci")
    args = parser.parse_args()

    gates_dir = load_settings().results_dir / "gates"
    results = {}
    for path in sorted(gates_dir.glob("multihop_em_*.jsonl")):
        flags = per_problem_flags(path)
        stats = bootstrap_j_over_random(flags, n_boot=args.n_boot, seed=args.seed)
        stats["excludes_zero"] = stats["lo"] > 0.0

        # J-ablation removes whatever the model was about to say, so the two
        # strata move in opposite directions and cancel when pooled.
        evaluated = {k: v for k, v in flags.items() if "j" in v}
        strata: dict[str, dict] = {}
        for label, ids in (
            ("clean_correct", [i for i in evaluated if evaluated[i]["clean"]]),
            ("clean_wrong", [i for i in evaluated if not evaluated[i]["clean"]]),
        ):
            if len(ids) < 3:
                continue
            sub = bootstrap_j_over_random(
                {i: evaluated[i] for i in ids}, n_boot=args.n_boot, seed=args.seed
            )
            sub["excludes_zero"] = sub["lo"] > 0.0
            sub.update(arm_accuracies(evaluated, ids))
            strata[label] = sub
        stats["strata"] = strata
        results[path.stem] = stats

        print(
            f"{path.stem:<30} n={stats['n']:<3} "
            f"j_over_random={stats['j_over_random']:+.3f} "
            f"95% CI [{stats['lo']:+.3f}, {stats['hi']:+.3f}] "
            f"{'EXCLUDES 0' if stats['excludes_zero'] else 'includes 0'}",
            flush=True,
        )
        for label, sub in strata.items():
            print(
                f"    {label:<14} n={sub['n']:<3} "
                f"clean={sub['clean_accuracy']:.3f} J={sub['j_accuracy']:.3f} "
                f"random={sub['random_accuracy']:.3f} "
                f"rand-J={sub['j_over_random']:+.3f} "
                f"[{sub['lo']:+.3f}, {sub['hi']:+.3f}]"
                f"{'  EXCLUDES 0' if sub['excludes_zero'] else ''}",
                flush=True,
            )

    payload = {"n_boot": args.n_boot, "seed": args.seed, "runs": results}
    out_path = gates_dir / f"{args.out_name}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
