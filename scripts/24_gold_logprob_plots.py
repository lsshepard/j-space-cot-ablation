#!/usr/bin/env python3
"""Teacher-forced gold-answer logprob plots (graded metric from traces)."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.plotting import (
    REPORT_ARM_COLORS,
    REPORT_ARM_LABELS,
    REPORT_ARM_ORDER,
    configure_matplotlib,
    report_cell_xticklabels,
)
from jspace.records import read_jsonl

N_BOOT = 10_000
SEED = 0
REQUIRED = [
    "direct_clean", "direct_j_ablated", "direct_random",
    "cot_clean", "cot_j_ablated", "cot_random",
]
DIRECT = ("direct_clean", "direct_random", "direct_j_ablated")
COT = ("cot_clean", "cot_random", "cot_j_ablated")
DEFAULT_TRACES = {
    "gsm": ROOT / "runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl",
    "math": ROOT / "runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl",
    "aime": ROOT / "runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl",
}
BAR_CENTERS = [-0.78, 0.0, 0.78]
BAR_WIDTH = 0.72


def slice_rows(rows, cell):
    if cell is None:
        return rows
    return [r for r in rows if (r.get("extra") or {}).get("cell") == cell]


def complete_ids(rows):
    by = defaultdict(set)
    for r in rows:
        by[r["problem_id"]].add(r["condition"])
    return {pid for pid, c in by.items() if set(REQUIRED) <= c}


def filter_complete(rows):
    ok = complete_ids(rows)
    return [r for r in rows if r["problem_id"] in ok]


def per_problem_lp(rows, cond):
    by = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("graded_logprob") is not None:
            by[r["problem_id"]].append(float(r["graded_logprob"]))
    return {pid: sum(v) / len(v) for pid, v in by.items()}


def bootstrap_mean(values, n_boot=N_BOOT, seed=SEED):
    vals = list(values)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = sum(vals) / n
    rng = random.Random(seed)
    means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return point, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def load_datasets():
    gsm = read_jsonl(DEFAULT_TRACES["gsm"])
    math = read_jsonl(DEFAULT_TRACES["math"])
    aime = read_jsonl(DEFAULT_TRACES["aime"])
    raw = [
        ("gsm8k", gsm, None),
        ("math500_L1", math, "math500_L1"),
        ("math500_L3", math, "math500_L3"),
        ("math500_L4", math, "math500_L4"),
        ("math500_L5", math, "math500_L5"),
        ("aime", aime, None),
    ]
    out = []
    for cell, rows, key in raw:
        xs = filter_complete(slice_rows(rows, key))
        out.append((cell, xs, len(complete_ids(xs))))
    return out


def plot_overview(datasets, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    modes = [("Direct", DIRECT), ("CoT", COT)]
    fig, axes = plt.subplots(2, len(datasets), figsize=(2.05 * len(datasets), 5.4), sharey="row")
    arm_labels = [REPORT_ARM_LABELS[a] for a in REPORT_ARM_ORDER]
    for row, (mode_label, conds) in enumerate(modes):
        row_lo, row_hi = 0.0, -1e9
        for col, (cell, xs, n) in enumerate(datasets):
            ax = axes[row][col]
            stats = [bootstrap_mean(per_problem_lp(xs, c).values()) for c in conds]
            means = [s[0] for s in stats]
            yerr = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
            for xi, (center, arm) in enumerate(zip(BAR_CENTERS, REPORT_ARM_ORDER)):
                ax.bar(
                    center, means[xi], width=BAR_WIDTH, color=REPORT_ARM_COLORS[arm],
                    yerr=yerr[:, xi : xi + 1], capsize=2.0,
                    error_kw={"elinewidth": 0.9, "ecolor": "0.25"},
                )
            if row == 0:
                ax.set_title(f"{report_cell_xticklabels([cell])[0]}\n(n={n})", fontsize=8)
            ax.set_xticks(BAR_CENTERS)
            ax.set_xticklabels(arm_labels, rotation=25, ha="right", fontsize=6)
            if col == 0:
                ax.set_ylabel(f"{mode_label}\ngold logprob", fontsize=8)
            for m, lo, hi in stats:
                row_lo, row_hi = min(row_lo, lo), max(row_hi, hi)
        pad = 0.15 * (row_hi - row_lo + 1e-6)
        for col in range(len(datasets)):
            axes[row][col].set_ylim(row_lo - pad, min(0.05, row_hi + pad))
    fig.suptitle("Teacher-forced gold-answer logprob  ·  band [27, 31], 95% bootstrap CI", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_j_bite(datasets, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, mode_label, clean_c, j_c in [
        (axes[0], "Direct", "direct_clean", "direct_j_ablated"),
        (axes[1], "CoT", "cot_clean", "cot_j_ablated"),
    ]:
        labels, points, los, his = [], [], [], []
        for cell, xs, _ in datasets:
            clean = per_problem_lp(xs, clean_c)
            jab = per_problem_lp(xs, j_c)
            pids = sorted(set(clean) & set(jab))
            point, lo, hi = bootstrap_mean([clean[p] - jab[p] for p in pids])
            labels.append(report_cell_xticklabels([cell])[0])
            points.append(point)
            los.append(lo)
            his.append(hi)
        x = np.arange(len(labels))
        yerr = np.array([
            [p - lo for p, lo in zip(points, los)],
            [hi - p for p, hi in zip(points, his)],
        ])
        ax.bar(x, points, color=REPORT_ARM_COLORS["j_ablated"], width=0.6, yerr=yerr,
               capsize=2.5, error_kw={"elinewidth": 1.0, "ecolor": "0.25"})
        ax.axhline(0, color="0.4", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{mode_label}: clean − J gold logprob")
        ax.set_ylabel("Δ gold logprob (clean − J)")
    fig.suptitle("J-ablation bite on teacher-forced gold logprob", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report/figures/final")
    args = parser.parse_args()
    datasets = load_datasets()
    plot_overview(datasets, args.out_dir / "gold_logprob_overview.png")
    plot_j_bite(datasets, args.out_dir / "gold_logprob_j_bite.png")


if __name__ == "__main__":
    main()
