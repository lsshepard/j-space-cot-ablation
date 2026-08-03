#!/usr/bin/env python3
"""Pilot EM and CoT-protection plots with bootstrap 95% CIs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.plotting import configure_matplotlib
from jspace.records import read_jsonl

N_BOOT = 10_000
SEED = 0
REQUIRED = [
    "direct_clean",
    "direct_j_ablated",
    "direct_random",
    "cot_clean",
    "cot_j_ablated",
    "cot_random",
]

DEFAULT_TRACES = {
    "gsm": ROOT / "runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl",
    "math": ROOT / "runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl",
    "aime": ROOT / "runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl",
}

ARM_LABELS = ["clean", "control", "J-ablated"]
COLORS = ["#4C78A8", "#72B7B2", "#F58518"]
COT_CONDS = ["cot_clean", "cot_random", "cot_j_ablated"]
DIR_CONDS = ["direct_clean", "direct_random", "direct_j_ablated"]


def slice_rows(rows: list[dict], cell: str | None) -> list[dict]:
    if cell is None:
        return rows
    return [r for r in rows if (r.get("extra") or {}).get("cell") == cell]


def complete_problem_ids(rows: list[dict]) -> set[str]:
    by: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by[r["problem_id"]].add(r["condition"])
    return {pid for pid, conds in by.items() if set(REQUIRED) <= conds}


def filter_complete(rows: list[dict]) -> tuple[list[dict], list[str]]:
    ok = complete_problem_ids(rows)
    return [r for r in rows if r["problem_id"] in ok], sorted(ok)


def per_problem_correct(rows: list[dict], cond: str) -> dict[str, float]:
    by: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r["condition"] == cond:
            by[r["problem_id"]].append(bool(r.get("correct")))
    return {pid: sum(vs) / len(vs) for pid, vs in by.items()}


def bootstrap_mean(values, *, n_boot: int = N_BOOT, seed: int = SEED):
    vals = list(values)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = sum(vals) / n
    rng = random.Random(seed)
    means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return point, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)], n


def arm_stats(rows: list[dict], conds: list[str]):
    return [bootstrap_mean(per_problem_correct(rows, c).values()) for c in conds]


def per_problem_protection(rows: list[dict]) -> list[float]:
    arms = {c: per_problem_correct(rows, c) for c in [
        "direct_clean", "direct_j_ablated", "cot_clean", "cot_j_ablated"
    ]}
    pids = sorted(set.intersection(*(set(v) for v in arms.values())))
    return [
        (arms["direct_clean"][pid] - arms["direct_j_ablated"][pid])
        - (arms["cot_clean"][pid] - arms["cot_j_ablated"][pid])
        for pid in pids
    ]


def load_datasets(traces: dict[str, Path]):
    gsm = read_jsonl(traces["gsm"])
    math = read_jsonl(traces["math"])
    aime = read_jsonl(traces["aime"])
    raw = [
        ("GSM8K", gsm, None),
        ("MATH L1", math, "math500_L1"),
        ("MATH L3", math, "math500_L3"),
        ("MATH L4", math, "math500_L4"),
        ("MATH L5", math, "math500_L5"),
        ("AIME", aime, "aime"),
    ]
    datasets = []
    for name, rows, cell in raw:
        xs, pids = filter_complete(slice_rows(rows, cell))
        datasets.append((name, xs, len(pids)))
        print(f"{name}: complete problems n={len(pids)} traces={len(xs)}")
    return datasets


def plot_overview(datasets, out_path: Path, *, title_suffix: str) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    ncols = len(datasets)
    fig, axes = plt.subplots(2, ncols, figsize=(2.7 * ncols, 6.5), sharey=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, (name, xs, n) in enumerate(datasets):
        for row, (mode, conds) in enumerate([("Direct", DIR_CONDS), ("CoT", COT_CONDS)]):
            ax = axes[row][col]
            stats = arm_stats(xs, conds)
            means = [s[0] for s in stats]
            yerr = np.array([
                [s[0] - s[1] for s in stats],
                [s[2] - s[0] for s in stats],
            ])
            x = np.arange(3) * 0.55
            ax.bar(
                x, means, color=COLORS, width=0.45, yerr=yerr, capsize=2.5,
                error_kw={"elinewidth": 1.1, "ecolor": "0.25"},
            )
            ax.set_xticks(x)
            ax.set_xticklabels(ARM_LABELS, fontsize=8)
            ax.set_ylim(0, 1.0)
            if row == 0:
                ax.set_title(f"{name}\n(n={n})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{mode}\nexact-match accuracy")
            for xi, (m, lo, hi, _) in zip(x, stats):
                ax.text(xi, min(hi + 0.025, 0.98), f"{m:.2f}", ha="center", va="bottom", fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"Pilot EM with 95% bootstrap CIs  (band [27, 31], seed 0, 10k resamples)\n{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_single_row(datasets, mode: str, conds: list[str], out_path: Path, *, title_suffix: str) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(2.7 * ncols, 3.8), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, (name, xs, n) in zip(axes, datasets):
        stats = arm_stats(xs, conds)
        means = [s[0] for s in stats]
        yerr = np.array([
            [s[0] - s[1] for s in stats],
            [s[2] - s[0] for s in stats],
        ])
        x = np.arange(3) * 0.55
        ax.bar(
            x, means, color=COLORS, width=0.45, yerr=yerr, capsize=2.5,
            error_kw={"elinewidth": 1.1, "ecolor": "0.25"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ARM_LABELS, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{name} (n={n})", fontsize=10)
        for xi, (m, lo, hi, _) in zip(x, stats):
            ax.text(xi, min(hi + 0.025, 0.98), f"{m:.2f}", ha="center", va="bottom", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("exact-match accuracy")
    fig.suptitle(
        f"{mode} EM with 95% bootstrap CIs  (band [27, 31]; {title_suffix})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_protection(datasets, out_path: Path, *, bootstrap: bool, title_suffix: str) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    names, points, los, his, ns = [], [], [], [], []
    for name, xs, n in datasets:
        if bootstrap:
            point, lo, hi, nn = bootstrap_mean(per_problem_protection(xs))
            ns.append(nn)
        else:
            vals = per_problem_protection(xs)
            point = sum(vals) / len(vals) if vals else float("nan")
            lo, hi = point, point
            ns.append(len(vals))
        names.append(name)
        points.append(point)
        los.append(lo)
        his.append(hi)
        print(f"protection {name}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}] n={ns[-1]}")

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.5))
    x = np.arange(len(names))
    kwargs = {"color": "#4C78A8", "width": 0.55}
    if bootstrap:
        yerr = np.array([
            [p - lo for p, lo in zip(points, los)],
            [hi - p for p, hi in zip(points, his)],
        ])
        kwargs.update(yerr=yerr, capsize=3.5, error_kw={"elinewidth": 1.3, "ecolor": "0.25"})
    ax.bar(x, points, **kwargs)
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("CoT-protection (EM)")
    ci_note = " with 95% bootstrap CIs" if bootstrap else ""
    ax.set_title(
        f"CoT-protection{ci_note}\n"
        r"$(direct_{clean}-direct_J)-(cot_{clean}-cot_J)$"
        f"  ·  band [27, 31], 10k resamples  ·  {title_suffix}",
    )
    for xi, p, lo, hi, n in zip(x, points, los, his, ns):
        y_text = (hi if bootstrap else p) + 0.02 if p >= 0 else (lo if bootstrap else p) - 0.02
        label = f"{p:+.2f}\n(n={n})" if bootstrap else f"{p:+.3f}"
        ax.text(xi, y_text, label, ha="center", va="bottom" if p >= 0 else "top", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pad = 0.06
    ax.set_ylim(min(los) - pad, max(his) + pad + 0.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_protection_with_drops(datasets, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    names, prots, d_drops, c_drops = [], [], [], []
    for name, xs, _ in datasets:
        dc = per_problem_correct(xs, "direct_clean")
        dj = per_problem_correct(xs, "direct_j_ablated")
        cc = per_problem_correct(xs, "cot_clean")
        cj = per_problem_correct(xs, "cot_j_ablated")
        pids = sorted(set(dc) & set(dj) & set(cc) & set(cj))
        dd = sum(dc[p] - dj[p] for p in pids) / len(pids)
        cd = sum(cc[p] - cj[p] for p in pids) / len(pids)
        names.append(name)
        d_drops.append(dd)
        c_drops.append(cd)
        prots.append(dd - cd)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 4.2))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, d_drops, width=w, label="direct drop (clean−J)", color="#F58518")
    ax.bar(x, c_drops, width=w, label="CoT drop (clean−J)", color="#72B7B2")
    ax.bar(x + w, prots, width=w, label="CoT-protection", color="#4C78A8")
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("EM difference")
    ax.set_title("J ablation drops and CoT-protection  ·  band [27, 31], n=15, seed 0")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "report_results")
    args = parser.parse_args()
    plot_dir = args.plot_dir

    all_sets = load_datasets(DEFAULT_TRACES)
    gsm_math = [d for d in all_sets if d[0] != "AIME"]

    plot_overview(all_sets, plot_dir / "em_overview_all_boot.png", title_suffix="all pilot cells")
    plot_overview(gsm_math, plot_dir / "em_overview_gsm8k_math500_boot.png", title_suffix="GSM8K + MATH levels")

    for mode, conds, fname in [
        ("CoT", COT_CONDS, "em_cot_all_boot.png"),
        ("Direct", DIR_CONDS, "em_direct_all_boot.png"),
    ]:
        plot_single_row(all_sets, mode, conds, plot_dir / fname, title_suffix="all pilot cells")

    for mode, conds, fname in [
        ("CoT", COT_CONDS, "em_cot_gsm8k_math500_boot.png"),
        ("Direct", DIR_CONDS, "em_direct_gsm8k_math500_boot.png"),
    ]:
        plot_single_row(gsm_math, mode, conds, plot_dir / fname, title_suffix="GSM8K + MATH levels")

    plot_protection(all_sets, plot_dir / "cot_protection_all_boot.png", bootstrap=True, title_suffix="all pilot cells")
    plot_protection(gsm_math, plot_dir / "cot_protection_by_set_boot.png", bootstrap=True, title_suffix="GSM8K + MATH levels")
    plot_protection(gsm_math, plot_dir / "cot_protection_by_set.png", bootstrap=False, title_suffix="GSM8K + MATH levels")
    plot_protection_with_drops(gsm_math, plot_dir / "cot_protection_with_drops.png")


if __name__ == "__main__":
    main()
