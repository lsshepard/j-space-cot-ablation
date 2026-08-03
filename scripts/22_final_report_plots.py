#!/usr/bin/env python3
"""Curated final report figures with consistent layout and colors."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.calibrate import BandSelection, plot_band_diagnostic
from jspace.plotting import (
    COT_CONDITIONS,
    DIRECT_CONDITIONS,
    REPORT_ARM_COLORS,
    REPORT_ARM_LABELS,
    REPORT_ARM_ORDER,
    REPORT_CELL_LABELS,
    configure_matplotlib,
    report_cell_xticklabels,
    style_report_axes,
)
from jspace.records import read_jsonl

N_BOOT = 10_000
SEED = 0
REPORT_BAND = (27, 31)
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

COPY_FROM_REPORT = ("backtrack_adjusted_rates_boot.png",)

STALE_FINAL_PLOTS = (
    "backtrack_precision_by_stratum.png",
    "backtrack_regex_rates.png",
)


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


def arm_stats(rows: list[dict], conds: tuple[str, ...]):
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
        ("gsm8k", gsm, None),
        ("math500_L1", math, "math500_L1"),
        ("math500_L3", math, "math500_L3"),
        ("math500_L4", math, "math500_L4"),
        ("math500_L5", math, "math500_L5"),
        ("aime", aime, "aime"),
    ]
    datasets = []
    for cell, rows, cell_key in raw:
        xs, pids = filter_complete(slice_rows(rows, cell_key))
        datasets.append((cell, xs, len(pids)))
    return datasets


def tokens_by_condition(rows: list[dict], cond: str) -> list[int]:
    return [int(r.get("trace_length_tokens") or 0) for r in rows if r["condition"] == cond]


def cap_hit_rate(rows: list[dict], cond: str) -> float:
    hits = [r for r in rows if r["condition"] == cond]
    if not hits:
        return float("nan")
    return sum(bool(r.get("hit_token_cap")) for r in hits) / len(hits)


def _fmt_cap(tokens: int) -> str:
    if tokens >= 1000:
        val = tokens / 1000
        s = f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(tokens)


def _cap_for_cond(rows: list[dict], cond: str) -> int:
    caps = [
        int((r.get("extra") or {}).get("max_new_tokens") or 0)
        for r in rows
        if r["condition"] == cond and (r.get("extra") or {}).get("max_new_tokens")
    ]
    return caps[0] if caps else 0


def _arm_cap_xticklabels(rows: list[dict], conds: tuple[str, ...]) -> list[str]:
    return [
        f"{REPORT_ARM_LABELS[arm]}\n{_fmt_cap(_cap_for_cond(rows, cond))}"
        for arm, cond in zip(REPORT_ARM_ORDER, conds)
    ]


def _facet_title(cell: str, n: int) -> str:
    return f"{report_cell_xticklabels([cell])[0]}\n(n={n})"


CAP_HIT_FOOTNOTE = (
    "Cap hit = fraction of traces that exhaust max_new_tokens. "
    "Budget cap per bar shown under arm label (J/r-ablated: up to 6× clean, ceiling 8k)."
)

TOKEN_USAGE_FOOTNOTE = (
    "Budget cap per bar under arm label. "
    "AIME direct uses uncalibrated 512-token fallback."
)


BAR_WIDTH = 0.22
BAR_CENTERS = (-0.42, 0.0, 0.42)
BAR_XLIM = (-0.68, 0.68)
FACET_COL_WIDTH = 2.1
FACET_WSPACE = 0.32
FACET_HSPACE = 0.38


def _make_facet_axes(nrows: int, ncols: int, *, height_per_row: float = 3.8):
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FACET_COL_WIDTH * ncols, height_per_row * nrows),
        sharey="row",
        gridspec_kw={"wspace": FACET_WSPACE, "hspace": FACET_HSPACE},
    )
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    for row in range(nrows):
        for col in range(1, ncols):
            axes[row][col].tick_params(axis="y", left=False, labelleft=False)
    return fig, axes


def _style_facet_cell(
    ax,
    *,
    col: int,
    y_label: str,
    x_labels: list[str] | None = None,
) -> None:
    ax.set_xlim(*BAR_XLIM)
    ax.set_xticks(BAR_CENTERS)
    if x_labels is not None:
        ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha="right")
        ax.tick_params(axis="x", pad=6)
    if col == 0:
        ax.set_ylabel(y_label, fontsize=9)
    style_report_axes(ax)


ARM_XLABELS = [REPORT_ARM_LABELS[a] for a in REPORT_ARM_ORDER]
LABEL_PAD = 0.05


def _label_above(ax, x: float, top_y: float, text: str, *, fontsize: int = 6) -> float:
    """Place value label above bar top or error whisker with fixed gap."""
    ax.text(x, top_y + LABEL_PAD, text, ha="center", va="bottom", fontsize=fontsize)
    return top_y + LABEL_PAD


def plot_em_faceted(
    datasets,
    out_path: Path,
    *,
    modes: tuple[tuple[str, tuple[str, ...]], ...],
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ncols = len(datasets)
    nrows = len(modes)
    fig, axes = _make_facet_axes(nrows, ncols)

    for row, (mode_label, conds) in enumerate(modes):
        row_label_top = 0.0
        for col, (cell, xs, n) in enumerate(datasets):
            ax = axes[row][col]
            stats = arm_stats(xs, conds)
            means = [s[0] for s in stats]
            yerr = np.array([
                [s[0] - s[1] for s in stats],
                [s[2] - s[0] for s in stats],
            ])
            panel_top = 0.0
            for xi, (center, arm) in enumerate(zip(BAR_CENTERS, REPORT_ARM_ORDER)):
                ax.bar(
                    center,
                    means[xi],
                    width=BAR_WIDTH,
                    color=REPORT_ARM_COLORS[arm],
                    yerr=yerr[:, xi : xi + 1],
                    capsize=2.5,
                    error_kw={"elinewidth": 1.0, "ecolor": "0.25"},
                )
            if row == 0:
                ax.set_title(f"{report_cell_xticklabels([cell])[0]}\n(n={n})", fontsize=9)
            _style_facet_cell(
                ax,
                col=col,
                y_label=f"{mode_label}\nexact-match",
                x_labels=ARM_XLABELS,
            )
            for center, (m, _, hi, _) in zip(BAR_CENTERS, stats):
                panel_top = max(panel_top, _label_above(ax, center, hi, f"{m:.2f}"))
            row_label_top = max(row_label_top, panel_top)
        axes[row][0].set_ylim(0, row_label_top + 0.04)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_em_grouped_row(
    datasets,
    mode_label: str,
    conds: tuple[str, ...],
    out_path: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = report_cell_xticklabels([c for c, _, _ in datasets])
    n_sets = len(datasets)
    x = np.arange(n_sets)
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(2.2 * n_sets, 4.2))
    for arm, off in zip(REPORT_ARM_ORDER, offsets):
        means, err_lo, err_hi = [], [], []
        for _, xs, _ in datasets:
            idx = REPORT_ARM_ORDER.index(arm)
            cond = conds[idx]
            point, lo, hi, _ = bootstrap_mean(per_problem_correct(xs, cond).values())
            means.append(point)
            err_lo.append(point - lo)
            err_hi.append(hi - point)
        ax.bar(
            x + off,
            means,
            width=width,
            label=REPORT_ARM_LABELS[arm],
            color=REPORT_ARM_COLORS[arm],
            yerr=[err_lo, err_hi],
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "ecolor": "0.25"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(f"{mode_label} EM by problem set  ·  band [{REPORT_BAND[0]}, {REPORT_BAND[1]}], 95% bootstrap CI")
    ax.legend(frameon=False, fontsize=8)
    style_report_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cot_protection(datasets, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = report_cell_xticklabels([c for c, _, _ in datasets])
    points, los, his, ns = [], [], [], []
    for _, xs, n in datasets:
        point, lo, hi, nn = bootstrap_mean(per_problem_protection(xs))
        points.append(point)
        los.append(lo)
        his.append(hi)
        ns.append(nn)

    fig, ax = plt.subplots(figsize=(2.0 * len(labels), 4.5))
    x = np.arange(len(labels))
    yerr = np.array([
        [p - lo for p, lo in zip(points, los)],
        [hi - p for p, hi in zip(points, his)],
    ])
    ax.bar(
        x,
        points,
        color=REPORT_ARM_COLORS["clean"],
        width=0.55,
        yerr=yerr,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "0.25"},
    )
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("CoT-protection (EM)")
    ax.set_title(
        r"CoT-protection $(direct_{clean}-direct_J)-(cot_{clean}-cot_J)$"
        f"  ·  band [{REPORT_BAND[0]}, {REPORT_BAND[1]}], 95% bootstrap CI",
    )
    for xi, p, hi, lo, n in zip(x, points, his, los, ns):
        y = hi + 0.02 if p >= 0 else lo - 0.02
        ax.text(xi, y, f"{p:+.2f}\n(n={n})", ha="center", va="bottom" if p >= 0 else "top", fontsize=8)
    pad = 0.06
    ax.set_ylim(min(los) - pad, max(his) + pad + 0.05)
    style_report_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def _percentile_strip(ax, x: float, values: list[int], width: float, color: str) -> float:
    import numpy as np

    if not values:
        return 0.0
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    ax.bar(x, p75 - p25, bottom=p25, width=width, color=color, alpha=0.45, linewidth=0)
    ax.plot([x - width / 2, x + width / 2], [p50, p50], color="0.15", linewidth=1.8, zorder=3)
    ax.vlines(x, p10, p90, color=color, linewidth=1.6, zorder=2)
    ax.plot(x, p10, marker="_", color=color, markersize=8, markeredgewidth=1.4)
    ax.plot(x, p90, marker="_", color=color, markersize=8, markeredgewidth=1.4)
    return float(p90)


def plot_token_usage(datasets, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    modes = [("Direct", DIRECT_CONDITIONS), ("CoT", COT_CONDITIONS)]
    ncols = len(datasets)
    fig, axes = _make_facet_axes(2, ncols, height_per_row=4.6)

    for row, (mode_label, conds) in enumerate(modes):
        row_max = 0.0
        for _, xs, _ in datasets:
            for cond in conds:
                vals = tokens_by_condition(xs, cond)
                if vals:
                    row_max = max(row_max, float(np.max(vals)))
        for col, (cell, xs, n) in enumerate(datasets):
            ax = axes[row][col]
            for center, arm, cond in zip(BAR_CENTERS, REPORT_ARM_ORDER, conds):
                _percentile_strip(
                    ax, center, tokens_by_condition(xs, cond), BAR_WIDTH, REPORT_ARM_COLORS[arm],
                )
            ax.set_title(_facet_title(cell, n), fontsize=9)
            _style_facet_cell(
                ax,
                col=col,
                y_label=f"{mode_label}\ntokens",
                x_labels=_arm_cap_xticklabels(xs, conds),
            )
        axes[row][0].set_ylim(0, row_max * 1.12 if row_max else 1)

    fig.suptitle(
        "Token usage by problem set  ·  box = p25–p75, line = median, whiskers = p10/p90",
        fontsize=11,
        y=1.04,
    )
    fig.text(0.5, 0.01, TOKEN_USAGE_FOOTNOTE, ha="center", fontsize=8, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cap_hit_rates(datasets, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [("Direct", DIRECT_CONDITIONS), ("CoT", COT_CONDITIONS)]
    ncols = len(datasets)
    fig, axes = _make_facet_axes(2, ncols, height_per_row=4.2)

    for row, (mode_label, conds) in enumerate(modes):
        row_label_top = 0.0
        for col, (cell, xs, n) in enumerate(datasets):
            ax = axes[row][col]
            rates = [cap_hit_rate(xs, c) for c in conds]
            panel_top = 0.0
            for center, arm, rate in zip(BAR_CENTERS, REPORT_ARM_ORDER, rates):
                ax.bar(center, rate, width=BAR_WIDTH, color=REPORT_ARM_COLORS[arm])
                if rate > 0:
                    panel_top = max(panel_top, _label_above(ax, center, rate, f"{rate:.0%}"))
            ax.set_title(_facet_title(cell, n), fontsize=9)
            _style_facet_cell(
                ax,
                col=col,
                y_label=f"{mode_label}\ncap hit rate",
                x_labels=_arm_cap_xticklabels(xs, conds),
            )
            row_label_top = max(row_label_top, panel_top, max(rates))
        axes[row][0].set_ylim(0, max(row_label_top + 0.04, 0.15))

    fig.suptitle("Budget cap hit rate by problem set", fontsize=11, y=1.04)
    fig.text(0.5, 0.01, CAP_HIT_FOOTNOTE, ha="center", fontsize=8, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_j_ablation_drops(datasets, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = report_cell_xticklabels([c for c, _, _ in datasets])
    direct_drops, cot_drops = [], []
    for _, xs, _ in datasets:
        dc = per_problem_correct(xs, "direct_clean")
        dj = per_problem_correct(xs, "direct_j_ablated")
        cc = per_problem_correct(xs, "cot_clean")
        cj = per_problem_correct(xs, "cot_j_ablated")
        pids = sorted(set(dc) & set(dj) & set(cc) & set(cj))
        direct_drops.append(sum(dc[p] - dj[p] for p in pids) / len(pids))
        cot_drops.append(sum(cc[p] - cj[p] for p in pids) / len(pids))

    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(2.0 * len(labels), 4.2))
    ax.bar(x - w / 2, direct_drops, width=w, label="direct drop (clean−J)", color=REPORT_ARM_COLORS["j_ablated"])
    ax.bar(x + w / 2, cot_drops, width=w, label="CoT drop (clean−J)", color=REPORT_ARM_COLORS["random"])
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("EM drop from J ablation")
    ax.set_title(f"J ablation harm by problem set  ·  band [{REPORT_BAND[0]}, {REPORT_BAND[1]}]")
    ax.legend(frameon=False, fontsize=8)
    style_report_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_band_selection(out_path: Path) -> None:
    band_path = ROOT / "runs/active/calibration/band.json"
    if not band_path.exists():
        shutil.copy2(ROOT / "report_results/band_selection.png", out_path)
        print(f"copied {out_path} (no band.json)")
        return
    raw = json.loads(band_path.read_text(encoding="utf-8"))
    by_topk = {int(k): v for k, v in raw["match_rates_by_topk"].items()}
    band = BandSelection(
        band_start=REPORT_BAND[0],
        band_end=REPORT_BAND[1],
        auto_selected=raw.get("auto_selected", False),
        strength_label=raw.get("strength_label", "human_override"),
        match_rates=raw.get("match_rates"),
        match_rates_by_topk=by_topk,
        select_topk=int(raw.get("select_topk", 16)),
    )
    configure_matplotlib()
    plot_band_diagnostic(by_topk, band, out_path)
    print(f"wrote {out_path}")


def copy_static_plots(src_dir: Path, out_dir: Path) -> None:
    for name in COPY_FROM_REPORT:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"copied {out_dir / name}")


def remove_stale_plots(out_dir: Path) -> None:
    for name in STALE_FINAL_PLOTS:
        path = out_dir / name
        if path.exists():
            path.unlink()
            print(f"removed {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report_results/final")
    parser.add_argument("--report-src", type=Path, default=ROOT / "report_results")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_plots(out_dir)

    datasets = load_datasets(DEFAULT_TRACES)
    for cell, _, n in datasets:
        print(f"{REPORT_CELL_LABELS.get(cell, cell)}: n={n}")

    modes_all = [("Direct", DIRECT_CONDITIONS), ("CoT", COT_CONDITIONS)]
    plot_em_faceted(
        datasets,
        out_dir / "em_overview_all.png",
        modes=modes_all,
        title=f"Exact-match accuracy  ·  band [{REPORT_BAND[0]}, {REPORT_BAND[1]}], 95% bootstrap CI",
    )
    plot_em_grouped_row(datasets, "Direct", DIRECT_CONDITIONS, out_dir / "em_direct_all.png")
    plot_em_grouped_row(datasets, "CoT", COT_CONDITIONS, out_dir / "em_cot_all.png")
    plot_cot_protection(datasets, out_dir / "cot_protection_all.png")
    plot_token_usage(datasets, out_dir / "token_usage_by_set.png")
    plot_cap_hit_rates(datasets, out_dir / "cap_hit_rate_by_set.png")
    plot_j_ablation_drops(datasets, out_dir / "j_ablation_drops_by_set.png")

    write_band_selection(out_dir / "band_selection.png")
    copy_static_plots(args.report_src, out_dir)


if __name__ == "__main__":
    main()
