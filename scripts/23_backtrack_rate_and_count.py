#!/usr/bin/env python3
"""Two-panel effective-backtracking figure: judge-calibrated rate + raw count.

Top: judge-precision-calibrated, length-normalized backtrack rate (per 1k tokens)
with 95% bootstrap intervals. Bottom: raw reversal-marker count per trace. Both use
the report clean -> random -> J-ablated ordering and palette.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.plotting import (
    BACKTRACK_COND_COLORS,
    BACKTRACK_COND_LABELS,
    BACKTRACK_CONDITIONS,
    backtrack_cell_xticklabels,
    configure_matplotlib,
    ordered_backtrack_cells,
)

FIGSIZE = (8.2, 2.67)  # 2/3 the height of the earlier 8.2x4.0 figure, same width
RATE_SCALE = 1000.0  # per-token adjusted rate -> per-1k-token


def _cells(summary: dict) -> list[str]:
    return ordered_backtrack_cells({row["cell"] for row in summary.values()})


def plot_rate_and_count(adjusted: dict, regex: dict, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    strata = regex["strata"]
    cells = _cells(adjusted)
    x = np.arange(len(cells))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, (ax_rate, ax_count) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"hspace": 0.18}
    )

    for offset, condition in zip(offsets, BACKTRACK_CONDITIONS):
        color = BACKTRACK_COND_COLORS[condition]
        rate_means, err_lo, err_hi, counts = [], [], [], []
        for cell in cells:
            key = f"{cell}::{condition}"
            row = adjusted.get(key)
            if row:
                m = row["mean_adjusted_rate"] * RATE_SCALE
                rate_means.append(m)
                err_lo.append(m - row["ci_lo"] * RATE_SCALE)
                err_hi.append(row["ci_hi"] * RATE_SCALE - m)
            else:
                rate_means.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
            srow = strata.get(key)
            counts.append(srow["mean_count"] if srow else 0.0)

        ax_rate.bar(
            x + offset,
            rate_means,
            width=width,
            label=BACKTRACK_COND_LABELS[condition],
            color=color,
            yerr=[err_lo, err_hi],
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "ecolor": "0.25"},
        )
        ax_count.bar(x + offset, counts, width=width, color=color)

    ax_rate.set_ylabel("adj. rate\n(backtracks/1k tok)", fontsize=10)
    ax_rate.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)

    ax_count.set_ylabel("raw markers\n/ trace", fontsize=10)
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(backtrack_cell_xticklabels(cells), fontsize=10)
    ax_count.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)

    fig.suptitle(
        "Effective backtracking: judge-calibrated rate (top) and raw marker count (bottom)",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjusted",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/adjusted_rates_boot.json",
    )
    parser.add_argument(
        "--regex",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/regex_summary.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "report/figures/final/backtrack_rate_and_count.png",
    )
    args = parser.parse_args()

    adjusted = json.loads(args.adjusted.read_text(encoding="utf-8"))
    regex = json.loads(args.regex.read_text(encoding="utf-8"))
    plot_rate_and_count(adjusted, regex, args.out)


if __name__ == "__main__":
    main()
