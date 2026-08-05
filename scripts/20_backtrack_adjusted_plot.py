#!/usr/bin/env python3
"""Judge-calibrated backtrack rates with bootstrap 95% CIs (§4.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.backtrack_metrics import adjusted_trace_rates, bootstrap_adjusted_means
from jspace.records import read_jsonl
from jspace.plotting import (
    BACKTRACK_COND_COLORS,
    BACKTRACK_CONDITIONS,
    BACKTRACK_COND_LABELS,
    backtrack_cell_xticklabels,
    configure_matplotlib,
    ordered_backtrack_cells,
)

DEFAULT_TRACES = [
    ROOT / "runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl",
    ROOT / "runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl",
    ROOT / "runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl",
]


def plot_adjusted(summary: dict[str, dict], out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    cells = ordered_backtrack_cells({row["cell"] for row in summary.values()})
    x = np.arange(len(cells))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, condition in zip(offsets, BACKTRACK_CONDITIONS):
        means, err_lo, err_hi = [], [], []
        for cell in cells:
            key = f"{cell}::{condition}"
            row = summary.get(key)
            if not row:
                means.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            means.append(row["mean_adjusted_rate"])
            err_lo.append(row["mean_adjusted_rate"] - row["ci_lo"])
            err_hi.append(row["ci_hi"] - row["mean_adjusted_rate"])
        ax.bar(
            x + offset,
            means,
            width=width,
            label=BACKTRACK_COND_LABELS[condition],
            color=BACKTRACK_COND_COLORS[condition],
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(backtrack_cell_xticklabels(cells), rotation=20, ha="right")
    ax.set_ylabel("mean adjusted backtrack rate (regex rate × stratum PPV)")
    ax.set_title("Judge-calibrated effective backtrack rate by cell × CoT condition (bootstrap 95% CI)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, nargs="*", default=DEFAULT_TRACES)
    parser.add_argument(
        "--precision",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/precision_by_stratum.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/adjusted_rates_boot.json",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "report/figures/supplementary/backtrack_adjusted_rates_boot.png",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.traces:
        rows.extend(read_jsonl(path))

    precision = json.loads(args.precision.read_text(encoding="utf-8"))
    traces = adjusted_trace_rates(rows, precision)
    summary = bootstrap_adjusted_means(traces, n_boot=args.n_boot, seed=args.seed)

    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_adjusted(summary, args.plot)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
