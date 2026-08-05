#!/usr/bin/env python3
"""Plot judge-estimated regex precision (PPV) per cell×condition stratum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.backtrack_metrics import precision_by_stratum
from jspace.judge import read_verdicts
from jspace.records import read_jsonl
from jspace.plotting import (
    BACKTRACK_COND_COLORS,
    BACKTRACK_CONDITIONS,
    BACKTRACK_COND_LABELS,
    backtrack_cell_xticklabels,
    configure_matplotlib,
    ordered_backtrack_cells,
)


def plot_precision(strata: dict[str, dict], out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    cells = ordered_backtrack_cells({s["cell"] for s in strata.values()})
    x = np.arange(len(cells))
    width = 0.24
    offsets = [-width, 0.0, width]

    overall_ppv = (
        sum(s["positives"] for s in strata.values())
        / sum(s["n"] for s in strata.values())
        if strata
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, condition in zip(offsets, BACKTRACK_CONDITIONS):
        ppvs, err_lo, err_hi, ns = [], [], [], []
        for cell in cells:
            key = f"{cell}::{condition}"
            row = strata.get(key)
            if not row:
                ppvs.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                ns.append(0)
                continue
            ppvs.append(row["ppv"])
            err_lo.append(row["ppv"] - row["ci_lo"])
            err_hi.append(row["ci_hi"] - row["ppv"])
            ns.append(row["n"])
        ax.bar(
            x + offset,
            ppvs,
            width=width,
            label=BACKTRACK_COND_LABELS[condition],
            color=BACKTRACK_COND_COLORS[condition],
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 1},
        )

    ax.axhline(
        overall_ppv,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"overall PPV ({overall_ppv:.2f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(backtrack_cell_xticklabels(cells), rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("precision (PPV): P(effective | regex hit)")
    ax.set_title("Stratified regex precision by cell × CoT condition (95% Wilson CI)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/judge_sample_jobs.jsonl",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/judge_verdicts.jsonl",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs/backtrack-analysis/precision_by_stratum.json",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "report/figures/supplementary/backtrack_precision_by_stratum.png",
    )
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs)
    verdicts = {v.span_id: v.is_backtrack for v in read_verdicts(args.verdicts)}
    strata = precision_by_stratum(jobs, verdicts)

    missing = len(jobs) - sum(s["n"] for s in strata.values())
    if missing:
        print(f"warning: {missing} jobs lack verdicts")

    args.out_json.write_text(json.dumps(strata, indent=2) + "\n", encoding="utf-8")
    plot_precision(strata, args.plot)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
