#!/usr/bin/env python3
"""Regex backtrack rates across pilot traces + stratified judge sample (§4.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.backtrack_metrics import (
    spans_from_rows,
    stratified_span_sample,
    summarize_regex_backtracks,
)
from jspace.judge import load_judge_prompt_template, assemble_judge_prompt
from jspace.plotting import (
    BACKTRACK_COND_COLORS,
    BACKTRACK_CONDITIONS,
    BACKTRACK_COND_LABELS,
    backtrack_cell_xticklabels,
    configure_matplotlib,
    ordered_backtrack_cells,
)
from jspace.records import read_jsonl

DEFAULT_TRACES = [
    ROOT / "runs/2026-07-26_gsm8k-pilot/grid_gsm8k_pilot/traces.jsonl",
    ROOT / "runs/2026-07-27_math500-pilot/grid_math500_pilot/traces.jsonl",
    ROOT / "runs/2026-07-27_math500-pilot/grid_aime_pilot/traces.jsonl",
]


def load_trace_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    return rows


def write_hand_label_template(path: Path, spans) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for span in spans[:50]:
            f.write(
                json.dumps(
                    {
                        "span_id": span.span_id,
                        "text": span.text,
                        "is_backtrack": None,
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def plot_regex_rates(summary: dict, out_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    cells = ordered_backtrack_cells(summary["cells"])
    x = list(range(len(cells)))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, condition in zip(offsets, BACKTRACK_CONDITIONS):
        rates = []
        ns = []
        for cell in cells:
            key = f"{cell}::{condition}"
            stratum = summary["strata"].get(key)
            rates.append(stratum["mean_rate"] if stratum else 0.0)
            ns.append(stratum["n_traces"] if stratum else 0)
        bars = ax.bar(
            [xi + offset for xi in x],
            rates,
            width=width,
            label=BACKTRACK_COND_LABELS[condition],
            color=BACKTRACK_COND_COLORS[condition],
        )
        for bar, n in zip(bars, ns):
            if n and n < 15:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(backtrack_cell_xticklabels(cells), rotation=20, ha="right")
    ax.set_ylabel("mean regex backtrack rate (count / tokens)")
    ax.set_title("Regex headline backtrack rate by cell × CoT condition")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="*",
        default=DEFAULT_TRACES,
        help="one or more traces.jsonl files (default: gsm8k + math500 + aime pilots)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs/backtrack-analysis",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=ROOT / "report/figures/supplementary",
    )
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = load_trace_rows(list(args.traces))
    summary = summarize_regex_backtracks(rows)
    all_spans, strata = spans_from_rows(rows)
    sample = stratified_span_sample(
        all_spans,
        strata,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "regex_summary.json"
    summary["n_candidate_spans"] = len(all_spans)
    summary["n_judge_sample"] = len(sample)
    summary["per_stratum"] = args.per_stratum
    summary["trace_sources"] = [str(p) for p in args.traces]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    jobs_path = out_dir / "judge_sample_jobs.jsonl"
    tmpl = load_judge_prompt_template()
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as f:
        for span in sample:
            job = {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "stratum": strata[span.span_id],
                "marker": span.marker,
                "text": span.text,
                "prompt": assemble_judge_prompt(span.text, tmpl),
            }
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    hand_path = out_dir / "hand_label_template.jsonl"
    write_hand_label_template(hand_path, sample)

    plot_path = args.plot_dir / "backtrack_regex_rates.png"
    if not args.no_plot:
        plot_regex_rates(summary, plot_path)

    print(f"wrote regex summary → {summary_path}")
    print(f"wrote {len(all_spans)} candidate spans; judge sample n={len(sample)} → {jobs_path}")
    print(f"wrote hand-label template (≤50) → {hand_path}")
    if not args.no_plot:
        print(f"wrote plot → {plot_path}")
    print(
        json.dumps(
            {
                "n_cot_traces": summary["n_cot_traces"],
                "n_candidate_spans": len(all_spans),
                "n_judge_sample": len(sample),
                "overall_mean_rate": summary["overall_mean_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
