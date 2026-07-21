#!/usr/bin/env python3
"""Local preview: harness + mini condition grid with calibrated token budgets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.ablation import AblationConfig
from jspace.config import load_settings
from jspace.data import load_gsm8k
from jspace.load import load_hf_model, load_model_and_lens
from jspace.metrics import accuracy
from jspace.progress import RunProgress, reset_jsonl
from jspace.records import append_jsonl
from jspace.run_one import run_problem
from jspace.token_budgets import (
    calibration_path,
    calibrate_token_budgets,
    load_token_budgets,
    resolve_max_new_tokens,
    save_token_budgets,
    summarize_profile,
)


def _run_label(problem_id: str, thinking: bool, ablation_kind: str, budget: int) -> str:
    mode = "CoT" if thinking else "direct"
    slow = " [J-ablation: slow]" if ablation_kind == "jspace" else ""
    return f"{problem_id} {mode} {ablation_kind} budget={budget}{slow}"


def _record_summary(rec) -> str:
    return (
        f"ok={rec.correct} ext={rec.extraction_success} "
        f"cap={rec.hit_token_cap} len={rec.trace_length_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--skip-grid", action="store_true", help="harness only")
    parser.add_argument("--grid-only", action="store_true", help="skip harness section")
    parser.add_argument(
        "--local-fast",
        action="store_true",
        help="cap ablated runs at local_fast_ablation_cap (preview only; clean runs keep calibrated budgets)",
    )
    args = parser.parse_args()
    if args.skip_grid and args.grid_only:
        raise SystemExit("use at most one of --skip-grid / --grid-only")

    settings = load_settings()
    out_dir = settings.results_dir / "preview"
    log_path = out_dir / "progress.log"
    profile = load_token_budgets(calibration_path(settings))

    progress = RunProgress(log_path, section="preview", total=0)
    fast_note = (
        f" local_fast ablation_cap={settings.local_fast_ablation_cap}"
        if args.local_fast
        else ""
    )
    progress.emit(f"model={settings.model_name} limit={args.limit}{fast_note}")

    loaded = load_hf_model(settings)
    if args.calibrate or profile is None:
        progress.emit("calibrating token budgets from unablated traces...")
        profile = calibrate_token_budgets(
            loaded,
            settings,
            datasets=("gsm8k",),
            problems_per_dataset=min(args.limit, 5),
        )
        save_token_budgets(calibration_path(settings), profile)
        progress.emit("token budgets: " + json.dumps(summarize_profile(profile)))

    problems = load_gsm8k(limit=args.limit)
    harness_path = out_dir / "harness.jsonl"
    grid_path = out_dir / "mini_grid.jsonl"
    harness_records = []

    if not args.grid_only:
        harness_total = len(problems) * 2
        harness_progress = RunProgress(log_path, section="harness", total=harness_total)
        reset_jsonl(harness_path)
        for problem in problems:
            for thinking in (False, True):
                budget = resolve_max_new_tokens(
                    settings, problem.dataset, enable_thinking=thinking, profile=profile
                )
                label = _run_label(problem.problem_id, thinking, "none", budget)
                timer = harness_progress.start_run(label)
                rec = run_problem(
                    loaded,
                    problem,
                    settings=settings,
                    enable_thinking=thinking,
                    ablation=AblationConfig(kind="none"),
                    seed=settings.seed,
                    token_profile=profile,
                )
                append_jsonl(harness_path, rec)
                harness_records.append(rec)
                timer.finish(_record_summary(rec))

    if args.skip_grid:
        progress.emit(f"finished harness → {harness_path}")
        return

    lens_loaded = load_model_and_lens(settings)
    n = lens_loaded.n_layers
    band_start, band_end = n // 3, (2 * n) // 3
    grid_records = []
    grid_total = len(problems) * 2 * 2
    grid_progress = RunProgress(log_path, section="mini_grid", total=grid_total)
    reset_jsonl(grid_path)
    progress.emit(
        f"grid band=[{band_start},{band_end}] k=5 "
        f"(J-ablated runs are ~2× forwards per token — expect minutes each)"
    )

    for problem in problems:
        for thinking in (False, True):
            for kind in ("none", "jspace"):
                budget = resolve_max_new_tokens(
                    settings,
                    problem.dataset,
                    enable_thinking=thinking,
                    profile=profile,
                    ablation_kind=kind,
                    local_fast=args.local_fast,
                )
                label = _run_label(problem.problem_id, thinking, kind, budget)
                timer = grid_progress.start_run(label)
                rec = run_problem(
                    lens_loaded,
                    problem,
                    settings=settings,
                    enable_thinking=thinking,
                    ablation=AblationConfig(
                        kind=kind,  # type: ignore[arg-type]
                        band_start=band_start,
                        band_end=band_end,
                        k=5,
                        ablate_prompt_tokens=True,
                    ),
                    seed=settings.seed,
                    token_profile=profile,
                    local_fast=args.local_fast,
                )
                append_jsonl(grid_path, rec)
                grid_records.append(rec)
                timer.finish(_record_summary(rec))

    by_cond: dict[str, list[bool | None]] = defaultdict(list)
    for rec in grid_records:
        by_cond[rec.condition].append(rec.correct)
    progress.emit("=== grid accuracy ===")
    for cond in sorted(by_cond):
        flags = [c for c in by_cond[cond] if c is not None]
        progress.emit(f"  {cond:16} {accuracy(flags):.0%}")

    total = harness_records + grid_records
    caps = sum(r.hit_token_cap for r in total)
    ext = sum(not r.extraction_success for r in total)
    progress.emit(
        f"FINISH cap_hits={caps} extract_failures={ext} "
        f"runs={len(total)} log={log_path}"
    )
    progress.emit(f"wrote {harness_path} and {grid_path}")


if __name__ == "__main__":
    main()
