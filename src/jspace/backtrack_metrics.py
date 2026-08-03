"""Regex backtrack aggregation and stratified judge sampling (§4.4)."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from jspace.backtrack import (
    BacktrackSpan,
    find_candidate_spans,
    normalized_backtrack_rate,
    regex_backtrack_count,
)

COT_CONDITIONS = frozenset({"cot_clean", "cot_j_ablated", "cot_random"})


@dataclass(frozen=True)
class TraceBacktrack:
    problem_id: str
    condition: str
    cell: str
    count: int
    rate: float
    trace_length_tokens: int


def cell_key(row: dict[str, Any]) -> str:
    extra = row.get("extra") or {}
    if cell := extra.get("cell"):
        return str(cell)
    dataset = str(row.get("dataset", "unknown"))
    difficulty = row.get("difficulty")
    if dataset == "math500" and difficulty is not None:
        return f"math500_L{difficulty}"
    return dataset


def is_cot_row(row: dict[str, Any]) -> bool:
    if not row.get("enable_thinking", False):
        return False
    return str(row.get("condition", "")) in COT_CONDITIONS


def cot_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if is_cot_row(row)]


def backtrack_from_row(row: dict[str, Any]) -> tuple[int, float]:
    trace = str(row.get("raw_trace", ""))
    length = int(row.get("trace_length_tokens") or 0)
    stored_count = row.get("backtrack_count")
    count = int(stored_count) if stored_count is not None else regex_backtrack_count(trace)
    stored_rate = row.get("backtrack_rate")
    rate = (
        float(stored_rate)
        if stored_rate is not None
        else normalized_backtrack_rate(count, length)
    )
    return count, rate


def trace_backtracks(rows: Iterable[dict[str, Any]]) -> list[TraceBacktrack]:
    out: list[TraceBacktrack] = []
    for row in cot_rows(rows):
        count, rate = backtrack_from_row(row)
        out.append(
            TraceBacktrack(
                problem_id=str(row["problem_id"]),
                condition=str(row["condition"]),
                cell=cell_key(row),
                count=count,
                rate=rate,
                trace_length_tokens=int(row.get("trace_length_tokens") or 0),
            )
        )
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize_regex_backtracks(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    traces = trace_backtracks(rows)
    by_cell_cond: dict[tuple[str, str], list[TraceBacktrack]] = defaultdict(list)
    for trace in traces:
        by_cell_cond[(trace.cell, trace.condition)].append(trace)

    cells = sorted({trace.cell for trace in traces})
    conditions = sorted({trace.condition for trace in traces})
    strata: dict[str, dict[str, Any]] = {}
    for cell in cells:
        for condition in conditions:
            bucket = by_cell_cond.get((cell, condition), [])
            if not bucket:
                continue
            key = f"{cell}::{condition}"
            strata[key] = {
                "cell": cell,
                "condition": condition,
                "n_traces": len(bucket),
                "mean_count": _mean([float(t.count) for t in bucket]),
                "mean_rate": _mean([t.rate for t in bucket]),
                "total_markers": sum(t.count for t in bucket),
            }

    return {
        "n_cot_traces": len(traces),
        "cells": cells,
        "conditions": conditions,
        "strata": strata,
        "overall_mean_rate": _mean([t.rate for t in traces]),
        "overall_mean_count": _mean([float(t.count) for t in traces]),
    }


def spans_from_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[BacktrackSpan], dict[str, str]]:
    spans: list[BacktrackSpan] = []
    strata: dict[str, str] = {}
    for row in cot_rows(rows):
        trace_id = f"{row['problem_id']}::{row['condition']}"
        stratum = f"{cell_key(row)}::{row['condition']}"
        for span in find_candidate_spans(str(row.get("raw_trace", "")), trace_id=trace_id):
            spans.append(span)
            strata[span.span_id] = stratum
    return spans, strata


def stratified_span_sample(
    spans: list[BacktrackSpan],
    strata: dict[str, str],
    *,
    per_stratum: int = 5,
    seed: int = 0,
    exclude_span_ids: set[str] | None = None,
) -> list[BacktrackSpan]:
    exclude = exclude_span_ids or set()
    by_stratum: dict[str, list[BacktrackSpan]] = defaultdict(list)
    for span in spans:
        if span.span_id in exclude:
            continue
        by_stratum[strata[span.span_id]].append(span)

    rng = random.Random(seed)
    sample: list[BacktrackSpan] = []
    for stratum in sorted(by_stratum):
        bucket = by_stratum[stratum]
        rng.shuffle(bucket)
        sample.extend(bucket[:per_stratum])
    return sample


def wilson_ci(pos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = pos / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def precision_by_stratum(
    jobs: list[dict[str, Any]],
    verdicts: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    """PPV (judge positive | regex candidate) per cell::condition stratum."""
    by_stratum: dict[str, list[bool]] = defaultdict(list)
    for job in jobs:
        sid = str(job["span_id"])
        if sid not in verdicts:
            continue
        key = str(job.get("stratum", "unknown"))
        by_stratum[key].append(verdicts[sid])

    out: dict[str, dict[str, Any]] = {}
    for key in sorted(by_stratum):
        labels = by_stratum[key]
        n = len(labels)
        pos = sum(labels)
        lo, hi = wilson_ci(pos, n)
        cell, condition = key.split("::", 1)
        out[key] = {
            "cell": cell,
            "condition": condition,
            "n": n,
            "positives": pos,
            "ppv": pos / n if n else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
        }
    return out


def ppv_lookup(
    precision: dict[str, dict[str, Any]],
    cell: str,
    condition: str,
    *,
    default: float,
) -> float:
    row = precision.get(f"{cell}::{condition}")
    if row is None:
        return default
    return float(row["ppv"])


@dataclass(frozen=True)
class AdjustedTraceRate:
    problem_id: str
    condition: str
    cell: str
    regex_rate: float
    adjusted_rate: float


def adjusted_trace_rates(
    rows: Iterable[dict[str, Any]],
    precision: dict[str, dict[str, Any]],
) -> list[AdjustedTraceRate]:
    judged = [row for row in precision.values() if row.get("n", 0) > 0]
    total_pos = sum(int(row["positives"]) for row in judged)
    total_n = sum(int(row["n"]) for row in judged)
    default_ppv = total_pos / total_n if total_n else 0.0

    out: list[AdjustedTraceRate] = []
    for row in cot_rows(rows):
        cell = cell_key(row)
        condition = str(row["condition"])
        _, regex_rate = backtrack_from_row(row)
        ppv = ppv_lookup(precision, cell, condition, default=default_ppv)
        out.append(
            AdjustedTraceRate(
                problem_id=str(row["problem_id"]),
                condition=condition,
                cell=cell,
                regex_rate=regex_rate,
                adjusted_rate=regex_rate * ppv,
            )
        )
    return out


def bootstrap_adjusted_means(
    traces: list[AdjustedTraceRate],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Bootstrap mean adjusted rate per cell::condition stratum."""
    by_key: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {}
    for trace in traces:
        key = f"{trace.cell}::{trace.condition}"
        by_key[key].append(trace.adjusted_rate)
        meta[key] = {"cell": trace.cell, "condition": trace.condition}

    rng = random.Random(seed)
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(by_key):
        rates = by_key[key]
        n = len(rates)
        point = _mean(rates)
        if n == 0:
            continue
        boot_means = sorted(
            _mean([rates[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
        )
        out[key] = {
            **meta[key],
            "n_traces": n,
            "mean_adjusted_rate": point,
            "ci_lo": boot_means[int(0.025 * n_boot)],
            "ci_hi": boot_means[int(0.975 * n_boot)],
        }
    return out
