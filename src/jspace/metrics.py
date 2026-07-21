"""Accuracy helpers, graded logprob stub, bootstrap placeholder."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from jspace.extract import answers_equal


def accuracy(correct_flags: Sequence[bool]) -> float:
    if not correct_flags:
        return float("nan")
    return float(sum(correct_flags) / len(correct_flags))


def mean_logprob(token_logprobs: Sequence[float]) -> float | None:
    if not token_logprobs:
        return None
    return float(np.mean(token_logprobs))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI over problem-level values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def score_prediction(dataset: str, predicted: str | None, gold: str) -> bool:
    return answers_equal(dataset, predicted, gold)
