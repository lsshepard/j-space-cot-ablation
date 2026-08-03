"""Matplotlib setup safe for headless / sandboxed runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Backtrack bar charts: difficulty order, clean → random → J-ablated, EM-plot palette.
BACKTRACK_CELL_ORDER = (
    "gsm8k",
    "math500_L1",
    "math500_L3",
    "math500_L4",
    "math500_L5",
    "aime",
)

BACKTRACK_CELL_LABELS = {
    "gsm8k": "GSM8K",
    "math500_L1": "MATH L1",
    "math500_L3": "MATH L3",
    "math500_L4": "MATH L4",
    "math500_L5": "MATH L5",
    "aime": "AIME",
}

BACKTRACK_CONDITIONS = ("cot_clean", "cot_random", "cot_j_ablated")

BACKTRACK_COND_LABELS = {
    "cot_clean": "clean",
    "cot_random": "random",
    "cot_j_ablated": "J-ablated",
}

BACKTRACK_COND_COLORS = {
    "cot_clean": "#4c78a8",
    "cot_random": "#72b7b2",
    "cot_j_ablated": "#f58518",
}

# Report charts: difficulty order, clean → random → J-ablated palette.
REPORT_CELL_ORDER = BACKTRACK_CELL_ORDER
REPORT_CELL_LABELS = BACKTRACK_CELL_LABELS

REPORT_ARM_ORDER = ("clean", "random", "j_ablated")
REPORT_ARM_LABELS = {
    "clean": "clean",
    "random": "r-ablated",
    "j_ablated": "j-ablated",
}
REPORT_ARM_COLORS = {
    "clean": "#4c78a8",
    "random": "#72b7b2",
    "j_ablated": "#f58518",
}

DIRECT_CONDITIONS = ("direct_clean", "direct_random", "direct_j_ablated")
COT_CONDITIONS = ("cot_clean", "cot_random", "cot_j_ablated")

CONDITION_TO_ARM = {
    "direct_clean": "clean",
    "direct_random": "random",
    "direct_j_ablated": "j_ablated",
    "cot_clean": "clean",
    "cot_random": "random",
    "cot_j_ablated": "j_ablated",
}


def configure_matplotlib() -> None:
    """Call before importing pyplot. Avoids font-cache crashes on macOS."""
    cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mpl-jspace-cache"))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ.setdefault("MPLBACKEND", "Agg")


def ordered_backtrack_cells(cells: Iterable[str]) -> list[str]:
    present = set(cells)
    ordered = [cell for cell in BACKTRACK_CELL_ORDER if cell in present]
    ordered.extend(sorted(present - set(BACKTRACK_CELL_ORDER)))
    return ordered


def backtrack_cell_xticklabels(cells: list[str]) -> list[str]:
    return [BACKTRACK_CELL_LABELS.get(cell, cell) for cell in cells]


def report_cell_xticklabels(cells: list[str]) -> list[str]:
    return [REPORT_CELL_LABELS.get(cell, cell) for cell in cells]


def ordered_report_cells(cells: Iterable[str]) -> list[str]:
    return ordered_backtrack_cells(cells)


def style_report_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
