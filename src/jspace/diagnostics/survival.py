"""Direction-survival + residual Δh summaries (instrument probe 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from jspace.ablation import AblationStepDiag


@dataclass(frozen=True)
class SurvivalSummary:
    """Aggregate over ablated (layer, position) edits."""

    n_steps: int
    mean_n_active: float
    mean_n_survivors: float
    mean_survivor_fraction: float
    fraction_zero_survivors: float
    mean_delta_h_norm: float
    median_delta_h_norm: float
    fraction_near_zero_delta: float
    near_zero_eps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_survival(
    steps: Sequence[AblationStepDiag],
    *,
    near_zero_eps: float = 1e-6,
) -> SurvivalSummary:
    """
    Summarize how many J-dirs survive exclude_topk and how large ‖Δh‖ is.

    survivor_fraction uses n_survivors / max(n_active, 1) per step.
    """
    if not steps:
        return SurvivalSummary(
            n_steps=0,
            mean_n_active=float("nan"),
            mean_n_survivors=float("nan"),
            mean_survivor_fraction=float("nan"),
            fraction_zero_survivors=float("nan"),
            mean_delta_h_norm=float("nan"),
            median_delta_h_norm=float("nan"),
            fraction_near_zero_delta=float("nan"),
            near_zero_eps=near_zero_eps,
        )

    actives = np.asarray([s.n_active for s in steps], dtype=np.float64)
    survivors = np.asarray([s.n_survivors for s in steps], dtype=np.float64)
    deltas = np.asarray([s.delta_h_norm for s in steps], dtype=np.float64)
    frac = survivors / np.maximum(actives, 1.0)

    return SurvivalSummary(
        n_steps=len(steps),
        mean_n_active=float(actives.mean()),
        mean_n_survivors=float(survivors.mean()),
        mean_survivor_fraction=float(frac.mean()),
        fraction_zero_survivors=float(np.mean(survivors == 0)),
        mean_delta_h_norm=float(deltas.mean()),
        median_delta_h_norm=float(np.median(deltas)),
        fraction_near_zero_delta=float(np.mean(deltas <= near_zero_eps)),
        near_zero_eps=near_zero_eps,
    )


def survival_verdict(summary: SurvivalSummary) -> str:
    """Short interpretability label for probe 1."""
    if summary.n_steps == 0:
        return "no_edits_recorded"
    if (
        summary.fraction_zero_survivors >= 0.9
        or summary.fraction_near_zero_delta >= 0.9
    ):
        return "toothless_exclusion_or_null_edit"
    if summary.mean_n_survivors > 0 and summary.mean_delta_h_norm > summary.near_zero_eps:
        return "mechanically_active"
    return "weak_or_mixed"
