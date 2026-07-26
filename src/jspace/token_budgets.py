"""Per-dataset, per-mode token caps from unablated calibration (§4.7)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jspace.config import Settings
from jspace.data import Problem, load_aime, load_gsm8k, load_math500
from jspace.generate import generate_clean
from jspace.load import LoadedModel

# Conservative fallbacks before any calibration run (direct << CoT for Qwen3 thinking).
_DEFAULT_CAPS: dict[str, dict[str, int]] = {
    "gsm8k": {"direct": 256, "cot": 2048},
    "math500": {"direct": 512, "cot": 4096},
    "aime": {"direct": 512, "cot": 8192},
    "multihop": {"direct": 64, "cot": 64},
}

_DATASET_LOADERS = {
    "gsm8k": lambda limit: load_gsm8k(limit=limit),
    "math500": lambda limit: load_math500(limit=limit),
    "aime": lambda limit: load_aime(limit=limit),
}


@dataclass(frozen=True)
class DatasetTokenCaps:
    direct: int
    cot: int
    direct_p95_observed: int | None = None
    cot_p95_observed: int | None = None


@dataclass(frozen=True)
class TokenBudgetProfile:
    """Saved calibration artifact: caps = ceil(p95_unablated * multiplier)."""

    model_name: str
    multiplier: float
    ceiling: int
    datasets: dict[str, DatasetTokenCaps]

    def cap_for(self, dataset: str, *, enable_thinking: bool) -> int:
        if dataset not in self.datasets:
            raise KeyError(f"no token budget for dataset {dataset!r}")
        caps = self.datasets[dataset]
        return caps.cot if enable_thinking else caps.direct


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return int(ordered[rank])


def _apply_multiplier(observed_p95: int, multiplier: float, ceiling: int, floor: int) -> int:
    return min(ceiling, max(floor, math.ceil(observed_p95 * multiplier)))


def default_profile(model_name: str, *, multiplier: float = 2.0, ceiling: int = 8192) -> TokenBudgetProfile:
    return TokenBudgetProfile(
        model_name=model_name,
        multiplier=multiplier,
        ceiling=ceiling,
        datasets={
            name: DatasetTokenCaps(direct=vals["direct"], cot=vals["cot"])
            for name, vals in _DEFAULT_CAPS.items()
        },
    )


def calibration_path(settings: Settings) -> Path:
    return settings.results_dir / "calibration" / "token_budgets.json"


def save_token_budgets(path: Path, profile: TokenBudgetProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": profile.model_name,
        "multiplier": profile.multiplier,
        "ceiling": profile.ceiling,
        "datasets": {
            name: asdict(caps) for name, caps in profile.datasets.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_token_budgets(path: Path) -> TokenBudgetProfile | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = {
        name: DatasetTokenCaps(**vals) for name, vals in payload["datasets"].items()
    }
    return TokenBudgetProfile(
        model_name=payload["model_name"],
        multiplier=float(payload["multiplier"]),
        ceiling=int(payload["ceiling"]),
        datasets=datasets,
    )


def resolve_max_new_tokens(
    settings: Settings,
    dataset: str,
    *,
    enable_thinking: bool,
    profile: TokenBudgetProfile | None = None,
    ablation_kind: str = "none",
    local_fast: bool = False,
) -> int:
    """Return the token cap for this dataset, thinking mode, and ablation arm."""
    if settings.max_new_tokens_override is not None:
        return settings.max_new_tokens_override

    if profile is None:
        profile = load_token_budgets(calibration_path(settings))

    observed_p95: int | None = None
    if profile is not None and dataset in profile.datasets:
        caps = profile.datasets[dataset]
        base = profile.cap_for(dataset, enable_thinking=enable_thinking)
        observed_p95 = (
            caps.cot_p95_observed if enable_thinking else caps.direct_p95_observed
        )
    else:
        fallback = _DEFAULT_CAPS.get(dataset, _DEFAULT_CAPS["gsm8k"])
        key = "cot" if enable_thinking else "direct"
        base = fallback[key]

    if ablation_kind != "none" and observed_p95 is not None:
        ablated_floor = math.ceil(
            observed_p95 * settings.ablated_token_budget_multiplier
        )
        ceiling = profile.ceiling if profile is not None else settings.token_budget_ceiling
        base = min(ceiling, max(base, ablated_floor))

    if local_fast and ablation_kind != "none":
        return min(base, settings.local_fast_ablation_cap)
    return base


def _measure_clean_lengths(
    loaded: LoadedModel,
    problems: list[Problem],
    *,
    probe_ceiling: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    direct_lengths: list[int] = []
    cot_lengths: list[int] = []
    n_problems = len(problems)
    total = max(1, n_problems * 2)
    done = 0
    t0 = time.perf_counter()
    for i, problem in enumerate(problems):
        for thinking, bucket in ((False, direct_lengths), (True, cot_lengths)):
            mode = "cot" if thinking else "direct"
            done += 1
            print(
                f"[calibrate] start {problem.dataset} "
                f"problem={i + 1}/{n_problems} mode={mode} "
                f"step={done}/{total} probe_ceiling={probe_ceiling}",
                flush=True,
            )
            t1 = time.perf_counter()
            # Length-only probe: skip score tensors (huge VRAM/time at 32k).
            result = generate_clean(
                loaded,
                problem.dataset,
                problem.prompt,
                enable_thinking=thinking,
                max_new_tokens=probe_ceiling,
                seed=seed,
                capture_logprobs=False,
            )
            n_tokens = len(result.token_ids)
            bucket.append(n_tokens)
            print(
                f"[calibrate] done  {problem.dataset} "
                f"problem={i + 1}/{n_problems} mode={mode} "
                f"tokens={n_tokens} hit_cap={result.hit_token_cap} "
                f"gen_sec={time.perf_counter() - t1:.1f} "
                f"elapsed_sec={time.perf_counter() - t0:.1f}",
                flush=True,
            )
    return direct_lengths, cot_lengths


def calibrate_token_budgets(
    loaded: LoadedModel,
    settings: Settings,
    *,
    datasets: tuple[str, ...] = ("gsm8k", "math500", "aime"),
    problems_per_dataset: int = 5,
    percentile: float | None = None,
    multiplier: float | None = None,
    ceiling: int | None = None,
    probe_ceiling: int | None = None,
) -> TokenBudgetProfile:
    """
    Measure unablated clean trace lengths; set caps = ceil(p95 * multiplier) (§4.7).

    Probe generation uses probe_ceiling (high) so p95 is not censored by the
    final cost ceiling applied when writing caps.
    """
    pct = percentile if percentile is not None else settings.token_budget_percentile
    mult = multiplier if multiplier is not None else settings.token_budget_multiplier
    cap_ceiling = ceiling if ceiling is not None else settings.token_budget_ceiling
    measure_ceiling = (
        probe_ceiling
        if probe_ceiling is not None
        else settings.token_budget_probe_ceiling
    )

    calibrated: dict[str, DatasetTokenCaps] = {}
    for dataset in datasets:
        if dataset not in _DATASET_LOADERS:
            raise ValueError(f"unsupported dataset for calibration: {dataset}")
        problems = _DATASET_LOADERS[dataset](problems_per_dataset)
        direct_lens, cot_lens = _measure_clean_lengths(
            loaded,
            problems,
            probe_ceiling=measure_ceiling,
            seed=settings.seed,
        )
        direct_p95 = _percentile(direct_lens, pct)
        cot_p95 = _percentile(cot_lens, pct)
        floor_direct = _DEFAULT_CAPS.get(dataset, _DEFAULT_CAPS["gsm8k"])["direct"]
        floor_cot = _DEFAULT_CAPS.get(dataset, _DEFAULT_CAPS["gsm8k"])["cot"]
        calibrated[dataset] = DatasetTokenCaps(
            direct=_apply_multiplier(direct_p95, mult, cap_ceiling, floor_direct),
            cot=_apply_multiplier(cot_p95, mult, cap_ceiling, floor_cot),
            direct_p95_observed=direct_p95,
            cot_p95_observed=cot_p95,
        )
        print(
            f"[calibrate] dataset={dataset} "
            f"direct_lens={direct_lens} cot_lens={cot_lens} "
            f"direct_p95={direct_p95} cot_p95={cot_p95} "
            f"caps=({calibrated[dataset].direct}, {calibrated[dataset].cot})",
            flush=True,
        )

    return TokenBudgetProfile(
        model_name=loaded.model_name,
        multiplier=mult,
        ceiling=cap_ceiling,
        datasets=calibrated,
    )


def summarize_profile(profile: TokenBudgetProfile) -> dict[str, Any]:
    return {
        "model_name": profile.model_name,
        "multiplier": profile.multiplier,
        "ceiling": profile.ceiling,
        "datasets": {
            name: {
                "direct": caps.direct,
                "cot": caps.cot,
                "direct_p95_observed": caps.direct_p95_observed,
                "cot_p95_observed": caps.cot_p95_observed,
            }
            for name, caps in profile.datasets.items()
        },
    }
