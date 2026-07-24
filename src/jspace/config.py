"""Single source of truth for model, lens, device, band, and run settings (§5.3)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RUBRICS_DIR = REPO_ROOT / "rubrics"

# Homework-fixed dataset repos (§4.8) — not an open choice.
GSM8K_REPO = "openai/gsm8k"
MATH500_REPO = "HuggingFaceH4/MATH-500"
AIME_REPO = "HuggingFaceH4/aime_2024"
AIME_N = 30

DEFAULT_GSM8K_SPLIT = "test"
DEFAULT_MATH500_SPLIT = "test"
DEFAULT_AIME_SPLIT = "train"

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILENAME = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"

MAIN_MODEL = "Qwen/Qwen3-4B"
DEV_MODEL = "Qwen/Qwen3-0.6B"
SIZE_CONTROL_MODEL = "Qwen/Qwen3-1.7B"

DEFAULT_K = 10
DEFAULT_RANDOM_SEEDS = (0, 1, 2)
DEFAULT_PROBLEMS_PER_CELL = 50
FLOOR_ACCURACY_THRESHOLD = 0.10
DEFAULT_TOKEN_BUDGET_MULTIPLIER = 2.0
# Ablated traces may run longer than clean (compensation); floor vs observed p95.
DEFAULT_ABLATED_TOKEN_BUDGET_MULTIPLIER = 3.0
DEFAULT_TOKEN_BUDGET_PERCENTILE = 0.95
DEFAULT_TOKEN_BUDGET_CEILING = 8192
# Probe lengths without censoring at the final cost ceiling (§4.7).
DEFAULT_TOKEN_BUDGET_PROBE_CEILING = 32768
# Local MPS preview only: ablated runs use 2× forwards/token — cap to finish in ~1hr.
DEFAULT_LOCAL_FAST_ABLATION_CAP = 512


@dataclass(frozen=True)
class Settings:
    """Env-overridable run settings. Band overrides win over auto-selection."""

    model_name: str = DEV_MODEL
    model_revision: str | None = None
    lens_repo: str = LENS_REPO
    lens_filename: str = LENS_FILENAME
    lens_revision: str | None = None
    gsm8k_revision: str | None = None
    math500_revision: str | None = None
    aime_revision: str | None = None
    device: str | None = None
    dtype: str | None = None
    attn_implementation: str = "eager"
    band_start: int | None = None
    band_end: int | None = None
    k: int = DEFAULT_K
    seed: int = 0
    random_ablation_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    # Token budgets (§4.7): per-dataset caps come from calibration/token_budgets.json.
    # max_new_tokens_override forces a single cap for all arms (debug only).
    max_new_tokens_override: int | None = None
    token_budget_multiplier: float = DEFAULT_TOKEN_BUDGET_MULTIPLIER
    ablated_token_budget_multiplier: float = DEFAULT_ABLATED_TOKEN_BUDGET_MULTIPLIER
    token_budget_percentile: float = DEFAULT_TOKEN_BUDGET_PERCENTILE
    token_budget_ceiling: int = DEFAULT_TOKEN_BUDGET_CEILING
    token_budget_probe_ceiling: int = DEFAULT_TOKEN_BUDGET_PROBE_CEILING
    local_fast_ablation_cap: int = DEFAULT_LOCAL_FAST_ABLATION_CAP
    problems_per_cell: int = DEFAULT_PROBLEMS_PER_CELL
    floor_accuracy_threshold: float = FLOOR_ACCURACY_THRESHOLD
    ablate_prompt_tokens: bool = True
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    def with_overrides(self, **kwargs: object) -> Settings:
        return replace(self, **kwargs)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else None


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw if raw is not None and raw != "" else None


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw != "" else None


def load_settings() -> Settings:
    """Build Settings from defaults and JSPACE_* environment variables."""
    override = _env_int("JSPACE_MAX_NEW_TOKENS")
    return Settings(
        model_name=_env_str("JSPACE_MODEL") or DEV_MODEL,
        model_revision=_env_str("JSPACE_MODEL_REVISION"),
        lens_repo=_env_str("JSPACE_LENS_REPO") or LENS_REPO,
        lens_filename=_env_str("JSPACE_LENS_FILE") or LENS_FILENAME,
        lens_revision=_env_str("JSPACE_LENS_REVISION"),
        gsm8k_revision=_env_str("JSPACE_GSM8K_REVISION"),
        math500_revision=_env_str("JSPACE_MATH500_REVISION"),
        aime_revision=_env_str("JSPACE_AIME_REVISION"),
        device=_env_str("JSPACE_DEVICE"),
        dtype=_env_str("JSPACE_DTYPE"),
        band_start=_env_int("JSPACE_BAND_START"),
        band_end=_env_int("JSPACE_BAND_END"),
        k=_env_int("JSPACE_K") or DEFAULT_K,
        seed=_env_int("JSPACE_SEED") or 0,
        max_new_tokens_override=override,
        token_budget_multiplier=_env_float("JSPACE_TOKEN_BUDGET_MULTIPLIER")
        or DEFAULT_TOKEN_BUDGET_MULTIPLIER,
        ablated_token_budget_multiplier=_env_float(
            "JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER"
        )
        or DEFAULT_ABLATED_TOKEN_BUDGET_MULTIPLIER,
        token_budget_percentile=_env_float("JSPACE_TOKEN_BUDGET_PERCENTILE")
        or DEFAULT_TOKEN_BUDGET_PERCENTILE,
        token_budget_ceiling=_env_int("JSPACE_TOKEN_BUDGET_CEILING")
        or DEFAULT_TOKEN_BUDGET_CEILING,
        token_budget_probe_ceiling=_env_int("JSPACE_TOKEN_BUDGET_PROBE_CEILING")
        or DEFAULT_TOKEN_BUDGET_PROBE_CEILING,
        local_fast_ablation_cap=_env_int("JSPACE_LOCAL_FAST_ABLATION_CAP")
        or DEFAULT_LOCAL_FAST_ABLATION_CAP,
        problems_per_cell=_env_int("JSPACE_PROBLEMS_PER_CELL")
        or DEFAULT_PROBLEMS_PER_CELL,
    )
