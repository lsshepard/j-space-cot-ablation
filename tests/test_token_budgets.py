from jspace.config import Settings
from jspace.token_budgets import (
    DatasetTokenCaps,
    TokenBudgetProfile,
    _apply_multiplier,
    _percentile,
    resolve_max_new_tokens,
)


def test_percentile_and_multiplier():
    assert _percentile([10, 20, 30, 40, 100], 0.95) == 100
    assert _apply_multiplier(50, 2.0, ceiling=200, floor=64) == 100
    assert _apply_multiplier(50, 2.0, ceiling=80, floor=64) == 80


def test_resolve_uses_profile_per_mode():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=8192,
        datasets={"gsm8k": DatasetTokenCaps(direct=128, cot=1024)},
    )
    settings = Settings()
    assert resolve_max_new_tokens(
        settings, "gsm8k", enable_thinking=False, profile=profile
    ) == 128
    assert resolve_max_new_tokens(
        settings, "gsm8k", enable_thinking=True, profile=profile
    ) == 1024


def test_override_wins():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=8192,
        datasets={"gsm8k": DatasetTokenCaps(direct=128, cot=1024)},
    )
    settings = Settings(max_new_tokens_override=512)
    assert (
        resolve_max_new_tokens(
            settings, "gsm8k", enable_thinking=True, profile=profile
        )
        == 512
    )


def test_local_fast_caps_ablated_only():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=8192,
        datasets={"gsm8k": DatasetTokenCaps(direct=422, cot=2370)},
    )
    settings = Settings(local_fast_ablation_cap=512)
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=True,
            profile=profile,
            ablation_kind="none",
            local_fast=True,
        )
        == 2370
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=True,
            profile=profile,
            ablation_kind="jspace",
            local_fast=True,
        )
        == 512
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=False,
            profile=profile,
            ablation_kind="jspace",
            local_fast=True,
        )
        == 422
    )


def test_ablated_cap_at_least_three_times_observed_p95():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=8192,
        datasets={
            "gsm8k": DatasetTokenCaps(
                direct=200,
                cot=2000,
                direct_p95_observed=100,
                cot_p95_observed=1000,
            )
        },
    )
    settings = Settings(ablated_token_budget_multiplier=3.0)
    assert (
        resolve_max_new_tokens(
            settings, "gsm8k", enable_thinking=True, profile=profile, ablation_kind="none"
        )
        == 2000
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=True,
            profile=profile,
            ablation_kind="jspace",
        )
        == 3000
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=False,
            profile=profile,
            ablation_kind="random",
        )
        == 300
    )


def test_ablated_cap_respects_profile_ceiling():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=2500,
        datasets={
            "gsm8k": DatasetTokenCaps(
                direct=200,
                cot=2000,
                cot_p95_observed=1000,
            )
        },
    )
    settings = Settings(
        ablated_token_budget_multiplier=3.0,
        token_budget_ceiling=16384,
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "gsm8k",
            enable_thinking=True,
            profile=profile,
            ablation_kind="jspace",
        )
        == 2500
    )


def test_ablated_cap_respects_settings_ceiling_tighten():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=16384,
        datasets={
            "math500": DatasetTokenCaps(
                direct=1360,
                cot=9302,
                cot_p95_observed=4651,
            )
        },
    )
    settings = Settings(
        ablated_token_budget_multiplier=6.0,
        token_budget_ceiling=10000,
    )
    assert (
        resolve_max_new_tokens(
            settings,
            "math500",
            enable_thinking=True,
            profile=profile,
            ablation_kind="jspace",
        )
        == 10000
    )


def test_global_ceiling_also_caps_clean_cot():
    profile = TokenBudgetProfile(
        model_name="test",
        multiplier=2.0,
        ceiling=16384,
        datasets={
            "math500": DatasetTokenCaps(
                direct=1360,
                cot=9302,
                cot_p95_observed=4651,
            )
        },
    )
    settings = Settings(token_budget_ceiling=8000)
    assert (
        resolve_max_new_tokens(
            settings,
            "math500",
            enable_thinking=True,
            profile=profile,
            ablation_kind="none",
        )
        == 8000
    )
