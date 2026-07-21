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
