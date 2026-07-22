"""Tests for clean-exclusion merge + KV vs full-prefix decode equivalence."""

from __future__ import annotations

import pytest
import torch

from jspace.ablation import (
    AblationConfig,
    AblationHookState,
    ablation_hooks,
    build_ablation_factors,
    clean_topk_by_position,
    past_token_count_from_cache,
)
from jspace.generate import _append_clean_exclusions, _extend_attention_mask


def test_append_clean_exclusions_first_chunk_replaces():
    state = AblationHookState(prompt_token_count=3)
    _append_clean_exclusions(state, [{1}, {2}, {3}], past_token_count=0)
    assert state.excluded_by_position == [{1}, {2}, {3}]


def test_append_clean_exclusions_extends_in_lockstep():
    state = AblationHookState(prompt_token_count=2)
    state.excluded_by_position = [{1}, {2}]
    _append_clean_exclusions(state, [{9}], past_token_count=2)
    assert state.excluded_by_position == [{1}, {2}, {9}]


def test_append_clean_exclusions_detects_desync():
    state = AblationHookState(prompt_token_count=2)
    state.excluded_by_position = [{1}]
    with pytest.raises(RuntimeError, match="desynced"):
        _append_clean_exclusions(state, [{9}], past_token_count=2)


def test_clean_topk_zero_returns_empty_sets():
    logits = torch.zeros(1, 3, 10)
    assert clean_topk_by_position(logits, k=0) == [set(), set(), set()]


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_kv_cache_matches_full_prefix_teacher_forced_logits():
    """
    Along a fixed token path, ablated next-token logits from KV decode must
    match full-prefix recompute (separate clean vs ablated caches).
    """
    from jspace.config import load_settings
    from jspace.generate import build_chat_text
    from jspace.load import load_model_and_lens

    settings = load_settings()
    loaded = load_model_and_lens(settings)
    mid = loaded.n_layers // 2
    # Single layer keeps the reference path cheap on MPS.
    ablation = AblationConfig(
        kind="jspace",
        band_start=mid,
        band_end=mid,
        k=3,
        exclude_topk=5,
        ablate_prompt_tokens=True,
    )
    factors = build_ablation_factors(loaded.hf_model, loaded.lens, ablation)

    text = build_chat_text(
        loaded.tokenizer, "gsm8k", "What is 2 + 2?", enable_thinking=False
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(loaded.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(loaded.device)
    prompt_len = int(prompt_ids.shape[-1])

    # Fixed continuation (not model-sampled) so both paths see the same prefixes.
    forced = [10, 20, 30]

    # --- KV path ---
    kv_state = AblationHookState(prompt_token_count=prompt_len)
    clean_past = None
    ablated_past = None
    chunk = prompt_ids
    kv_mask = mask
    kv_logits: list[torch.Tensor] = []
    for step in range(len(forced) + 1):
        clean_past_len = past_token_count_from_cache(clean_past)
        clean_out = loaded.hf_model(
            input_ids=chunk,
            attention_mask=kv_mask,
            past_key_values=clean_past,
            use_cache=True,
        )
        clean_past = clean_out.past_key_values
        _append_clean_exclusions(
            kv_state,
            clean_topk_by_position(clean_out.logits, ablation.exclude_topk),
            past_token_count=clean_past_len,
        )
        kv_state.past_token_count = past_token_count_from_cache(ablated_past)
        with ablation_hooks(
            loaded.hf_model, loaded.lens, ablation, kv_state, factors=factors
        ):
            ablated_out = loaded.hf_model(
                input_ids=chunk,
                attention_mask=kv_mask,
                past_key_values=ablated_past,
                use_cache=True,
            )
        ablated_past = ablated_out.past_key_values
        kv_logits.append(ablated_out.logits[0, -1].float().cpu())
        if step == len(forced):
            break
        chunk = torch.tensor([[forced[step]]], device=loaded.device)
        kv_mask = _extend_attention_mask(kv_mask, chunk)

    # --- Full-prefix reference ---
    ref_state = AblationHookState(prompt_token_count=prompt_len)
    input_ids = prompt_ids
    ref_mask = mask
    ref_logits: list[torch.Tensor] = []
    for step in range(len(forced) + 1):
        clean_out = loaded.hf_model(input_ids=input_ids, attention_mask=ref_mask)
        ref_state.excluded_by_position = clean_topk_by_position(
            clean_out.logits, ablation.exclude_topk
        )
        ref_state.past_token_count = 0
        with ablation_hooks(
            loaded.hf_model, loaded.lens, ablation, ref_state, factors=factors
        ):
            ablated_out = loaded.hf_model(
                input_ids=input_ids, attention_mask=ref_mask
            )
        ref_logits.append(ablated_out.logits[0, -1].float().cpu())
        if step == len(forced):
            break
        next_token = torch.tensor([[forced[step]]], device=loaded.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        ref_mask = _extend_attention_mask(ref_mask, next_token)

    assert len(kv_logits) == len(ref_logits)
    for i, (kv, ref) in enumerate(zip(kv_logits, ref_logits)):
        max_err = float((kv - ref).abs().max())
        assert torch.allclose(kv, ref, atol=1e-3, rtol=1e-3), (
            f"logit mismatch at step {i}: max_abs_err={max_err}"
        )
