"""Generation with thinking toggle, logprobs, and optional per-token ablation (§4.5–4.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from jspace.ablation import (
    AblationConfig,
    AblationHookState,
    ablation_hooks,
    build_ablation_factors,
    clean_topk_by_position,
    past_token_count_from_cache,
)
from jspace.load import LoadedModel
from jspace.prompts import format_user, system_prompt


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    hit_token_cap: bool
    prompt_token_count: int
    hook_call_count: int = 0
    direct_leak_flag: bool = False
    prompt_input_ids: Any | None = None
    prompt_attention_mask: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_LEAK_MARKERS = (
    "step 1",
    "step 2",
    "first,",
    "second,",
    "let me",
    "let's",
    "we need to",
    "wait,",
)


def build_chat_text(
    tokenizer: Any,
    dataset: str,
    problem: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt(dataset)},
        {"role": "user", "content": format_user(dataset, problem)},
    ]
    # Qwen3 chat template: enable_thinking toggles CoT vs direct (§4.5).
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def flag_direct_leak(text: str, *, enable_thinking: bool, token_count: int) -> bool:
    if enable_thinking:
        return False
    lowered = text.lower()
    if token_count > 128 and any(m in lowered for m in _LEAK_MARKERS):
        return True
    return False


@torch.inference_mode()
def generate_clean(
    loaded: LoadedModel,
    dataset: str,
    problem: str,
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    seed: int,
    capture_logprobs: bool = True,
) -> GenerationResult:
    """Greedy generate without ablation; optionally capture per-token logprobs."""
    torch.manual_seed(seed)
    text = build_chat_text(
        loaded.tokenizer, dataset, problem, enable_thinking=enable_thinking
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    output = loaded.hf_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=capture_logprobs,
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    seq = output.sequences[0]
    gen_ids = seq[prompt_len:].tolist()
    logprobs: list[float] = []
    if capture_logprobs and output.scores:
        for step_scores, token_id in zip(output.scores, gen_ids):
            log_p = torch.log_softmax(step_scores[0], dim=-1)[token_id]
            logprobs.append(float(log_p.item()))

    decoded = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit_cap = len(gen_ids) >= max_new_tokens
    return GenerationResult(
        text=decoded,
        token_ids=gen_ids,
        token_logprobs=logprobs,
        hit_token_cap=hit_cap,
        prompt_token_count=prompt_len,
        direct_leak_flag=flag_direct_leak(
            decoded, enable_thinking=enable_thinking, token_count=len(gen_ids)
        ),
        prompt_input_ids=inputs["input_ids"],
        prompt_attention_mask=inputs.get("attention_mask"),
    )


def _extend_attention_mask(
    attention_mask: torch.Tensor | None,
    token_chunk: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    ones = torch.ones(
        (attention_mask.shape[0], token_chunk.shape[-1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([attention_mask, ones], dim=-1)


def _append_clean_exclusions(
    state: AblationHookState,
    chunk_excl: list[set[int]],
    *,
    past_token_count: int,
) -> None:
    """Merge chunk-local clean top-k into absolute per-position exclusions."""
    if past_token_count == 0:
        state.excluded_by_position = list(chunk_excl)
        return
    if past_token_count != len(state.excluded_by_position):
        raise RuntimeError(
            "clean exclusion cache desynced: "
            f"past={past_token_count} stored={len(state.excluded_by_position)}"
        )
    state.excluded_by_position.extend(chunk_excl)


@torch.inference_mode()
def generate_with_ablation(
    loaded: LoadedModel,
    dataset: str,
    problem: str,
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    seed: int,
    ablation: AblationConfig,
    use_kv_cache: bool = True,
) -> GenerationResult:
    """
    Per-token generation with clean top-10 exclusion then ablated step (§4.6).

    Clean and ablated streams keep separate KV caches: exclusion must see an
    unablated prefix, while ablated decode reuses only ablated past states.
    Set use_kv_cache=False for the quadratic full-prefix path (tests / debug).
    """
    if ablation.kind == "none":
        return generate_clean(
            loaded,
            dataset,
            problem,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            seed=seed,
            capture_logprobs=True,
        )

    torch.manual_seed(seed)
    text = build_chat_text(
        loaded.tokenizer, dataset, problem, enable_thinking=enable_thinking
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(loaded.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(loaded.device)
    prompt_len = int(prompt_ids.shape[-1])
    prompt_input_ids = prompt_ids.clone()
    prompt_attention_mask = (
        attention_mask.clone() if attention_mask is not None else None
    )

    generated: list[int] = []
    logprobs: list[float] = []
    state = AblationHookState(prompt_token_count=prompt_len)
    eos_id = loaded.tokenizer.eos_token_id
    factors = build_ablation_factors(loaded.hf_model, loaded.lens, ablation)

    if use_kv_cache:
        # Separate caches: clean top-10 vs ablated decode must not share past states.
        clean_past: Any | None = None
        ablated_past: Any | None = None
        chunk_ids = prompt_ids

        for _ in range(max_new_tokens):
            clean_past_len = past_token_count_from_cache(clean_past)
            clean_out = loaded.hf_model(
                input_ids=chunk_ids,
                attention_mask=attention_mask,
                past_key_values=clean_past,
                use_cache=True,
            )
            clean_past = clean_out.past_key_values
            chunk_excl = clean_topk_by_position(
                clean_out.logits, ablation.exclude_topk
            )
            _append_clean_exclusions(
                state, chunk_excl, past_token_count=clean_past_len
            )

            state.past_token_count = past_token_count_from_cache(ablated_past)
            with ablation_hooks(
                loaded.hf_model,
                loaded.lens,
                ablation,
                state,
                factors=factors,
            ):
                ablated_out = loaded.hf_model(
                    input_ids=chunk_ids,
                    attention_mask=attention_mask,
                    past_key_values=ablated_past,
                    use_cache=True,
                )
            ablated_past = ablated_out.past_key_values

            logits = ablated_out.logits[0, -1]
            next_id = int(torch.argmax(logits).item())
            log_p = float(torch.log_softmax(logits, dim=-1)[next_id].item())
            generated.append(next_id)
            logprobs.append(log_p)

            chunk_ids = torch.tensor([[next_id]], device=loaded.device)
            attention_mask = _extend_attention_mask(attention_mask, chunk_ids)
            if eos_id is not None and next_id == eos_id:
                break
    else:
        input_ids = prompt_ids
        for _ in range(max_new_tokens):
            clean_out = loaded.hf_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            state.excluded_by_position = clean_topk_by_position(
                clean_out.logits, ablation.exclude_topk
            )
            state.past_token_count = 0
            with ablation_hooks(
                loaded.hf_model,
                loaded.lens,
                ablation,
                state,
                factors=factors,
            ):
                ablated_out = loaded.hf_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            logits = ablated_out.logits[0, -1]
            next_id = int(torch.argmax(logits).item())
            log_p = float(torch.log_softmax(logits, dim=-1)[next_id].item())
            generated.append(next_id)
            logprobs.append(log_p)

            next_token = torch.tensor([[next_id]], device=loaded.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = _extend_attention_mask(attention_mask, next_token)
            if eos_id is not None and next_id == eos_id:
                break

    decoded = loaded.tokenizer.decode(generated, skip_special_tokens=True)
    hit_cap = len(generated) >= max_new_tokens
    return GenerationResult(
        text=decoded,
        token_ids=generated,
        token_logprobs=logprobs,
        hit_token_cap=hit_cap,
        prompt_token_count=prompt_len,
        hook_call_count=state.hook_call_count,
        direct_leak_flag=flag_direct_leak(
            decoded, enable_thinking=enable_thinking, token_count=len(generated)
        ),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        extra={"use_kv_cache": use_kv_cache},
    )


def generate(
    loaded: LoadedModel,
    dataset: str,
    problem: str,
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    seed: int,
    ablation: AblationConfig | None = None,
) -> GenerationResult:
    cfg = ablation or AblationConfig(kind="none")
    return generate_with_ablation(
        loaded,
        dataset,
        problem,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        seed=seed,
        ablation=cfg,
    )
