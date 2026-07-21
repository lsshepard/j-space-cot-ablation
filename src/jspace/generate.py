"""Generation with thinking toggle, logprobs, and optional per-token ablation (§4.5–4.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from jspace.ablation import (
    AblationConfig,
    AblationHookState,
    ablation_hooks,
    clean_topk_token_ids,
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
) -> GenerationResult:
    """Greedy generate without ablation; capture per-token logprobs."""
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
        output_scores=True,
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    seq = output.sequences[0]
    gen_ids = seq[prompt_len:].tolist()
    logprobs: list[float] = []
    if output.scores:
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
    )


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
) -> GenerationResult:
    """
    Per-token generation with clean top-10 exclusion then ablated step (§4.6).

    Same code path for direct and CoT arms — only enable_thinking differs.
    """
    if ablation.kind == "none":
        return generate_clean(
            loaded,
            dataset,
            problem,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    torch.manual_seed(seed)
    text = build_chat_text(
        loaded.tokenizer, dataset, problem, enable_thinking=enable_thinking
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(loaded.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(loaded.device)
    prompt_len = int(input_ids.shape[-1])

    generated: list[int] = []
    logprobs: list[float] = []
    state = AblationHookState()
    eos_id = loaded.tokenizer.eos_token_id

    # Optional: ablate prompt positions once before generation.
    if ablation.ablate_prompt_tokens:
        with ablation_hooks(loaded.hf_model, loaded.lens, ablation, state):
            _ = loaded.hf_model(input_ids=input_ids, attention_mask=attention_mask)

    for _ in range(max_new_tokens):
        # Clean pass on current prefix → top-10 exclusion set.
        clean_out = loaded.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        state.excluded_token_ids = clean_topk_token_ids(
            clean_out.logits, ablation.exclude_topk
        )

        # Ablated pass produces the next token.
        with ablation_hooks(loaded.hf_model, loaded.lens, ablation, state):
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
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )
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
