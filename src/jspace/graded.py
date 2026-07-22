"""Teacher-forced gold-answer logprob for floored-cell graded metric (§4.2 / §4.7)."""

from __future__ import annotations

import re

import torch

from jspace.ablation import AblationConfig, AblationHookState, ablation_hooks
from jspace.load import LoadedModel

_OPEN_THINK_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def format_gold_answer_surface(dataset: str, gold: str) -> str:
    """Canonical answer surface string for teacher-forcing (matches extract formats)."""
    gold = gold.strip()
    if dataset == "gsm8k":
        return f"#### {gold}"
    if dataset in {"math500", "aime"}:
        return f"\\boxed{{{gold}}}"
    if dataset == "multihop":
        return gold
    raise ValueError(f"unknown dataset: {dataset}")


def thinking_block_unclosed(text: str) -> bool:
    """True if a Qwen-style think block was opened but not closed."""
    return len(_OPEN_THINK_RE.findall(text)) > len(_CLOSE_THINK_RE.findall(text))


def close_think_suffix(text: str) -> str:
    """Closing tag matching the opened think variant, if any."""
    opens = _OPEN_THINK_RE.findall(text)
    if not opens:
        return "</think>"
    opened = opens[-1].lower()
    if "thinking" in opened:
        return "</thinking>"
    return "</think>"


def build_graded_prefix_ids(
    loaded: LoadedModel,
    prompt_input_ids: torch.Tensor,
    *,
    enable_thinking: bool,
    generated_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """
    Prefix before teacher-forcing the gold answer.

    Direct: chat prompt only.
    CoT: prompt + generated tokens; if think block is still open, append a close tag
    so gold is forced as answer-surface continuation, not mid-thought.
    """
    if not enable_thinking or not generated_token_ids:
        return prompt_input_ids

    device = prompt_input_ids.device
    gen = torch.tensor([generated_token_ids], device=device, dtype=prompt_input_ids.dtype)
    prefix = torch.cat([prompt_input_ids, gen], dim=-1)
    decoded = loaded.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    if thinking_block_unclosed(decoded):
        close_ids = loaded.tokenizer.encode(
            "\n" + close_think_suffix(decoded) + "\n",
            add_special_tokens=False,
        )
        if close_ids:
            close = torch.tensor([close_ids], device=device, dtype=prompt_input_ids.dtype)
            prefix = torch.cat([prefix, close], dim=-1)
    return prefix


@torch.inference_mode()
def teacher_forced_gold_logprob(
    loaded: LoadedModel,
    prompt_input_ids: torch.Tensor,
    dataset: str,
    gold_answer: str,
    *,
    enable_thinking: bool = False,
    generated_token_ids: list[int] | None = None,
    ablation: AblationConfig | None = None,
    attention_mask: torch.Tensor | None = None,
) -> float | None:
    """
    Mean logprob of gold answer tokens teacher-forced after the graded prefix.

    Direct: after the chat prompt.
    CoT: after generated reasoning (plus close-think if needed), so the score is
    closer to P(gold | thinking) rather than “answer instead of think.”
    """
    surface = format_gold_answer_surface(dataset, gold_answer)
    gold_ids = loaded.tokenizer(
        surface, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].tolist()
    if not gold_ids:
        return None

    input_ids = build_graded_prefix_ids(
        loaded,
        prompt_input_ids,
        enable_thinking=enable_thinking,
        generated_token_ids=generated_token_ids,
    )
    if attention_mask is not None and input_ids.shape[-1] != attention_mask.shape[-1]:
        # Prefix grew (CoT tokens / close-think); rebuild a matching mask of ones.
        mask = torch.ones_like(input_ids)
    else:
        mask = attention_mask

    logprobs: list[float] = []
    cfg = ablation or AblationConfig(kind="none")
    prompt_len = int(prompt_input_ids.shape[-1])
    state = AblationHookState(prompt_token_count=prompt_len)

    for gold_id in gold_ids:
        if cfg.kind == "none":
            out = loaded.hf_model(input_ids=input_ids, attention_mask=mask)
        else:
            with ablation_hooks(loaded.hf_model, loaded.lens, cfg, state):
                out = loaded.hf_model(input_ids=input_ids, attention_mask=mask)
        logits = out.logits[0, -1]
        logprobs.append(float(torch.log_softmax(logits, dim=-1)[gold_id].item()))
        next_token = torch.tensor([[gold_id]], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=-1)

    return float(sum(logprobs) / len(logprobs))
