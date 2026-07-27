"""Generation with thinking toggle, logprobs, and optional per-token ablation (§4.5–4.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from jspace.ablation import (
    AblationConfig,
    AblationHookState,
    AblationStepDiag,
    ablation_hooks,
    build_ablation_factors,
    clean_topk_by_position,
    clean_topk_ids_matrix,
    past_token_count_from_cache,
)
from jspace.extract import answer_ready_for_early_stop
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
    early_stopped: bool = False
    prompt_input_ids: Any | None = None
    prompt_attention_mask: Any | None = None
    diag_steps: list[AblationStepDiag] = field(default_factory=list)
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


def _make_answer_stopping_criteria(
    loaded: LoadedModel,
    dataset: str,
    *,
    enable_thinking: bool,
    prompt_len: int,
    early_stopped_flag: list[bool],
):
    """HF StoppingCriteria: halt once a scorable final answer is on the page."""
    from transformers import StoppingCriteria

    class _AnswerStop(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
        ) -> bool:
            gen = input_ids[0, prompt_len:]
            if gen.numel() == 0:
                return False
            text = loaded.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
            if answer_ready_for_early_stop(
                dataset, text, enable_thinking=enable_thinking
            ):
                early_stopped_flag[0] = True
                return True
            return False

    return _AnswerStop()


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
    early_stop_on_answer: bool = False,
) -> GenerationResult:
    """Greedy generate without ablation; optionally capture per-token logprobs."""
    torch.manual_seed(seed)
    text = build_chat_text(
        loaded.tokenizer, dataset, problem, enable_thinking=enable_thinking
    )
    inputs = loaded.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=capture_logprobs,
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    early_stopped_flag = [False]
    if early_stop_on_answer:
        from transformers import StoppingCriteriaList

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                _make_answer_stopping_criteria(
                    loaded,
                    dataset,
                    enable_thinking=enable_thinking,
                    prompt_len=prompt_len,
                    early_stopped_flag=early_stopped_flag,
                )
            ]
        )

    output = loaded.hf_model.generate(**inputs, **gen_kwargs)
    seq = output.sequences[0]
    gen_ids = seq[prompt_len:].tolist()
    logprobs: list[float] = []
    if capture_logprobs and output.scores:
        for step_scores, token_id in zip(output.scores, gen_ids):
            log_p = torch.log_softmax(step_scores[0], dim=-1)[token_id]
            logprobs.append(float(log_p.item()))

    decoded = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)
    early_stopped = bool(early_stopped_flag[0])
    hit_cap = (not early_stopped) and len(gen_ids) >= max_new_tokens
    return GenerationResult(
        text=decoded,
        token_ids=gen_ids,
        token_logprobs=logprobs,
        hit_token_cap=hit_cap,
        prompt_token_count=prompt_len,
        early_stopped=early_stopped,
        direct_leak_flag=flag_direct_leak(
            decoded, enable_thinking=enable_thinking, token_count=len(gen_ids)
        ),
        prompt_input_ids=inputs["input_ids"],
        prompt_attention_mask=inputs.get("attention_mask"),
        extra={"early_stop_on_answer": early_stop_on_answer},
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
    chunk_excl: list[set[int]] | torch.Tensor,
    *,
    past_token_count: int,
) -> None:
    """Merge chunk-local clean top-k into absolute per-position exclusions."""
    if isinstance(chunk_excl, torch.Tensor):
        if past_token_count == 0:
            state.excluded_ids = chunk_excl
            state.excluded_by_position = []
            return
        if state.excluded_ids is None or past_token_count != int(
            state.excluded_ids.shape[0]
        ):
            stored = 0 if state.excluded_ids is None else int(state.excluded_ids.shape[0])
            raise RuntimeError(
                "clean exclusion cache desynced: "
                f"past={past_token_count} stored={stored}"
            )
        state.excluded_ids = torch.cat([state.excluded_ids, chunk_excl], dim=0)
        return

    if past_token_count == 0:
        state.excluded_by_position = list(chunk_excl)
        state.excluded_ids = None
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
    early_stop_on_answer: bool = False,
) -> GenerationResult:
    """
    Per-token generation with clean top-10 exclusion then ablated step (§4.6).

    Clean and ablated streams keep separate KV caches: exclusion must see an
    unablated prefix, while ablated decode reuses only ablated past states.
    Set use_kv_cache=False for the quadratic full-prefix path (tests / debug).

    When ``early_stop_on_answer`` is set, stop once a closed-think scorable
    answer marker is present (cuts post-answer J-CoT spin).
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
            early_stop_on_answer=early_stop_on_answer,
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
    early_stopped = False
    state = AblationHookState(
        prompt_token_count=prompt_len,
        collect_diag=ablation.collect_diag,
    )
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
            _append_clean_exclusions(
                state,
                clean_topk_ids_matrix(clean_out.logits, ablation.exclude_topk),
                past_token_count=clean_past_len,
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
            log_probs = torch.log_softmax(logits, dim=-1)
            next_id_t = torch.argmax(log_probs)
            # Single host sync for (token id, logprob).
            next_id_f, log_p = torch.stack(
                [next_id_t.to(torch.float32), log_probs[next_id_t]]
            ).tolist()
            next_id = int(next_id_f)
            generated.append(next_id)
            logprobs.append(float(log_p))

            chunk_ids = next_id_t.view(1, 1)
            attention_mask = _extend_attention_mask(attention_mask, chunk_ids)
            if eos_id is not None and next_id == eos_id:
                break
            if early_stop_on_answer:
                decoded_so_far = loaded.tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                if answer_ready_for_early_stop(
                    dataset, decoded_so_far, enable_thinking=enable_thinking
                ):
                    early_stopped = True
                    break
    else:
        input_ids = prompt_ids
        for _ in range(max_new_tokens):
            clean_out = loaded.hf_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            state.excluded_ids = clean_topk_ids_matrix(
                clean_out.logits, ablation.exclude_topk
            )
            state.excluded_by_position = []
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
            log_probs = torch.log_softmax(logits, dim=-1)
            next_id_t = torch.argmax(log_probs)
            next_id_f, log_p = torch.stack(
                [next_id_t.to(torch.float32), log_probs[next_id_t]]
            ).tolist()
            next_id = int(next_id_f)
            generated.append(next_id)
            logprobs.append(float(log_p))

            next_token = next_id_t.view(1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = _extend_attention_mask(attention_mask, next_token)
            if eos_id is not None and next_id == eos_id:
                break
            if early_stop_on_answer:
                decoded_so_far = loaded.tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                if answer_ready_for_early_stop(
                    dataset, decoded_so_far, enable_thinking=enable_thinking
                ):
                    early_stopped = True
                    break

    decoded = loaded.tokenizer.decode(generated, skip_special_tokens=True)
    hit_cap = (not early_stopped) and len(generated) >= max_new_tokens
    return GenerationResult(
        text=decoded,
        token_ids=generated,
        token_logprobs=logprobs,
        hit_token_cap=hit_cap,
        prompt_token_count=prompt_len,
        hook_call_count=state.hook_call_count,
        early_stopped=early_stopped,
        direct_leak_flag=flag_direct_leak(
            decoded, enable_thinking=enable_thinking, token_count=len(generated)
        ),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        diag_steps=list(state.diag_steps),
        extra={
            "use_kv_cache": use_kv_cache,
            "early_stop_on_answer": early_stop_on_answer,
        },
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
    early_stop_on_answer: bool = False,
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
        early_stop_on_answer=early_stop_on_answer,
    )
