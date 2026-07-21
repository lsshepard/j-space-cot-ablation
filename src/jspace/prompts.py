"""Frozen prompt templates — identical across conditions except thinking toggle (§4.9)."""

from __future__ import annotations

# Shared tail: same string in every dataset user prompt so only enable_thinking differs.
_GSM8K_ANSWER_RULE = (
    "The last line of your response must be exactly: #### <integer>"
)
_MATH_ANSWER_RULE = "Put the final answer in \\boxed{}."
_AIME_ANSWER_RULE = (
    "The final answer is an integer from 0 to 999. "
    "Put it in \\boxed{} on the last line."
)

SYSTEM_MATH = (
    "You are a careful math solver. "
    "Follow the user's answer-format instructions exactly. "
    "Do not add text after the final formatted answer."
)

GSM8K_USER = (
    "{problem}\n\n"
    "Solve the problem. If extended reasoning is available, show your work first.\n"
    f"{_GSM8K_ANSWER_RULE}"
)

MATH500_USER = (
    "{problem}\n\n"
    "Write a step-by-step solution. "
    f"{_MATH_ANSWER_RULE}"
)

AIME_USER = (
    "{problem}\n\n"
    "Write a step-by-step solution. "
    f"{_AIME_ANSWER_RULE}"
)

MULTIHOP_USER = "{problem}"


def format_user(dataset: str, problem: str) -> str:
    templates = {
        "gsm8k": GSM8K_USER,
        "math500": MATH500_USER,
        "aime": AIME_USER,
        "multihop": MULTIHOP_USER,
    }
    if dataset not in templates:
        raise ValueError(f"unknown dataset template: {dataset}")
    return templates[dataset].format(problem=problem)


def system_prompt(dataset: str) -> str:
    if dataset == "multihop":
        return (
            "Answer with the final entity or short phrase only. "
            "One line, no explanation."
        )
    return SYSTEM_MATH
