"""Answer extractors for GSM8K / MATH-500 / AIME (§4.8)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_GSM8K_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_AIME_INT_RE = re.compile(r"\b([0-9]{1,3})\b")
_LAST_INT_RE = re.compile(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
# Qwen3 thinking blocks precede the scored answer surface.
_THINKING_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class Extraction:
    answer: str | None
    success: bool


def _normalize_number(text: str) -> str:
    return text.replace(",", "").strip()


def answer_surface_text(text: str) -> str:
    """Text used for answer extraction — drops Qwen3 thinking blocks."""
    return _THINKING_BLOCK_RE.sub("", text).strip()


def extract_gsm8k(text: str) -> Extraction:
    text = answer_surface_text(text)
    matches = _GSM8K_RE.findall(text)
    if matches:
        return Extraction(_normalize_number(matches[-1]), True)
    # Fallback: last integer-like token (still mark failure for logging).
    fallback = _LAST_INT_RE.findall(text)
    if fallback:
        return Extraction(_normalize_number(fallback[-1]), False)
    return Extraction(None, False)


def extract_boxed(text: str) -> Extraction:
    text = answer_surface_text(text)
    matches = _BOXED_RE.findall(text)
    if matches:
        return Extraction(matches[-1].strip(), True)
    return Extraction(None, False)


def extract_aime(text: str) -> Extraction:
    surface = answer_surface_text(text)
    boxed = extract_boxed(surface)
    if boxed.success and boxed.answer is not None:
        digits = re.sub(r"[^0-9]", "", boxed.answer)
        if digits.isdigit():
            value = int(digits)
            if 0 <= value <= 999:
                return Extraction(str(value), True)
    # Prefer #### then last 0–999 integer.
    gsm = extract_gsm8k(surface)
    if gsm.answer is not None and gsm.answer.isdigit():
        value = int(gsm.answer)
        if 0 <= value <= 999:
            return Extraction(str(value), gsm.success)
    candidates = [int(m) for m in _AIME_INT_RE.findall(text) if 0 <= int(m) <= 999]
    if candidates:
        return Extraction(str(candidates[-1]), False)
    return Extraction(None, False)


def extract_answer(dataset: str, text: str) -> Extraction:
    if dataset == "gsm8k":
        return extract_gsm8k(text)
    if dataset == "math500":
        return extract_boxed(text)
    if dataset == "aime":
        return extract_aime(text)
    if dataset == "multihop":
        cleaned = text.strip().splitlines()[-1].strip() if text.strip() else None
        return Extraction(cleaned, cleaned is not None and len(cleaned) > 0)
    raise ValueError(f"unknown dataset: {dataset}")


def answers_equal(dataset: str, predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    if dataset in {"gsm8k", "aime"}:
        return _normalize_number(predicted) == _normalize_number(gold)
    if dataset == "math500":
        return math500_equal(predicted, gold)
    if dataset == "multihop":
        return predicted.strip().lower() == gold.strip().lower()
    raise ValueError(f"unknown dataset: {dataset}")


def math500_equal(predicted: str, gold: str) -> bool:
    """Symbolic equivalence via math_verify when available; else normalized string."""
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        pred_parsed = parse(predicted)
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return predicted.strip() == gold.strip()
