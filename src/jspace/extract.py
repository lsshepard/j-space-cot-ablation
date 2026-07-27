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
    candidates = [
        int(m) for m in _AIME_INT_RE.findall(surface) if 0 <= int(m) <= 999
    ]
    if candidates:
        return Extraction(str(candidates[-1]), False)
    return Extraction(None, False)


def _normalize_multihop(text: str) -> str:
    """SQuAD-style light normalize: case, punct, articles, whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _multihop_equal(predicted: str, gold: str) -> bool:
    """Normalized EM, or gold token span contained in the prediction (entity aliases)."""
    pred_n = _normalize_multihop(predicted)
    gold_n = _normalize_multihop(gold)
    if not pred_n or not gold_n:
        return False
    if pred_n == gold_n:
        return True
    pred_toks = pred_n.split()
    gold_toks = gold_n.split()
    g = len(gold_toks)
    for i in range(len(pred_toks) - g + 1):
        if pred_toks[i : i + g] == gold_toks:
            return True
    return False


def answer_ready_for_early_stop(
    dataset: str,
    text: str,
    *,
    enable_thinking: bool,
) -> bool:
    """
    True when generation can stop because a scorable final answer is present.

    For thinking modes, require a closed ``</think>`` so we do not freeze on a
    draft marker inside an open think block. Uses the same extractors as scoring
    (last successful ``####`` / ``\\boxed`` on the answer surface).

    GSM8K / AIME ``####`` answers must be followed by a newline so we do not stop
    mid-numeral (e.g. ``#### 1`` while still generating ``18``). ``\\boxed{...}``
    is complete once the closing brace matches.
    """
    if enable_thinking:
        # Local import avoids graded↔extract cycles; pattern matches graded.py.
        open_n = len(re.findall(r"<think(?:ing)?>", text, flags=re.IGNORECASE))
        close_n = len(re.findall(r"</think(?:ing)?>", text, flags=re.IGNORECASE))
        if open_n > close_n:
            return False

    # Do not .strip() — trailing newline is the "answer line complete" signal.
    surface = _THINKING_BLOCK_RE.sub("", text)
    if dataset in {"gsm8k", "aime"}:
        matches = list(_GSM8K_RE.finditer(surface))
        if matches:
            rest = surface[matches[-1].end() :]
            # Require a completed answer line (newline after the number).
            if rest.startswith("\n") or rest.startswith("\r"):
                return True
        if dataset == "gsm8k":
            return False
        # AIME may also finish via \\boxed{...} below.

    if dataset in {"math500", "aime"}:
        boxed = list(_BOXED_RE.finditer(surface))
        if boxed:
            # Closing brace means the marker is complete; prefer last box.
            return True
        return False

    ext = extract_answer(dataset, text)
    return bool(ext.success and ext.answer is not None)


def extract_answer(dataset: str, text: str) -> Extraction:
    if dataset == "gsm8k":
        return extract_gsm8k(text)
    if dataset == "math500":
        return extract_boxed(text)
    if dataset == "aime":
        return extract_aime(text)
    if dataset == "multihop":
        surface = answer_surface_text(text)
        cleaned = surface.splitlines()[-1].strip() if surface else None
        if cleaned is not None:
            cleaned = cleaned.strip().rstrip(".,;:!")
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
        return _multihop_equal(predicted, gold)
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
