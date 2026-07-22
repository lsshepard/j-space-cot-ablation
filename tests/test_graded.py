from jspace.graded import (
    close_think_suffix,
    format_gold_answer_surface,
    thinking_block_unclosed,
)

_OPEN = "<" + "think" + ">"
_CLOSE = "</" + "think" + ">"
_OPEN_ING = "<" + "thinking" + ">"
_CLOSE_ING = "</" + "thinking" + ">"


def test_format_gold_surfaces():
    assert format_gold_answer_surface("gsm8k", "42") == "#### 42"
    assert format_gold_answer_surface("math500", "1/2") == "\\boxed{1/2}"
    assert format_gold_answer_surface("aime", "7") == "\\boxed{7}"
    assert format_gold_answer_surface("multihop", "Rome") == "Rome"


def test_thinking_block_unclosed():
    assert thinking_block_unclosed(f"{_OPEN}step") is True
    assert thinking_block_unclosed(f"{_OPEN}done{_CLOSE}") is False
    assert thinking_block_unclosed("no tags") is False


def test_close_think_suffix_matches_open_tag():
    assert close_think_suffix(_OPEN) == _CLOSE
    assert close_think_suffix(_OPEN_ING) == _CLOSE_ING
