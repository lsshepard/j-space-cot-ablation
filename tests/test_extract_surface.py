from jspace.extract import answer_surface_text, extract_gsm8k

_OPEN = "<" + "think" + ">"
_CLOSE = "</" + "think" + ">"


def test_thinking_block_removed_before_extract():
    text = f"{_OPEN}scratch 5{_CLOSE}\n#### 42\n"
    surface = answer_surface_text(text)
    assert "scratch" not in surface
    assert "#### 42" in surface
    assert extract_gsm8k(text).answer == "42"


def test_gsm8k_prefers_surface_answer():
    text = f"{_OPEN}guess 5{_CLOSE}\nWork.\n#### 18\n"
    ex = extract_gsm8k(text)
    assert ex.success and ex.answer == "18"
