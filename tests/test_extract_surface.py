from jspace.extract import answer_surface_text, extract_gsm8k


def test_thinking_block_removed_before_extract():
    text = "#### 42\n"
    surface = answer_surface_text(text)
    assert "#### 42" in surface
    assert extract_gsm8k(text).answer == "42"


def test_gsm8k_prefers_surface_answer():
    text = "guess 5\nWork.\n#### 18\n"
    ex = extract_gsm8k(text)
    assert ex.success and ex.answer == "18"
