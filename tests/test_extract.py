from jspace.extract import answers_equal, extract_aime, extract_boxed, extract_gsm8k


def test_gsm8k_hash_line():
    text = "Something\n#### 42\n"
    ex = extract_gsm8k(text)
    assert ex.success and ex.answer == "42"


def test_gsm8k_comma():
    ex = extract_gsm8k("#### 1,234")
    assert ex.success and ex.answer == "1234"


def test_boxed():
    ex = extract_boxed(r"final is \boxed{\frac{1}{2}}")
    assert ex.success and ex.answer == r"\frac{1}{2}"


def test_aime_boxed():
    ex = extract_aime(r"answer \boxed{007}")
    assert ex.success and ex.answer == "7"


def test_aime_range():
    ex = extract_aime("I think 1000 is wrong and 42 is right")
    assert ex.answer == "42"
    assert not ex.success  # no boxed / ####


def test_answers_equal_gsm8k():
    assert answers_equal("gsm8k", "42", "42")
    assert not answers_equal("gsm8k", None, "42")
