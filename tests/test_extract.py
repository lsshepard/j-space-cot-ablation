from jspace.extract import (
    answer_ready_for_early_stop,
    answers_equal,
    extract_aime,
    extract_boxed,
    extract_gsm8k,
)


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


def test_aime_fallback_ignores_thinking_integers():
    open_t = "<" + "think" + ">"
    close_t = "</" + "think" + ">"
    # Integers inside thinking must not win over surface boxed.
    text = (
        f"{open_t}lots of scratch 12 34 56{close_t}\n"
        "Final outside.\n"
        "\\boxed{7}\n"
    )
    assert extract_aime(text).answer == "7"

    # Fallback also searches surface only — last in-think int must not win.
    text_fb = (
        f"{open_t}noise 99 and 88{close_t}\n"
        "so the answer is 17\n"
    )
    ex = extract_aime(text_fb)
    assert ex.answer == "17"
    assert not ex.success


def test_answers_equal_gsm8k():
    assert answers_equal("gsm8k", "42", "42")
    assert not answers_equal("gsm8k", None, "42")


def test_answers_equal_multihop_normalized_and_alias():
    assert answers_equal("multihop", "Rome.", "Rome")
    assert answers_equal("multihop", "George Washington", "Washington")
    assert answers_equal("multihop", "Indian Ocean.", "Indian")
    assert not answers_equal("multihop", "Pacific Ocean", "Indian")


def test_early_stop_requires_closed_think_for_cot():
    open_t = "<" + "think" + ">"
    close_t = "</" + "think" + ">"
    # Draft #### inside open think must not trigger stop.
    assert not answer_ready_for_early_stop(
        "gsm8k",
        f"{open_t}\nscratch #### 1\n",
        enable_thinking=True,
    )
    # After think closes, surface #### + newline is enough.
    assert answer_ready_for_early_stop(
        "gsm8k",
        f"{open_t}\nscratch{close_t}\n#### 18\n",
        enable_thinking=True,
    )


def test_early_stop_gsm8k_waits_for_newline_not_mid_number():
    # Generating "#### 18": after first digit must NOT stop.
    assert not answer_ready_for_early_stop(
        "gsm8k", "work\n#### 1", enable_thinking=False
    )
    assert not answer_ready_for_early_stop(
        "gsm8k", "work\n#### 18", enable_thinking=False
    )
    assert answer_ready_for_early_stop(
        "gsm8k", "work\n#### 18\n", enable_thinking=False
    )


def test_early_stop_math500_boxed_after_think():
    open_t = "<" + "think" + ">"
    close_t = "</" + "think" + ">"
    assert not answer_ready_for_early_stop(
        "math500",
        f"{open_t}\n\\boxed{{1}} still thinking",
        enable_thinking=True,
    )
    assert answer_ready_for_early_stop(
        "math500",
        f"{open_t}\nwork{close_t}\n\\boxed{{\\frac{{1}}{{2}}}}\n",
        enable_thinking=True,
    )
