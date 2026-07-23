from jspace.prompts import format_user


def test_format_user_preserves_boxed_braces():
    text = format_user("math500", "Compute $1+1$.")
    assert "\\boxed{}." in text
    assert "Compute $1+1$." in text


def test_format_user_tolerates_braces_in_problem():
    problem = r"Simplify $\frac{a}{b}$ where $\{a,b\}=\{1,2\}$."
    text = format_user("aime", problem)
    assert problem in text
    assert "\\boxed{}" in text
