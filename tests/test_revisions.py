from pathlib import Path

from jspace.revisions import resolve_dataset_revisions, resolve_repo_revision


def test_resolve_keeps_explicit_revision():
    assert resolve_repo_revision("openai/gsm8k", "abc123") == "abc123"


def test_resolve_dataset_revisions_respects_pins():
    meta = resolve_dataset_revisions(
        gsm8k_repo="openai/gsm8k",
        math500_repo="HuggingFaceH4/MATH-500",
        aime_repo="HuggingFaceH4/aime_2024",
        gsm8k_revision="sha-gsm",
        math500_revision="sha-math",
        aime_revision="sha-aime",
    )
    assert meta["gsm8k_revision"] == "sha-gsm"
    assert meta["math500_revision"] == "sha-math"
    assert meta["aime_revision"] == "sha-aime"
