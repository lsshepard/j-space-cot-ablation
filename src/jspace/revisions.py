"""Resolve and record Hugging Face repo revisions for reproducibility (§4.9)."""

from __future__ import annotations

from typing import Any


def resolve_repo_revision(repo_id: str, revision: str | None = None) -> str | None:
    """
    Return a concrete revision string for logging / pinning.

    If `revision` is already set, return it. Otherwise resolve the Hub default
    branch tip SHA. Returns None when Hub is unreachable (offline / tests).
    """
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id=repo_id, revision=None)
        sha = getattr(info, "sha", None)
        return str(sha) if sha else None
    except Exception:
        return None


def resolve_dataset_revisions(
    *,
    gsm8k_repo: str,
    math500_repo: str,
    aime_repo: str,
    gsm8k_revision: str | None = None,
    math500_revision: str | None = None,
    aime_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "gsm8k_repo": gsm8k_repo,
        "gsm8k_revision": resolve_repo_revision(gsm8k_repo, gsm8k_revision),
        "math500_repo": math500_repo,
        "math500_revision": resolve_repo_revision(math500_repo, math500_revision),
        "aime_repo": aime_repo,
        "aime_revision": resolve_repo_revision(aime_repo, aime_revision),
    }
