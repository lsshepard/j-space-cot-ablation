"""Model / tokenizer / Jacobian lens loading (§5.1, §5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from jspace.config import Settings


@dataclass
class LoadedModel:
    hf_model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    model_name: str
    model_revision: str | None
    n_layers: int
    jlens_model: Any | None = None
    lens: Any | None = None
    lens_meta: dict[str, Any] | None = None


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, requested: str | None = None) -> torch.dtype:
    if requested:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        key = requested.lower()
        if key not in mapping:
            raise ValueError(f"unsupported dtype: {requested}")
        return mapping[key]
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_hf_model(settings: Settings) -> LoadedModel:
    """Load HF causal LM + tokenizer with eager attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(settings.device)
    dtype = resolve_dtype(device, settings.dtype)
    tok_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": settings.attn_implementation,
    }
    if settings.model_revision:
        tok_kwargs["revision"] = settings.model_revision
        model_kwargs["revision"] = settings.model_revision

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(settings.model_name, **model_kwargs)
    hf_model.to(device)
    hf_model.eval()

    n_layers = int(hf_model.config.num_hidden_layers)
    from jspace.revisions import resolve_repo_revision

    resolved_revision = resolve_repo_revision(
        settings.model_name, settings.model_revision
    )
    return LoadedModel(
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_name=settings.model_name,
        model_revision=resolved_revision,
        n_layers=n_layers,
    )


def attach_jlens(loaded: LoadedModel, settings: Settings) -> LoadedModel:
    """Wrap model with jlens and load the pre-fitted Jacobian lens."""
    import jlens

    jlens_model = jlens.from_hf(loaded.hf_model, loaded.tokenizer)
    lens_kwargs: dict[str, Any] = {"filename": settings.lens_filename}
    if settings.lens_revision:
        lens_kwargs["revision"] = settings.lens_revision

    lens = jlens.JacobianLens.from_pretrained(settings.lens_repo, **lens_kwargs)
    from jspace.revisions import resolve_repo_revision

    resolved_lens = resolve_repo_revision(settings.lens_repo, settings.lens_revision)
    meta = {
        "lens_repo": settings.lens_repo,
        "lens_filename": settings.lens_filename,
        "lens_revision": resolved_lens or settings.lens_revision or "default",
        "source_layers": list(getattr(lens, "source_layers", [])),
        "d_model": getattr(lens, "d_model", None),
    }
    loaded.jlens_model = jlens_model
    loaded.lens = lens
    loaded.lens_meta = meta
    return loaded


def load_model_and_lens(settings: Settings) -> LoadedModel:
    loaded = load_hf_model(settings)
    return attach_jlens(loaded, settings)
