"""J-space and random-direction residual ablation with top-10 exclusion (§4.6)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import torch
import torch.nn.functional as F

AblationKind = Literal["none", "jspace", "random"]
_PROXY_WARNED = False


@dataclass
class AblationConfig:
    kind: AblationKind = "none"
    band_start: int = 0
    band_end: int = 0  # inclusive
    k: int = 10
    seed: int = 0
    exclude_topk: int = 10
    ablate_prompt_tokens: bool = True


@dataclass
class AblationHookState:
    """Mutable state shared between clean top-10 capture and ablated step."""

    excluded_token_ids: set[int] = field(default_factory=set)
    hook_call_count: int = 0
    last_position: int | None = None


def gram_schmidt(vectors: torch.Tensor) -> torch.Tensor:
    """Orthonormalize rows of [n, d]; drop near-zero rows."""
    basis: list[torch.Tensor] = []
    for vec in vectors:
        v = vec.clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = torch.linalg.vector_norm(v)
        if float(norm) < 1e-8:
            continue
        basis.append(v / norm)
    if not basis:
        return vectors.new_zeros((0, vectors.shape[-1]))
    return torch.stack(basis, dim=0)


def project_out_directions(hidden: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Project out rows of `directions` from last-dim of hidden (§4.6)."""
    if directions.numel() == 0:
        return hidden
    # hidden: [..., d], directions: [k, d]
    coeffs = torch.einsum("...d,kd->...k", hidden, directions)
    return hidden - torch.einsum("...k,kd->...d", coeffs, directions)


def active_j_directions(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Top-k residual directions by singular activity |(Vh @ h) * S|."""
    # SVD on CPU float32 — MPS lacks linalg_svd; keeps local/GPU paths uniform.
    j32 = jacobian.detach().float().cpu()
    h32 = residual.detach().float().cpu()
    _u, s, vh = torch.linalg.svd(j32, full_matrices=False)
    activity = (vh @ h32).abs() * s
    kk = min(k, activity.numel())
    idx = torch.topk(activity, k=kk).indices
    return gram_schmidt(vh[idx].to(device=residual.device, dtype=residual.dtype))


def direction_top_token(
    direction: torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
) -> int:
    """Argmax vocab token for transporting a unit residual direction through J."""
    transported = jacobian.float() @ direction.float()
    logits = unembed.float() @ transported
    return int(torch.argmax(logits).item())


def filter_directions_by_exclusion(
    directions: torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    excluded_token_ids: set[int],
) -> torch.Tensor:
    """Drop directions whose J-decoded top token is in the clean top-10 (§4.6)."""
    if directions.numel() == 0 or not excluded_token_ids:
        return directions
    kept: list[torch.Tensor] = []
    for direction in directions:
        top_tok = direction_top_token(direction, jacobian, unembed)
        if top_tok not in excluded_token_ids:
            kept.append(direction)
    if not kept:
        return directions.new_zeros((0, directions.shape[-1]))
    return torch.stack(kept, dim=0)


def random_directions(
    d_model: int,
    k: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    raw = torch.randn(k, d_model, generator=rng)
    return gram_schmidt(raw.to(device=device, dtype=dtype))


def layer_jacobians(
    lens: Any,
    band_start: int,
    band_end: int,
    *,
    d_model: int | None = None,
) -> dict[int, torch.Tensor]:
    """Extract per-layer J matrices for layers in [band_start, band_end].

    If the pre-fitted lens d_model disagrees with the loaded model (common when
    plumbing on Qwen3-0.6B against the 4B lens), substitute identity matrices
    so the hook path can be smoke-tested. Authoritative runs require matching
    `Qwen/Qwen3-4B` + the Neuronpedia 4B lens.
    """
    global _PROXY_WARNED
    matrices: dict[int, torch.Tensor] = {}
    store = getattr(lens, "jacobians", None) or getattr(lens, "Js", None) or getattr(lens, "J", None)
    if store is None and hasattr(lens, "layers"):
        store = lens.layers
    if isinstance(store, dict):
        items = store.items()
    elif isinstance(store, (list, tuple)):
        items = enumerate(store)
    else:
        raise AttributeError("cannot locate Jacobian matrices on lens object")

    for layer_idx, mat in items:
        idx = int(layer_idx)
        if band_start <= idx <= band_end and mat is not None:
            matrices[idx] = mat if isinstance(mat, torch.Tensor) else torch.as_tensor(mat)
    if not matrices:
        raise ValueError(f"no Jacobians in band [{band_start}, {band_end}]")

    if d_model is not None:
        sample = next(iter(matrices.values()))
        if int(sample.shape[-1]) != int(d_model):
            if not _PROXY_WARNED:
                warnings.warn(
                    f"lens d_model={sample.shape[-1]} != model d_model={d_model}; "
                    "using identity Jacobian proxy for local plumbing only. "
                    "Use Qwen/Qwen3-4B for authoritative J-ablation.",
                    stacklevel=2,
                )
                _PROXY_WARNED = True
            matrices = {
                idx: torch.eye(d_model, dtype=torch.float32) for idx in matrices
            }
    return matrices


def get_unembed(hf_model: Any) -> torch.Tensor:
    if hasattr(hf_model, "lm_head"):
        return hf_model.lm_head.weight
    if hasattr(hf_model, "get_output_embeddings"):
        emb = hf_model.get_output_embeddings()
        if emb is not None:
            return emb.weight
    raise AttributeError("could not locate unembedding matrix")


def _transformer_layers(hf_model: Any) -> list[Any]:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return list(hf_model.model.layers)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return list(hf_model.transformer.h)
    raise AttributeError("unsupported model layout for residual hooks")


@contextmanager
def ablation_hooks(
    hf_model: Any,
    lens: Any | None,
    config: AblationConfig,
    state: AblationHookState,
) -> Iterator[AblationHookState]:
    """Install forward hooks that project out directions on band layers."""
    if config.kind == "none":
        yield state
        return

    layers = _transformer_layers(hf_model)
    handles: list[Any] = []
    unembed = get_unembed(hf_model)
    d_model = int(get_unembed(hf_model).shape[-1])
    jacobians = (
        layer_jacobians(lens, config.band_start, config.band_end, d_model=d_model)
        if config.kind == "jspace"
        else {}
    )

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            state.hook_call_count += 1
            if isinstance(output, tuple):
                hidden, *rest = output
                packed = True
            else:
                hidden, rest, packed = output, (), False

            # hidden: [batch, seq, d]
            pos = hidden.shape[1] - 1
            state.last_position = pos
            if not config.ablate_prompt_tokens and pos == 0:
                return output

            h_last = hidden[:, -1, :]
            d_model = h_last.shape[-1]
            if config.kind == "random":
                directions = random_directions(
                    d_model,
                    config.k,
                    seed=config.seed + layer_idx * 1009 + pos,
                    device=h_last.device,
                    dtype=h_last.dtype,
                )
            else:
                J = jacobians[layer_idx].to(device=h_last.device, dtype=h_last.dtype)
                directions = active_j_directions(h_last[0], J, config.k)
                directions = filter_directions_by_exclusion(
                    directions,
                    J,
                    unembed.to(device=h_last.device),
                    state.excluded_token_ids,
                )

            # Matched-norm project-out on the last position (generation focus).
            # Also ablate all positions when ablate_prompt_tokens is set (prompt pass).
            if config.ablate_prompt_tokens and hidden.shape[1] > 1:
                # Apply to full sequence for prompt+generation consistency on first pass.
                flat = hidden.reshape(-1, d_model)
                # Use last-token directions as approximate active set for the step.
                flat = project_out_directions(flat, directions)
                hidden = flat.reshape(hidden.shape)
            else:
                hidden = hidden.clone()
                hidden[:, -1, :] = project_out_directions(h_last, directions)

            if packed:
                return (hidden, *rest)
            return hidden

        return hook

    for idx in range(config.band_start, config.band_end + 1):
        if idx < 0 or idx >= len(layers):
            continue
        if config.kind == "jspace" and idx not in jacobians:
            continue
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def clean_topk_token_ids(
    logits: torch.Tensor,
    k: int,
) -> set[int]:
    """Top-k vocabulary ids from next-token logits (§4.6 exclusion)."""
    topk = torch.topk(logits[0, -1], k=min(k, logits.shape[-1])).indices.tolist()
    return set(int(i) for i in topk)


def matched_norm_scale(original: torch.Tensor, ablated: torch.Tensor) -> torch.Tensor:
    """Optional rescale to preserve residual norm after projection."""
    orig_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True).clamp_min(1e-8)
    abl_norm = torch.linalg.vector_norm(ablated, dim=-1, keepdim=True).clamp_min(1e-8)
    return ablated * (orig_norm / abl_norm)


def cosine_loading(residual: torch.Tensor, direction: torch.Tensor) -> float:
    return float(F.cosine_similarity(residual.float(), direction.float(), dim=-1).item())
