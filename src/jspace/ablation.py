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

    # Per-position clean next-token top-k (paper: exclusion is position-local).
    excluded_by_position: list[set[int]] = field(default_factory=list)
    hook_call_count: int = 0
    last_position: int | None = None
    prompt_token_count: int = 0
    # Tokens already represented in past_key_values before the current chunk.
    past_token_count: int = 0


@dataclass(frozen=True)
class JacobianSvd:
    """Cached SVD factors of a fixed layer Jacobian (S, Vh only)."""

    s: torch.Tensor
    vh: torch.Tensor


@dataclass(frozen=True)
class AblationFactors:
    """Per-run factors shared across decode steps (avoid re-SVD / re-extract)."""

    jacobians: dict[int, torch.Tensor]
    svds: dict[int, JacobianSvd]
    unembed: torch.Tensor


def positions_to_ablate(
    seq_len: int,
    *,
    ablate_prompt_tokens: bool,
    prompt_token_count: int,
) -> range:
    """
    Absolute token positions to project out on a prefix of length seq_len.

    With ablate_prompt_tokens=True (prereg default): every position.
    Otherwise: generation positions only; if the prefix is still prompt-only,
    ablate the last position (next-token prediction site).
    """
    if seq_len <= 0:
        return range(0)
    if ablate_prompt_tokens:
        return range(seq_len)
    start = min(max(prompt_token_count, 0), seq_len)
    if start >= seq_len:
        return range(seq_len - 1, seq_len)
    return range(start, seq_len)


def chunk_positions_to_ablate(
    chunk_len: int,
    *,
    past_token_count: int,
    ablate_prompt_tokens: bool,
    prompt_token_count: int,
) -> list[int]:
    """
    Local indices within the current forward chunk to ablate.

    Used with KV cache: absolute positions outside the chunk were ablated when
    those tokens were first decoded and are already baked into that stream's cache.
    """
    if chunk_len <= 0:
        return []
    total = past_token_count + chunk_len
    absolute = positions_to_ablate(
        total,
        ablate_prompt_tokens=ablate_prompt_tokens,
        prompt_token_count=prompt_token_count,
    )
    return [
        abs_pos - past_token_count
        for abs_pos in absolute
        if past_token_count <= abs_pos < total
    ]


def past_token_count_from_cache(past_key_values: Any | None) -> int:
    """Number of cached tokens in an HF past_key_values / Cache object."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    first = past_key_values[0]
    key = first[0] if isinstance(first, (tuple, list)) else first
    return int(key.shape[-2])


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
    coeffs = torch.einsum("...d,kd->...k", hidden, directions)
    return hidden - torch.einsum("...k,kd->...d", coeffs, directions)


def scale_perturbation_to_norm(
    original: torch.Tensor,
    ablated: torch.Tensor,
    target_delta_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Rescale (original - ablated) so ‖Δh‖ matches target (paper matched-norm).

    target_delta_norm broadcasts against the last dim of hidden.
    """
    delta = original - ablated
    delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    target = target_delta_norm.reshape_as(delta_norm).clamp_min(0.0)
    return original - delta * (target / delta_norm)


def factor_jacobian_svd(jacobian: torch.Tensor) -> JacobianSvd:
    """SVD(J) once; activity scoring only needs S and Vh."""
    _u, s, vh = torch.linalg.svd(jacobian.detach().float(), full_matrices=False)
    return JacobianSvd(s=s, vh=vh)


def active_j_directions_from_svd(
    residual: torch.Tensor,
    svd: JacobianSvd,
    k: int,
) -> torch.Tensor:
    """Top-k residual directions from cached Jacobian SVD factors."""
    h = residual.detach().float()
    vh = svd.vh.to(device=h.device)
    s = svd.s.to(device=h.device)
    activity = (vh @ h).abs() * s
    kk = min(k, activity.numel())
    idx = torch.topk(activity, k=kk).indices
    return gram_schmidt(vh[idx].to(device=residual.device, dtype=residual.dtype))


def active_j_directions(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    k: int,
    *,
    svd: JacobianSvd | None = None,
) -> torch.Tensor:
    """Top-k residual directions by singular activity |(Vh @ h) * S|."""
    factors = svd if svd is not None else factor_jacobian_svd(jacobian)
    return active_j_directions_from_svd(residual, factors, k)


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


def build_ablation_factors(
    hf_model: Any,
    lens: Any,
    config: AblationConfig,
) -> AblationFactors:
    """Extract Jacobians once and cache SVD(J) per band layer for a whole decode."""
    unembed = get_unembed(hf_model)
    d_model = int(unembed.shape[-1])
    jacobians = layer_jacobians(
        lens, config.band_start, config.band_end, d_model=d_model
    )
    svds = {idx: factor_jacobian_svd(mat) for idx, mat in jacobians.items()}
    return AblationFactors(jacobians=jacobians, svds=svds, unembed=unembed)


def _transformer_layers(hf_model: Any) -> list[Any]:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return list(hf_model.model.layers)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return list(hf_model.transformer.h)
    raise AttributeError("unsupported model layout for residual hooks")


def _excluded_at(
    state: AblationHookState,
    pos: int,
) -> set[int]:
    if 0 <= pos < len(state.excluded_by_position):
        return state.excluded_by_position[pos]
    return set()


@contextmanager
def ablation_hooks(
    hf_model: Any,
    lens: Any | None,
    config: AblationConfig,
    state: AblationHookState,
    *,
    factors: AblationFactors | None = None,
) -> Iterator[AblationHookState]:
    """Install forward hooks that project out directions on band layers."""
    if config.kind == "none":
        yield state
        return

    if factors is None:
        if lens is None:
            raise RuntimeError(f"lens required for ablation kind={config.kind}")
        factors = build_ablation_factors(hf_model, lens, config)

    layers = _transformer_layers(hf_model)
    handles: list[Any] = []
    jacobians = factors.jacobians
    svds = factors.svds
    unembed = factors.unembed

    def _j_dirs_for(
        residual: torch.Tensor,
        layer_idx: int,
        excluded: set[int],
    ) -> torch.Tensor:
        J = jacobians[layer_idx].to(device=residual.device, dtype=residual.dtype)
        directions = active_j_directions(
            residual, J, config.k, svd=svds[layer_idx]
        )
        return filter_directions_by_exclusion(
            directions,
            J,
            unembed.to(device=residual.device),
            excluded,
        )

    def _ablate_position(
        h_pos: torch.Tensor,
        layer_idx: int,
        abs_pos: int,
    ) -> torch.Tensor:
        """Ablate one residual vector [batch, d] or [d]; batch dim optional."""
        squeeze = h_pos.ndim == 1
        if squeeze:
            h_pos = h_pos.unsqueeze(0)
        residual = h_pos[0]
        excluded = _excluded_at(state, abs_pos)
        j_dirs = _j_dirs_for(residual, layer_idx, excluded)

        if config.kind == "jspace":
            out = project_out_directions(h_pos, j_dirs)
        else:
            d = residual.shape[-1]
            r_dirs = random_directions(
                d,
                config.k,
                seed=config.seed + layer_idx * 1009 + abs_pos,
                device=residual.device,
                dtype=residual.dtype,
            )
            h_j = project_out_directions(h_pos, j_dirs)
            target_norm = torch.linalg.vector_norm(h_pos - h_j, dim=-1, keepdim=True)
            h_r = project_out_directions(h_pos, r_dirs)
            out = scale_perturbation_to_norm(h_pos, h_r, target_norm)

        return out.squeeze(0) if squeeze else out

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            state.hook_call_count += 1
            if isinstance(output, tuple):
                hidden, *rest = output
                packed = True
            else:
                hidden, rest, packed = output, (), False

            chunk_len = int(hidden.shape[1])
            past = state.past_token_count
            state.last_position = past + chunk_len - 1
            local_positions = chunk_positions_to_ablate(
                chunk_len,
                past_token_count=past,
                ablate_prompt_tokens=config.ablate_prompt_tokens,
                prompt_token_count=state.prompt_token_count,
            )
            if not local_positions:
                return output

            hidden = hidden.clone()
            for local_pos in local_positions:
                abs_pos = past + local_pos
                hidden[:, local_pos, :] = _ablate_position(
                    hidden[:, local_pos, :], layer_idx, abs_pos
                )

            if packed:
                return (hidden, *rest)
            return hidden

        return hook

    for idx in range(config.band_start, config.band_end + 1):
        if idx < 0 or idx >= len(layers):
            continue
        if idx not in jacobians:
            continue
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def clean_topk_by_position(
    logits: torch.Tensor,
    k: int,
) -> list[set[int]]:
    """Per-position clean next-token top-k from logits [batch, seq, vocab]."""
    if logits.ndim != 3:
        raise ValueError(f"expected [batch, seq, vocab] logits, got {tuple(logits.shape)}")
    seq_len = logits.shape[1]
    if k <= 0:
        return [set() for _ in range(seq_len)]
    vocab = logits.shape[-1]
    take = min(k, vocab)
    out: list[set[int]] = []
    for pos in range(seq_len):
        ids = torch.topk(logits[0, pos], k=take).indices.tolist()
        out.append({int(i) for i in ids})
    return out


def clean_topk_token_ids(
    logits: torch.Tensor,
    k: int,
) -> set[int]:
    """Top-k vocabulary ids from last-position next-token logits."""
    if logits.ndim == 3:
        return clean_topk_by_position(logits, k)[-1]
    topk = torch.topk(logits[0, -1], k=min(k, logits.shape[-1])).indices.tolist()
    return set(int(i) for i in topk)


def cosine_loading(residual: torch.Tensor, direction: torch.Tensor) -> float:
    return float(F.cosine_similarity(residual.float(), direction.float(), dim=-1).item())
