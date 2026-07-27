"""J-space and random-direction residual ablation with top-10 exclusion (§4.6).

Paper ablation (Anthropic workspace §3.5.2): at each position in a layer band,
select the k most activated *J-lens token vectors* v_t = J^T u_t by lens logit
on h, skip tokens in the clean next-token top-10, and project those directions
out of the residual. Random control matches band/k and ‖Δh‖.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence

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
    # When True, hooks append AblationStepDiag into state.diag_steps (§instrument probe).
    collect_diag: bool = False


@dataclass(frozen=True)
class DirectionExclusionInfo:
    """One active J-direction after selection, with clean-top-k exclusion label."""

    rank: int
    top_token_id: int
    excluded: bool
    # |⟨h, d⟩| on the residual being edited (post gram-schmidt direction).
    coeff_abs: float = 0.0


@dataclass(frozen=True)
class AblationStepDiag:
    """One band-layer edit: direction survival after exclusion + ‖Δh‖."""

    layer_idx: int
    abs_pos: int
    n_active: int
    n_survivors: int
    delta_h_norm: float
    # Populated when collect_diag=True (same directions the hook filters/projects).
    directions: tuple[DirectionExclusionInfo, ...] = ()


@dataclass
class AblationHookState:
    """Mutable state shared between clean top-10 capture and ablated step."""

    # Per-position clean next-token top-k (paper: exclusion is position-local).
    # Prefer ``excluded_ids`` ([n_pos, k] on device) in the decode hot path to
    # avoid per-step host syncs; ``excluded_by_position`` remains for tests /
    # diagnostics that materialize Python sets.
    excluded_by_position: list[set[int]] = field(default_factory=list)
    excluded_ids: torch.Tensor | None = None
    hook_call_count: int = 0
    last_position: int | None = None
    prompt_token_count: int = 0
    # Tokens already represented in past_key_values before the current chunk.
    past_token_count: int = 0
    collect_diag: bool = False
    diag_steps: list[AblationStepDiag] = field(default_factory=list)


@dataclass(frozen=True)
class AblationFactors:
    """Per-run factors shared across decode steps.

    Jacobians live on the model device. ``final_norm`` matches jlens/HF unembed
    (RMSNorm then lm_head).

    ``unembed_f32`` is the float32 cast of ``unembed``, materialized once here:
    the lens path needs float32 and would otherwise rebuild a ~1.5 GB copy on
    every (band layer, position) it touches.
    """

    jacobians: dict[int, torch.Tensor]
    unembed: torch.Tensor
    final_norm: Any | None = None
    unembed_f32: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.unembed_f32 is None:
            object.__setattr__(
                self, "unembed_f32", self.unembed.detach().to(torch.float32)
            )


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


def _norm_input_dtype(final_norm: Any, fallback: torch.dtype) -> torch.dtype:
    """Dtype ``final_norm`` runs in (its own weight), not the lens matmul dtype.

    jlens casts the residual to the model dtype before the final RMSNorm, so
    this must track the norm module rather than whatever precision the caller
    happens to hold the unembed in.
    """
    weight = getattr(final_norm, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    return fallback


def lens_logits_for_residual(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    *,
    final_norm: Any | None = None,
) -> torch.Tensor:
    """
    J-lens scores over vocab: unembed(norm(J @ h)), matching jlens HF unembed.

    Pass a float32 ``unembed`` (see ``AblationFactors.unembed_f32``) to avoid
    re-casting the full unembedding matrix on every call.

    Returns a 1-D float tensor [vocab].
    """
    h = residual.detach()
    transported = jacobian.float() @ h.float()
    if final_norm is not None:
        dtype = _norm_input_dtype(final_norm, unembed.dtype)
        transported = final_norm(transported.to(dtype=dtype)).float()
    return unembed.float() @ transported


def j_lens_vectors_for_tokens(
    token_ids: Sequence[int] | torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
) -> torch.Tensor:
    """
    Unit J-lens vectors v_t = J^T u_t for each token (rows of W_U J).

    Pass a float32 ``unembed`` (see ``AblationFactors.unembed_f32``) to avoid
    re-casting the full unembedding matrix on every call.

    Returns [n, d_model] float32 on jacobian's device.
    """
    J = jacobian.float()
    if isinstance(token_ids, torch.Tensor):
        if token_ids.numel() == 0:
            return J.new_zeros((0, J.shape[-1]))
        rows = unembed.float()[token_ids.long().reshape(-1)]
    else:
        if not token_ids:
            return J.new_zeros((0, J.shape[-1]))
        rows = unembed.float()[list(token_ids)]  # [n, d_out]
    dirs = (J.T @ rows.T).T  # [n, d_model]
    norms = torch.linalg.vector_norm(dirs, dim=-1, keepdim=True).clamp_min(1e-8)
    return dirs / norms


def _exclusion_mask(
    top_ids: torch.Tensor,
    excluded_token_ids: set[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Boolean mask over ``top_ids`` — True means skip (clean top-k hit)."""
    if excluded_token_ids is None:
        return torch.zeros(top_ids.shape[0], dtype=torch.bool, device=top_ids.device)
    if isinstance(excluded_token_ids, torch.Tensor):
        excl = excluded_token_ids.reshape(-1).to(
            device=top_ids.device, dtype=top_ids.dtype
        )
        if excl.numel() == 0:
            return torch.zeros(
                top_ids.shape[0], dtype=torch.bool, device=top_ids.device
            )
        return torch.isin(top_ids, excl)
    if not excluded_token_ids:
        return torch.zeros(top_ids.shape[0], dtype=torch.bool, device=top_ids.device)
    excl = torch.tensor(
        list(excluded_token_ids), device=top_ids.device, dtype=top_ids.dtype
    )
    return torch.isin(top_ids, excl)


def select_active_j_lens_directions(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    k: int,
    excluded_token_ids: set[int] | torch.Tensor,
    *,
    final_norm: Any | None = None,
    candidate_pool: int | None = None,
    collect_infos: bool = True,
) -> tuple[torch.Tensor, list[DirectionExclusionInfo]]:
    """
    Paper §3.5.2: top-k most activated J-lens token directions, skipping clean top-10.

    Activation = lens logit for token t on h. Directions are v_t (not SVD axes).
    Returns (orthonormal survivor directions to project out, per-token infos).

    Hot path keeps selection on-device (no ``.tolist()``) unless ``collect_infos``.
    """
    if k <= 0:
        return residual.new_zeros((0, residual.shape[-1])), []

    scores = lens_logits_for_residual(
        residual, jacobian, unembed, final_norm=final_norm
    )
    vocab = int(scores.numel())
    if isinstance(excluded_token_ids, torch.Tensor):
        n_excl = int(excluded_token_ids.numel())
    else:
        n_excl = len(excluded_token_ids)
    # Scan far enough down the lens ranking to fill k after exclusions.
    pool = candidate_pool if candidate_pool is not None else min(
        vocab, max(k + n_excl + 32, k * 8)
    )
    pool = min(vocab, max(pool, k))
    top_scores, top_ids = torch.topk(scores, k=pool)
    excl_mask = _exclusion_mask(top_ids, excluded_token_ids)
    kept_ids = top_ids[~excl_mask][:k]

    infos: list[DirectionExclusionInfo] = []
    if collect_infos:
        # Diagnostics only — sync once here, never on the decode hot path.
        n_keep = 0
        for rank, (score, tid_t, excluded) in enumerate(
            zip(top_scores.tolist(), top_ids.tolist(), excl_mask.tolist())
        ):
            infos.append(
                DirectionExclusionInfo(
                    rank=rank,
                    top_token_id=int(tid_t),
                    excluded=bool(excluded),
                    coeff_abs=float(score),
                )
            )
            if not excluded:
                n_keep += 1
            if n_keep >= k:
                break

    if kept_ids.numel() == 0:
        return residual.new_zeros((0, residual.shape[-1])), infos

    # Keep float32 through orthonormalization — bf16 GS is discontinuous for top-k.
    dirs = j_lens_vectors_for_tokens(kept_ids, jacobian, unembed)
    dirs = gram_schmidt(dirs.to(device=residual.device, dtype=torch.float32))
    return dirs, infos


def active_j_directions(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    k: int,
    *,
    unembed: torch.Tensor,
    excluded_token_ids: set[int] | None = None,
    final_norm: Any | None = None,
) -> torch.Tensor:
    """Top-k activated J-lens token directions (paper §3.5.2)."""
    dirs, _infos = select_active_j_lens_directions(
        residual,
        jacobian,
        unembed,
        k,
        excluded_token_ids or set(),
        final_norm=final_norm,
    )
    return dirs


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


def get_final_norm(hf_model: Any) -> Any | None:
    """Final pre-unembed norm (matches jlens HFLensModel.unembed)."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "norm"):
        return hf_model.model.norm
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "ln_f"):
        return hf_model.transformer.ln_f
    return None


def build_ablation_factors(
    hf_model: Any,
    lens: Any,
    config: AblationConfig,
) -> AblationFactors:
    """Extract band Jacobians onto the model device (no SVD required)."""
    unembed = get_unembed(hf_model)
    device = unembed.device
    d_model = int(unembed.shape[-1])
    raw = layer_jacobians(
        lens, config.band_start, config.band_end, d_model=d_model
    )
    jacobians = {
        idx: mat.detach().to(device=device, dtype=torch.float32, non_blocking=True)
        for idx, mat in raw.items()
    }
    return AblationFactors(
        jacobians=jacobians,
        unembed=unembed,
        final_norm=get_final_norm(hf_model),
    )


def _transformer_layers(hf_model: Any) -> list[Any]:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return list(hf_model.model.layers)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return list(hf_model.transformer.h)
    raise AttributeError("unsupported model layout for residual hooks")


def _excluded_at(
    state: AblationHookState,
    pos: int,
) -> set[int] | torch.Tensor:
    if state.excluded_ids is not None:
        if 0 <= pos < int(state.excluded_ids.shape[0]):
            return state.excluded_ids[pos]
        return state.excluded_ids.new_empty((0,), dtype=torch.long)
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

    if config.collect_diag:
        state.collect_diag = True

    layers = _transformer_layers(hf_model)
    handles: list[Any] = []
    jacobians = factors.jacobians
    unembed = factors.unembed_f32
    final_norm = factors.final_norm

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
        J = jacobians[layer_idx].to(device=residual.device, dtype=torch.float32)
        U = unembed.to(device=residual.device)
        # Paper: top-k activated J-lens token vectors, skipping clean top-10.
        j_dirs, dir_infos = select_active_j_lens_directions(
            residual,
            J,
            U,
            config.k,
            excluded,
            final_norm=final_norm,
            collect_infos=state.collect_diag,
        )

        # Edit in float32 for stable top-k / projection, then cast back.
        h_f = h_pos.float()
        if config.kind == "jspace":
            out_f = project_out_directions(h_f, j_dirs)
        else:
            d = residual.shape[-1]
            r_dirs = random_directions(
                d,
                config.k,
                seed=config.seed + layer_idx * 1009 + abs_pos,
                device=residual.device,
                dtype=torch.float32,
            )
            h_j = project_out_directions(h_f, j_dirs)
            target_norm = torch.linalg.vector_norm(h_f - h_j, dim=-1, keepdim=True)
            h_r = project_out_directions(h_f, r_dirs)
            out_f = scale_perturbation_to_norm(h_f, h_r, target_norm)
        out = out_f.to(dtype=h_pos.dtype)

        if state.collect_diag:
            delta = float(torch.linalg.vector_norm((h_f - out_f).detach()).item())
            state.diag_steps.append(
                AblationStepDiag(
                    layer_idx=layer_idx,
                    abs_pos=abs_pos,
                    n_active=int(len(dir_infos)),
                    n_survivors=int(j_dirs.shape[0]),
                    delta_h_norm=delta,
                    directions=tuple(dir_infos),
                )
            )

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


def clean_topk_ids_matrix(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-position clean next-token top-k ids as ``[seq, take]`` on-device."""
    if logits.ndim != 3:
        raise ValueError(f"expected [batch, seq, vocab] logits, got {tuple(logits.shape)}")
    seq_len = logits.shape[1]
    if k <= 0:
        return logits.new_empty((seq_len, 0), dtype=torch.long)
    take = min(k, logits.shape[-1])
    return torch.topk(logits[0], k=take, dim=-1).indices


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
    ids = clean_topk_ids_matrix(logits, k)
    return [{int(i) for i in row.tolist()} for row in ids]


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
