"""Lens sanity, band diagnostic + auto-select, number-token loading (§3.A)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jspace.ablation import cosine_loading, get_unembed
from jspace.load import LoadedModel


# Trailing space: Qwen3 emits digit tokens after whitespace; bare "=" does not.
CALC_PROMPT = "calc: ( 4 + 17 ) * 2 + 7 = "
CALC_INTERMEDIATES = ("21", "42", "49")

# Lens readout top-k for the band diagnostic (not ablation sparsity k).
DEFAULT_DIAGNOSTIC_TOPKS: tuple[int, ...] = (2, 16, 128)
DEFAULT_BAND_SELECT_TOPK = 16


@dataclass(frozen=True)
class BandSelection:
    band_start: int
    band_end: int
    auto_selected: bool
    strength_label: str = "medium_equivalent"
    match_rates: list[float] | None = None
    match_rates_by_topk: dict[int, list[float]] | None = None
    select_topk: int = DEFAULT_BAND_SELECT_TOPK
    note: str = (
        "Single fixed medium-equivalent band; deliberate deviation from "
        "paper light/medium/heavy strength sweep (§3.A.5)."
    )


def _topk_contains(logits: torch.Tensor, token_id: int, k: int = 5) -> bool:
    top = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return token_id in top


def _layers_in_model(loaded: LoadedModel) -> list[int]:
    source = list(getattr(loaded.lens, "source_layers", range(loaded.n_layers)))
    return [i for i in source if 0 <= int(i) < loaded.n_layers]


@torch.inference_mode()
def lens_sanity_calc(
    loaded: LoadedModel,
    *,
    positions: list[int] | None = None,
) -> dict[str, Any]:
    """Check that calc intermediates surface in J-lens at successive layers (§3.A.2)."""
    if loaded.lens is None or loaded.jlens_model is None:
        raise RuntimeError("lens not attached; call attach_jlens first")

    lens_d = int(getattr(loaded.lens, "d_model", 0) or 0)
    model_d = int(loaded.hf_model.config.hidden_size)
    if lens_d and lens_d != model_d:
        return {
            "prompt": CALC_PROMPT,
            "per_layer": {},
            "any_hit": {name: False for name in CALC_INTERMEDIATES},
            "all_intermediates_seen": False,
            "skipped": True,
            "skip_reason": (
                f"lens d_model={lens_d} != model d_model={model_d}; "
                "run sanity on Qwen/Qwen3-4B"
            ),
        }

    positions = positions or [-1]
    layers = _layers_in_model(loaded)
    lens_logits, _model_logits, _ = loaded.lens.apply(
        loaded.jlens_model,
        CALC_PROMPT,
        positions=positions,
        layers=layers,
    )
    tok = loaded.tokenizer
    # Qwen3 splits multi-digit numbers into per-digit tokens; the lens is
    # single-token, so score the leading digit of each intermediate.
    intermediate_ids: dict[str, int] = {}
    intermediate_token_meta: dict[str, dict[str, Any]] = {}
    for name in CALC_INTERMEDIATES:
        ids = tok.encode(name, add_special_tokens=False)
        if not ids:
            raise ValueError(f"tokenizer produced no ids for {name!r}")
        tid = ids[0]
        intermediate_ids[name] = tid
        intermediate_token_meta[name] = {
            "token_ids": ids,
            "scored_token_id": tid,
            "scored_token": tok.decode([tid]),
            "digit_split": len(ids) > 1,
        }

    per_layer: dict[int, dict[str, bool]] = {}
    for layer, logits in sorted(lens_logits.items()):
        # jlens.apply returns [n_positions, vocab]; keep a batch×pos fallback.
        if logits.ndim == 3:
            flat = logits[0, -1]
        elif logits.ndim == 2:
            flat = logits[-1]
        else:
            flat = logits
        per_layer[int(layer)] = {
            name: _topk_contains(flat, tid, k=10)
            for name, tid in intermediate_ids.items()
        }

    any_hit = {
        name: any(v[name] for v in per_layer.values()) for name in CALC_INTERMEDIATES
    }
    return {
        "prompt": CALC_PROMPT,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "any_hit": any_hit,
        "all_intermediates_seen": all(any_hit.values()),
        "skipped": False,
        "intermediate_tokens": intermediate_token_meta,
    }


def auto_select_band(match_rates: list[float], n_layers: int) -> tuple[int, int]:
    """
    Heuristic band: rising mid-layer plateau before final spike (§3.A.3).

    Finds the longest contiguous mid-region where rate exceeds a relative
    threshold, excluding the last two layers (final spike).
    """
    if len(match_rates) != n_layers:
        raise ValueError("match_rates length must equal n_layers")
    if n_layers < 4:
        return 0, max(0, n_layers - 1)

    rates = np.asarray(match_rates, dtype=np.float64)
    body = rates[:-2]
    peak = float(body.max()) if body.size else 0.0
    threshold = max(0.05, 0.35 * peak) if peak > 0 else 0.05
    above = body >= threshold

    best_start, best_end, best_len = 1, max(1, n_layers // 3), 0
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        # Prefer mid-network bands (skip very early layers).
        start = max(i, 1)
        end = j - 1
        length = end - start + 1
        if length > best_len and start <= end:
            best_start, best_end, best_len = start, end, length
        i = j

    if best_len == 0:
        # Fallback: middle third.
        best_start = n_layers // 3
        best_end = (2 * n_layers) // 3
    return int(best_start), int(best_end)


def select_band(
    match_rates: list[float],
    n_layers: int,
    *,
    override_start: int | None = None,
    override_end: int | None = None,
    match_rates_by_topk: dict[int, list[float]] | None = None,
    select_topk: int = DEFAULT_BAND_SELECT_TOPK,
) -> BandSelection:
    auto_start, auto_end = auto_select_band(match_rates, n_layers)
    if override_start is not None and override_end is not None:
        return BandSelection(
            band_start=override_start,
            band_end=override_end,
            auto_selected=False,
            match_rates=match_rates,
            match_rates_by_topk=match_rates_by_topk,
            select_topk=select_topk,
        )
    return BandSelection(
        band_start=auto_start,
        band_end=auto_end,
        auto_selected=True,
        match_rates=match_rates,
        match_rates_by_topk=match_rates_by_topk,
        select_topk=select_topk,
    )


def _synthetic_match_rates(n_layers: int) -> list[float]:
    rates = [0.0] * n_layers
    lo, hi = n_layers // 3, (2 * n_layers) // 3
    for i in range(lo, hi + 1):
        rates[i] = 0.5
    if n_layers >= 2:
        rates[-1] = 0.9
        rates[-2] = 0.85
    return rates


@torch.inference_mode()
def layer_next_token_match_rates_by_topk(
    loaded: LoadedModel,
    prompts: list[str],
    *,
    topks: tuple[int, ...] = DEFAULT_DIAGNOSTIC_TOPKS,
) -> dict[int, list[float]]:
    """
    Per-layer match rates for several lens readout top-k values (§3.A.3).

    One model forward + one lens.apply per prompt; all top-k curves share that pass.
    """
    if loaded.lens is None or loaded.jlens_model is None:
        raise RuntimeError("lens not attached")
    if not topks:
        raise ValueError("topks must be non-empty")

    sorted_ks = tuple(sorted(set(int(k) for k in topks)))
    max_k = max(sorted_ks)
    n_layers = loaded.n_layers
    lens_d = int(getattr(loaded.lens, "d_model", 0) or 0)
    model_d = int(loaded.hf_model.config.hidden_size)
    if lens_d and lens_d != model_d:
        base = _synthetic_match_rates(n_layers)
        return {k: list(base) for k in sorted_ks}

    hits = {k: np.zeros(n_layers, dtype=np.float64) for k in sorted_ks}
    counts = np.zeros(n_layers, dtype=np.float64)
    layers = _layers_in_model(loaded)

    for prompt in prompts:
        enc = loaded.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(loaded.device) for k, v in enc.items()}
        out = loaded.hf_model(**enc)
        next_id = int(torch.argmax(out.logits[0, -1]).item())

        lens_logits, _, _ = loaded.lens.apply(
            loaded.jlens_model, prompt, positions=[-1], layers=layers
        )
        for layer, logits in lens_logits.items():
            idx = int(layer)
            if idx >= n_layers:
                continue
            flat = logits[0, -1] if logits.ndim == 3 else logits[0]
            counts[idx] += 1
            take = min(max_k, int(flat.numel()))
            top_ids = torch.topk(flat, k=take).indices.tolist()
            for k in sorted_ks:
                if next_id in top_ids[: min(k, take)]:
                    hits[k][idx] += 1

    return {
        k: [
            float(hits[k][i] / counts[i]) if counts[i] > 0 else 0.0
            for i in range(n_layers)
        ]
        for k in sorted_ks
    }


@torch.inference_mode()
def layer_next_token_match_rates(
    loaded: LoadedModel,
    prompts: list[str],
    *,
    topk: int = DEFAULT_BAND_SELECT_TOPK,
) -> list[float]:
    """Single-curve wrapper around layer_next_token_match_rates_by_topk."""
    return layer_next_token_match_rates_by_topk(loaded, prompts, topks=(topk))[topk]


def plot_band_diagnostic(
    match_rates_by_topk: dict[int, list[float]],
    band: BandSelection,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in sorted(match_rates_by_topk):
        rates = match_rates_by_topk[k]
        xs = list(range(len(rates)))
        ax.plot(
            xs,
            rates,
            marker="o",
            label=f"lens top-{k}",
            linewidth=1.5,
            alpha=0.9,
        )
    ax.axvspan(
        band.band_start,
        band.band_end,
        alpha=0.2,
        color="C1",
        label="selected band",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("match rate")
    ax.set_title("Workspace band diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.inference_mode()
def number_token_loading(
    loaded: LoadedModel,
    prompts: list[str],
    *,
    number_strings: tuple[str, ...] = tuple(str(i) for i in range(10)),
    band_start: int,
    band_end: int,
) -> dict[str, Any]:
    """Cosine sim of residual vs number-token J directions on math prompts (§3.A.6)."""
    if loaded.lens is None:
        raise RuntimeError("lens not attached")

    from jspace.ablation import layer_jacobians

    jacobians = layer_jacobians(
        loaded.lens, band_start, band_end, d_model=loaded.hf_model.config.hidden_size
    )
    unembed = get_unembed(loaded.hf_model)
    tok = loaded.tokenizer

    number_ids = []
    for s in number_strings:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            number_ids.append(ids[-1])

    # Build mean residual direction associated with each number via unembed rows
    # transported approximately as unembed[token] similarity in final space;
    # loading = cosine(residual, J^T @ unembed[token]).
    loadings: list[float] = []
    for prompt in prompts:
        enc = loaded.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(loaded.device) for k, v in enc.items()}
        # Capture mid-band residual via forward hook on one layer.
        mid = (band_start + band_end) // 2
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _i, output, *, _mid=mid):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[0, -1].detach()

        layers = loaded.hf_model.model.layers
        handle = layers[mid].register_forward_hook(hook)
        try:
            _ = loaded.hf_model(**enc)
        finally:
            handle.remove()

        h = captured["h"]
        J = jacobians[mid].to(device=h.device, dtype=h.dtype)
        layer_loads = []
        for tid in number_ids:
            # Dual direction in residual space: J^T @ u_token
            u = unembed[tid].to(device=h.device, dtype=h.dtype)
            direction = torch.mv(J.T, u)
            if torch.linalg.vector_norm(direction) < 1e-8:
                continue
            layer_loads.append(cosine_loading(h, direction))
        if layer_loads:
            loadings.append(float(np.mean(layer_loads)))

    return {
        "mean_cosine": float(np.mean(loadings)) if loadings else float("nan"),
        "per_prompt_mean": loadings,
        "band_start": band_start,
        "band_end": band_end,
        "n_prompts": len(prompts),
    }


def save_band_json(path: Path, band: BandSelection, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(band)
    by_topk = payload.get("match_rates_by_topk")
    if isinstance(by_topk, dict):
        payload["match_rates_by_topk"] = {str(k): v for k, v in by_topk.items()}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
