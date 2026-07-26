"""GPU check: our J-lens readout must agree with jlens' own ``lens.apply``.

``select_active_j_lens_directions`` picks the ablated directions from
``lens_logits_for_residual``. If that readout disagrees with the reference
lens, the "most activated J-lens tokens" are not lens tokens at all and every
downstream gate number is meaningless — which is exactly how the SVD-direction
selection bug survived. Compare the two readouts head-on.
"""

from __future__ import annotations

import pytest
import torch

AGREEMENT_PROMPT = "The capital of the most populous country in the world is"
TOP_K = 10
DEFAULT_BAND = (27, 33)


def _band_layers(lens, n_layers: int, band_start: int, band_end: int) -> list[int]:
    """Up to three fitted layers spread across the band."""
    fitted = [
        layer
        for layer in lens.source_layers
        if band_start <= layer <= band_end and 0 <= layer < n_layers
    ]
    if not fitted:
        raise AssertionError(
            f"no fitted lens layers in band [{band_start}, {band_end}]; "
            f"source_layers={lens.source_layers}"
        )
    if len(fitted) <= 3:
        return fitted
    return [fitted[0], fitted[len(fitted) // 2], fitted[-1]]


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_lens_logits_for_residual_matches_jlens_apply():
    from jlens.hooks import ActivationRecorder

    from jspace.ablation import (
        AblationConfig,
        build_ablation_factors,
        lens_logits_for_residual,
    )
    from jspace.config import load_settings
    from jspace.load import load_model_and_lens

    settings = load_settings()
    loaded = load_model_and_lens(settings)
    jlens_model = loaded.jlens_model
    lens = loaded.lens

    band_start = settings.band_start if settings.band_start is not None else DEFAULT_BAND[0]
    band_end = settings.band_end if settings.band_end is not None else DEFAULT_BAND[1]
    layers = _band_layers(lens, loaded.n_layers, band_start, band_end)

    input_ids = jlens_model.encode(AGREEMENT_PROMPT)
    seq_len = int(input_ids.shape[-1])
    # Last position is the prediction site the ablation actually cares about.
    positions = sorted({max(0, seq_len // 2), max(0, seq_len - 2), seq_len - 1})

    ref_logits, _model_logits, ref_ids = lens.apply(
        jlens_model,
        AGREEMENT_PROMPT,
        layers=layers,
        positions=positions,
    )
    assert torch.equal(ref_ids.cpu(), input_ids.cpu()), (
        "lens.apply tokenized the prompt differently than jlens_model.encode"
    )

    with ActivationRecorder(jlens_model.layers, at=layers) as recorder:
        jlens_model.forward(input_ids)
        residuals = {layer: recorder.activations[layer].detach() for layer in layers}

    factors = build_ablation_factors(
        loaded.hf_model,
        lens,
        AblationConfig(
            kind="jspace",
            band_start=min(layers),
            band_end=max(layers),
            k=TOP_K,
        ),
    )

    mismatches: list[str] = []
    for layer in layers:
        for slot, pos in enumerate(positions):
            h = residuals[layer][0, pos]
            ours = lens_logits_for_residual(
                h,
                factors.jacobians[layer],
                factors.unembed_f32,
                final_norm=factors.final_norm,
            )
            ours_top = torch.topk(ours.float().cpu(), TOP_K).indices.tolist()
            ref_top = torch.topk(ref_logits[layer][slot].float(), TOP_K).indices.tolist()
            if set(ours_top) != set(ref_top) or ours_top[0] != ref_top[0]:
                decode = loaded.tokenizer.decode
                mismatches.append(
                    f"L{layer} pos={pos}: "
                    f"ours={[decode([t]) for t in ours_top]} "
                    f"ref={[decode([t]) for t in ref_top]}"
                )

    assert not mismatches, (
        "J-lens readout disagrees with jlens lens.apply top-"
        f"{TOP_K} at {len(mismatches)}/{len(layers) * len(positions)} "
        "(layer, position) sites:\n" + "\n".join(mismatches)
    )
