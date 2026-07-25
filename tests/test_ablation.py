import torch

from jspace.ablation import (
    active_j_directions,
    active_j_directions_from_svd,
    chunk_positions_to_ablate,
    clean_topk_by_position,
    clean_topk_token_ids,
    factor_jacobian_svd,
    filter_directions_by_exclusion,
    gram_schmidt,
    past_token_count_from_cache,
    positions_to_ablate,
    project_out_directions,
    random_directions,
    scale_perturbation_to_norm,
)


def test_gram_schmidt_orthonormal():
    raw = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    basis = gram_schmidt(raw)
    assert basis.shape[0] == 2
    grams = basis @ basis.T
    assert torch.allclose(grams, torch.eye(2), atol=1e-5)


def test_project_out_removes_component():
    direction = torch.tensor([[1.0, 0.0, 0.0]])
    hidden = torch.tensor([3.0, 4.0, 0.0])
    out = project_out_directions(hidden, direction)
    assert abs(float(out[0])) < 1e-5
    assert abs(float(out[1]) - 4.0) < 1e-5


def test_active_j_directions_shape():
    d = 8
    J = torch.eye(d)
    h = torch.randn(d)
    unembed = torch.randn(20, d)
    dirs = active_j_directions(h, J, k=3, unembed=unembed)
    assert dirs.shape[0] <= 3
    assert dirs.shape[1] == d


def test_cached_svd_matches_fresh_svd():
    torch.manual_seed(0)
    d = 16
    J = torch.randn(d, d)
    h = torch.randn(d)
    svd = factor_jacobian_svd(J)
    fresh = active_j_directions_from_svd(h, svd, k=4)
    cached = active_j_directions_from_svd(h, svd, k=4)
    via_kw = active_j_directions(h, J, k=4, svd=svd)
    assert torch.allclose(fresh, cached, atol=1e-5)
    assert torch.allclose(fresh, via_kw, atol=1e-5)


def test_select_active_j_lens_skips_clean_topk_and_uses_token_vectors():
    from jspace.ablation import (
        j_lens_vectors_for_tokens,
        lens_logits_for_residual,
        select_active_j_lens_directions,
    )

    torch.manual_seed(0)
    d = 4
    J = torch.eye(d)
    # Token i's unembed row = e_i so lens logits ≈ residual coords.
    unembed = torch.eye(d)
    h = torch.tensor([0.1, 5.0, 4.0, 3.0])
    scores = lens_logits_for_residual(h, J, unembed)
    assert int(torch.argmax(scores).item()) == 1
    dirs, infos = select_active_j_lens_directions(
        h, J, unembed, k=2, excluded_token_ids={1}
    )
    # Top lens token 1 excluded → survivors are tokens 2 and 3.
    survivor_ids = [i.top_token_id for i in infos if not i.excluded][:2]
    assert survivor_ids == [2, 3]
    assert dirs.shape[0] == 2
    expected = j_lens_vectors_for_tokens([2, 3], J, unembed)
    # After gram_schmidt, span should match.
    assert dirs.shape == expected.shape


def test_chunk_positions_with_past_matches_absolute():
    # Full-prefix ablate: first chunk ablates all local indices.
    assert chunk_positions_to_ablate(
        4,
        past_token_count=0,
        ablate_prompt_tokens=True,
        prompt_token_count=4,
    ) == [0, 1, 2, 3]
    # Later cached step: only the new token (absolute 4) is in-chunk.
    assert chunk_positions_to_ablate(
        1,
        past_token_count=4,
        ablate_prompt_tokens=True,
        prompt_token_count=4,
    ) == [0]
    # Generation-only: prompt already past, new gen token ablates.
    assert chunk_positions_to_ablate(
        1,
        past_token_count=3,
        ablate_prompt_tokens=False,
        prompt_token_count=3,
    ) == [0]


def test_past_token_count_from_cache_none_and_tuple():
    assert past_token_count_from_cache(None) == 0
    key = torch.zeros(1, 2, 7, 4)
    value = torch.zeros(1, 2, 7, 4)
    past = ((key, value),)
    assert past_token_count_from_cache(past) == 7


def test_random_directions_seed_stable():
    a = random_directions(16, 4, seed=7, device=torch.device("cpu"), dtype=torch.float32)
    b = random_directions(16, 4, seed=7, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(a, b)


def test_clean_topk_token_ids():
    logits = torch.zeros(1, 2, 10)
    logits[0, -1, 3] = 10
    logits[0, -1, 5] = 9
    ids = clean_topk_token_ids(logits, k=2)
    assert ids == {3, 5}


def test_clean_topk_by_position_is_local():
    logits = torch.zeros(1, 3, 10)
    logits[0, 0, 1] = 10
    logits[0, 1, 4] = 10
    logits[0, 2, 7] = 10
    by_pos = clean_topk_by_position(logits, k=1)
    assert by_pos[0] == {1}
    assert by_pos[1] == {4}
    assert by_pos[2] == {7}


def test_scale_perturbation_matches_target_norm():
    original = torch.tensor([[3.0, 4.0, 0.0]])
    # Weak ablation: remove only a little.
    ablated = torch.tensor([[2.9, 4.0, 0.0]])
    target = torch.tensor([[2.0]])
    out = scale_perturbation_to_norm(original, ablated, target)
    delta = original - out
    assert abs(float(torch.linalg.vector_norm(delta)) - 2.0) < 1e-5


def test_layer_jacobians_proxy_on_mismatch():
    import warnings
    from types import SimpleNamespace

    from jspace.ablation import layer_jacobians

    lens = SimpleNamespace(
        jacobians={2: torch.randn(8, 8), 3: torch.randn(8, 8)}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        mats = layer_jacobians(lens, 2, 3, d_model=4)
    assert mats[2].shape == (4, 4)
    assert torch.allclose(mats[2], torch.eye(4))


def test_filter_exclusion_drops_matching_direction():
    from jspace.ablation import classify_directions_by_exclusion, direction_top_tokens

    d = 4
    J = torch.eye(d)
    unembed = torch.zeros(10, d)
    unembed[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    unembed[5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    kept = filter_directions_by_exclusion(directions, J, unembed, excluded_token_ids={2})
    assert kept.shape[0] == 1
    assert torch.allclose(kept[0], directions[1])
    kept2, infos = classify_directions_by_exclusion(
        directions, J, unembed, {2}
    )
    assert torch.allclose(kept, kept2)
    assert infos[0].excluded and infos[0].top_token_id == 2
    assert not infos[1].excluded
    assert direction_top_tokens(directions, J, unembed) == [2, 5]


def test_build_ablation_factors_moves_jacobian_to_model_device():
    """Lens J often loads on CPU; factors must move to the model device."""
    from types import SimpleNamespace

    from jspace.ablation import AblationConfig, build_ablation_factors

    device = torch.device("cpu")
    d = 8
    lm_head = SimpleNamespace(weight=torch.randn(16, d, device=device))
    hf_model = SimpleNamespace(lm_head=lm_head, model=SimpleNamespace(norm=None))
    lens = SimpleNamespace(
        jacobians={2: torch.randn(d, d), 3: torch.randn(d, d)}  # CPU by default
    )
    factors = build_ablation_factors(
        hf_model,
        lens,
        AblationConfig(kind="jspace", band_start=2, band_end=3, k=3),
    )
    for mat in factors.jacobians.values():
        assert mat.device == device
    assert factors.svds == {}


def test_positions_to_ablate_prompt_and_generation():
    assert list(positions_to_ablate(5, ablate_prompt_tokens=True, prompt_token_count=3)) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(
        positions_to_ablate(5, ablate_prompt_tokens=False, prompt_token_count=3)
    ) == [3, 4]
    assert list(
        positions_to_ablate(3, ablate_prompt_tokens=False, prompt_token_count=3)
    ) == [2]


def test_ablation_hook_state_tracks_prompt_len():
    from jspace.ablation import AblationHookState

    state = AblationHookState(prompt_token_count=12)
    assert state.prompt_token_count == 12
    assert state.excluded_by_position == []
