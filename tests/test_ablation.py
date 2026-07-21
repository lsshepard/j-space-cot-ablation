import torch

from jspace.ablation import (
    active_j_directions,
    clean_topk_token_ids,
    filter_directions_by_exclusion,
    gram_schmidt,
    project_out_directions,
    random_directions,
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
    dirs = active_j_directions(h, J, k=3)
    assert dirs.shape[0] <= 3
    assert dirs.shape[1] == d


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
    d = 4
    J = torch.eye(d)
    unembed = torch.eye(10, d)  # vocab 10, d_model 4 — pad
    unembed = torch.nn.functional.pad(unembed, (0, 0, 0, 0))[:10, :d]
    # Make token 2 align with e0 after J
    unembed = torch.zeros(10, d)
    unembed[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    kept = filter_directions_by_exclusion(directions, J, unembed, excluded_token_ids={2})
    assert kept.shape[0] == 1
    assert torch.allclose(kept[0], directions[1])
