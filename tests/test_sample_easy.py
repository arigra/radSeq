import torch

from src.sample import generate_easy


def test_easy_sampler_reproducible_and_contains_moving_target():
    x = generate_easy(2, seed=11, seq_len=8)
    assert x.shape == (2, 8, 64, 64)
    assert torch.equal(x, generate_easy(2, seed=11, seq_len=8))
    assert torch.isfinite(x).all()
    # A sampled sequence has a bright target and nonconstant frames.
    assert (x.amax(dim=(-2, -1)) - x.median(dim=-1).values.median(dim=-1).values).min() > 20
    assert (x[:, 1:] - x[:, :-1]).abs().mean() > 0.1
