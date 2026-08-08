import torch
from src.dit import TemporalDiT, timestep_embedding


def _small_model():
    return TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=4,
                       dim=32, depth=2, heads=4)


def test_timestep_embedding_shape():
    e = timestep_embedding(torch.tensor([0, 500]), 32)
    assert e.shape == (2, 32)


def test_forward_shape():
    torch.manual_seed(0)
    m = _small_model()
    x = torch.randn(2, 4, 16, 16)
    out = m(x, torch.tensor([10, 20]))
    assert out.shape == (2, 4, 16, 16)


def test_zero_init_output():
    """adaLN-Zero: an untrained model must output exactly zero."""
    m = _small_model()
    x = torch.randn(1, 4, 16, 16)
    out = m(x, torch.tensor([5]))
    assert out.abs().max() == 0.0


def test_spatial_independence():
    """Perturbing a distant spatial region must not change the output
    in a region whose overlapping patches don't cover it.
    With patch=8/stride=4 on 16x16, pixel (0,0) is only in patch (0,0)
    covering rows/cols 0-7; pixel (15,15) is only in patch (2,2)
    covering rows/cols 8-15. Disjoint -> output at (0,0) fixed."""
    torch.manual_seed(0)
    m = _small_model()
    for p in m.parameters():  # break zero-init so the test is non-trivial
        torch.nn.init.normal_(p, std=0.02)
    m.eval()
    x = torch.randn(1, 4, 16, 16)
    x2 = x.clone()
    x2[:, :, 12:, 12:] += 10.0
    with torch.no_grad():
        o1 = m(x, torch.tensor([100]))
        o2 = m(x2, torch.tensor([100]))
    assert torch.allclose(o1[:, :, :4, :4], o2[:, :, :4, :4], atol=1e-5)


# ---- factorized spatial+temporal variant (control experiment) -------------

def _small_factorized():
    return TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=4,
                       dim=32, depth=2, heads=4, attn_mode="factorized")


def test_factorized_forward_shape():
    torch.manual_seed(0)
    m = _small_factorized()
    out = m(torch.randn(2, 4, 16, 16), torch.tensor([10, 20]))
    assert out.shape == (2, 4, 16, 16)


def test_factorized_zero_init_output():
    """adaLN-Zero must gate the spatial sublayer too, so an untrained
    factorized model still outputs exactly zero."""
    m = _small_factorized()
    out = m(torch.randn(1, 4, 16, 16), torch.tensor([5]))
    assert out.abs().max() == 0.0


def test_factorized_mixes_spatially():
    """The mirror of test_spatial_independence: with spatial attention a
    perturbation in a disjoint patch MUST reach the output at (0,0).
    This is the whole point of the control experiment."""
    torch.manual_seed(0)
    m = _small_factorized()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.02)
    m.eval()
    x = torch.randn(1, 4, 16, 16)
    x2 = x.clone()
    x2[:, :, 12:, 12:] += 10.0
    with torch.no_grad():
        o1 = m(x, torch.tensor([100]))
        o2 = m(x2, torch.tensor([100]))
    delta = (o1[:, :, :4, :4] - o2[:, :, :4, :4]).abs().max()
    assert delta > 1e-4, f"spatial attention did not propagate (delta={delta})"


def test_factorized_depth5_stays_under_baseline_params():
    """The control brackets the baseline's parameter count from below, so a
    positive result cannot be explained by extra capacity. Guards that."""
    base = TemporalDiT(seq_len=16, N=64, K=64, patch=8, stride=4,
                       dim=256, depth=8, heads=8)
    small = TemporalDiT(seq_len=16, N=64, K=64, patch=8, stride=4,
                        dim=256, depth=5, heads=8, attn_mode="factorized")
    n_base = sum(p.numel() for p in base.parameters())
    n_small = sum(p.numel() for p in small.parameters())
    assert n_small < n_base, f"{n_small} !< {n_base}"
