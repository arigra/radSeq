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
