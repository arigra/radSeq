import torch
from src.patching import patchify, unpatchify, num_patches


def test_num_patches():
    assert num_patches(64, 64, 8, 4) == (15, 15)


def test_patchify_shape():
    x = torch.randn(2, 16, 64, 64)
    t = patchify(x, p=8, s=4)
    assert t.shape == (2, 16, 225, 64)


def test_roundtrip_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64, 64)
    t = patchify(x, p=8, s=4)
    y = unpatchify(t, N=64, K=64, p=8, s=4)
    assert torch.allclose(x, y, atol=1e-5)


def test_tile_reduction_roundtrip_identity():
    """A deterministic non-overlapping patch subset also covers every pixel."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 16)
    t = patchify(x, p=8, s=4)
    y = unpatchify(t, N=16, K=16, p=8, s=4, reduction="tile")
    assert torch.equal(x, y)


def test_tile_reduction_exposes_overlap_disagreement():
    """Mean and tile agree only when overlapping predictions agree."""
    t = torch.zeros(1, 1, 9, 64)
    t[:, :, 0] = 2.0
    mean = unpatchify(t, N=16, K=16, p=8, s=4, reduction="mean")
    tile = unpatchify(t, N=16, K=16, p=8, s=4, reduction="tile")
    assert not torch.equal(mean, tile)
    assert tile[:, :, :8, :8].eq(2.0).all()
