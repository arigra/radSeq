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


def test_hann_reduction_roundtrip_identity():
    """Weighted overlap-add still reconstructs consistent patches exactly."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    t = patchify(x, p=8, s=4)
    y = unpatchify(t, N=64, K=64, p=8, s=4, reduction="hann")
    assert torch.allclose(x, y, atol=1e-5)


def test_hann_preserves_a_peak_the_mean_dilutes():
    """When one patch predicts a peak and its neighbours predict nothing,
    the mean splits the difference four ways; raised-cosine weighting lets the
    patch that sees the pixel centrally dominate."""
    pr, pc = num_patches(16, 16, 8, 4)
    t = torch.zeros(1, 1, pr * pc, 64)
    t[0, 0, 0] = 2.0                      # only the patch at (0, 0) sees a peak
    mean = unpatchify(t, N=16, K=16, p=8, s=4, reduction="mean")
    hann = unpatchify(t, N=16, K=16, p=8, s=4, reduction="hann")
    # pixel (4, 4) is covered by four patches; it is central only to patch (0,0)
    assert hann[0, 0, 4, 4] > mean[0, 0, 4, 4]
