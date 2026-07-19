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
