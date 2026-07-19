import torch
from src.eval.metrics import cfar_detection_stats


def _seq_with_targets(n_targets, L=4):
    torch.manual_seed(0)
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for _ in range(L):
        f = torch.randn(64, 64).abs() * 0.5
        for k in range(n_targets):
            r, c = 10 + 12 * k, 15 + 10 * k
            f = f + 25 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 1.5)
        frames.append(20 * torch.log10(f + 1e-6))
    return torch.stack(frames)


def test_more_targets_more_detections():
    x0 = _seq_with_targets(0)[None]
    x3 = _seq_with_targets(3)[None]
    d0 = cfar_detection_stats(x0)
    d3 = cfar_detection_stats(x3)
    assert d3 > d0 + 1.0
