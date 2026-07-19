import torch
from src.eval.metrics import velocity_adherence


def _blob_seq(dop_bin, L=16):
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        r = 10 + 0.5 * l
        frames.append(30 * torch.exp(-((rr - r) ** 2 + (cc - dop_bin) ** 2) / 2.0)
                      + 0.1 * torch.randn(64, 64))
    return torch.stack(frames)


def test_velocity_adherence_perfect():
    torch.manual_seed(0)
    v_cmd = torch.tensor([-5.0, -2.0, 1.0, 4.0, 7.0])
    dop_bins = ((v_cmd - (-7.987220447284345)) / 0.2496006389776358).round()
    x = torch.stack([_blob_seq(b) for b in dop_bins])
    corr = velocity_adherence(x, v_cmd)
    assert corr > 0.95


def test_velocity_adherence_random_is_low():
    torch.manual_seed(1)
    v_cmd = torch.tensor([-5.0, -2.0, 1.0, 4.0, 7.0])
    dop_bins = torch.randint(5, 59, (5,)).float()
    x = torch.stack([_blob_seq(b) for b in dop_bins])
    corr = velocity_adherence(x, v_cmd)
    assert abs(corr) < 0.9
