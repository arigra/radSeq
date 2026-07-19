import torch
from src.simulator import TemporalRadarSimulator


def _clutter_corr(sim, rho, n_seq=8):
    """Mean lag-1 correlation of clutter IQ across frames."""
    torch.manual_seed(0)
    cors = []
    for _ in range(n_seq):
        C = sim._clutter_frames(rho=rho, nu=0.5)          # (L, 64, 64)
        a = C[:-1].flatten(1)
        b = C[1:].flatten(1)
        num = (a.conj() * b).sum(dim=1).real
        den = (a.abs().pow(2).sum(dim=1).sqrt()
               * b.abs().pow(2).sum(dim=1).sqrt())
        cors.append((num / den).mean())
    return torch.stack(cors).mean().item()


def test_clutter_shape_and_nonzero():
    torch.manual_seed(0)
    sim = TemporalRadarSimulator(seq_len=16)
    C = sim._clutter_frames(rho=0.5, nu=0.5)
    assert C.shape == (16, 64, 64) and C.dtype == torch.cfloat
    assert C.abs().sum() > 0


def test_ar1_correlation_tracks_rho():
    sim = TemporalRadarSimulator(seq_len=16)
    c_lo = _clutter_corr(sim, rho=0.1)
    c_hi = _clutter_corr(sim, rho=0.9)
    assert c_hi > c_lo + 0.3
    assert abs(c_hi - 0.9) < 0.15


def test_rho_zero_uncorrelated():
    sim = TemporalRadarSimulator(seq_len=16)
    assert abs(_clutter_corr(sim, rho=0.0)) < 0.15
