import torch
from src.eval.metrics import (detect_peaks, link_tracks, velocity_consistency,
                              doppler_drift, persistence, marginal_l1,
                              evaluate_sequences)


def _synthetic_sequence(vel=(1.0, 0.5), start=(10.0, 20.0), L=16, noise=0.1):
    """One Gaussian blob moving at constant velocity on a noisy floor."""
    torch.manual_seed(0)
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        r = start[0] + vel[0] * l
        c = start[1] + vel[1] * l
        blob = 30 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 2.0)
        frames.append(blob + noise * torch.randn(64, 64))
    return torch.stack(frames)


def test_detect_peaks_finds_blob():
    x = _synthetic_sequence()
    pk = detect_peaks(x[0])
    assert len(pk) >= 1
    assert abs(pk[0][0] - 10.0) <= 1 and abs(pk[0][1] - 20.0) <= 1


def test_linking_and_consistency_constant_velocity():
    x = _synthetic_sequence()
    tracks = link_tracks([detect_peaks(f) for f in x])
    assert persistence(tracks, 16) > 0.99
    # integer-pixel peak detection of a blob moving 0.5 px/frame produces
    # alternating +/-1 rounding jitter in the second difference (~0.7);
    # still orders of magnitude below teleporting motion
    assert velocity_consistency(tracks) < 1.0


def test_consistency_penalizes_teleporting():
    x1 = _synthetic_sequence()
    torch.manual_seed(1)
    # teleporting blob: random position each frame
    frames = []
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    pos = torch.rand(16, 2) * 40 + 10
    for l in range(16):
        blob = 30 * torch.exp(-((rr - pos[l, 0])**2 + (cc - pos[l, 1])**2) / 2.0)
        frames.append(blob + 0.1 * torch.randn(64, 64))
    x2 = torch.stack(frames)
    t1 = link_tracks([detect_peaks(f) for f in x1])
    t2 = link_tracks([detect_peaks(f) for f in x2], gate=100.0)
    assert velocity_consistency(t2) > 10 * max(velocity_consistency(t1), 1e-6)


def test_marginal_l1_identical_is_zero():
    x = _synthetic_sequence()
    assert marginal_l1(x[None], x[None]) < 1e-9


def test_evaluate_sequences_keys():
    x = _synthetic_sequence()[None]
    out = evaluate_sequences(x, x)
    for k in ("velocity_consistency", "doppler_drift", "persistence",
              "marginal_l1", "mean_tracks_per_seq"):
        assert k in out
