import numpy as np


def fake_frame(n_range=64, n_doppler=32, target=(20, 10), seed=0):
    """Complex RD frame with 16 Rx: noise plus one strong target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_range, n_doppler, 16)) + 1j * rng.normal(size=(n_range, n_doppler, 16))
    x[target] += 30
    return x


def power_map(x):
    """RD power in dB, summed over receive channels."""
    return 10 * np.log10((np.abs(x) ** 2).sum(axis=-1) + 1e-12)
