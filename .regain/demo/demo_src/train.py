import numpy as np


def train(clean, steps=200, lr=0.1, noise_std=0.5, seed=0):
    """Learn w so that w * noisy is close to clean. Returns w and the loss history."""
    rng = np.random.default_rng(seed)
    x = (clean - clean.mean()) / clean.std()
    w, losses = 0.0, []
    for _ in range(steps):
        noisy = x + noise_std * rng.normal(size=x.shape)
        err = w * noisy - x
        losses.append((err ** 2).mean())
        w -= lr * 2 * (err * noisy).mean()
    return w, losses
