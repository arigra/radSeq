"""Exponential moving averages for diffusion-model sampling."""
from copy import deepcopy

import torch


def make_ema(module):
    """Return a frozen copy initialized exactly from ``module``."""
    ema = deepcopy(module).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, source, decay):
    """Move ``ema`` parameters toward ``source`` and copy its buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    source_params = dict(source.named_parameters())
    for name, target in ema.named_parameters():
        target.lerp_(source_params[name].detach(), 1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, target in ema.named_buffers():
        target.copy_(source_buffers[name].detach())
