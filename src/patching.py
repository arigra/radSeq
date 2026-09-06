"""Overlapping patch extraction and reassembly for RD maps.

Patches of size p with stride s < p overlap; unpatchify averages
contributions in overlapped regions (F.fold divided by hit counts),
so patchify -> unpatchify is the identity on raw maps.
"""
import math

import torch
import torch.nn.functional as F


def num_patches(N: int, K: int, p: int, s: int) -> tuple[int, int]:
    return ((N - p) // s + 1, (K - p) // s + 1)


def patchify(x: torch.Tensor, p: int = 8, s: int = 4) -> torch.Tensor:
    """(B, L, N, K) -> (B, L, P, p*p)"""
    B, L, N, K = x.shape
    u = F.unfold(x.reshape(B * L, 1, N, K), kernel_size=p, stride=s)  # (B*L, p*p, P)
    P = u.shape[-1]
    return u.transpose(1, 2).reshape(B, L, P, p * p)


def _raised_cosine_window(p: int, device, dtype) -> torch.Tensor:
    """Separable raised-cosine weights over a p x p patch, flattened.

    Sampled at half-integer positions so no weight is exactly zero: a border
    pixel covered by a single patch still has that patch's prediction.
    """
    n = torch.arange(p, device=device, dtype=torch.float32)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * (n + 0.5) / p)
    return (w[:, None] * w[None, :]).reshape(-1).to(dtype)


def unpatchify(tokens: torch.Tensor, N: int = 64, K: int = 64,
               p: int = 8, s: int = 4,
               reduction: str = "mean") -> torch.Tensor:
    """Reassemble ``(B, L, P, p*p)`` patch predictions.

    ``mean`` is the training/default path and averages every prediction for an
    overlapped pixel. ``tile`` is a diagnostic path: it chooses the patches
    starting at ``(0, 0), (0, p), ...`` and stitches that non-overlapping
    subset. Comparing the two on the same checkpoint and initial diffusion
    noise directly measures how much overlap averaging changes a sample.
    """
    B, L, P, d = tokens.shape
    if reduction == "tile":
        if p % s or N % p or K % p:
            raise ValueError(
                "tile reduction requires stride to divide patch size and "
                "patch size to divide both output dimensions")
        pr, pc = num_patches(N, K, p, s)
        if P != pr * pc or d != p * p:
            raise ValueError("token shape is incompatible with output geometry")
        grid = tokens.reshape(B, L, pr, pc, p, p)
        step = p // s
        tiled = grid[:, :, ::step, ::step]
        expected = (N // p, K // p)
        if tiled.shape[2:4] != expected:
            raise ValueError("selected patches do not tile the output")
        return (tiled.permute(0, 1, 2, 4, 3, 5)
                .reshape(B, L, N, K))
    if reduction == "hann":
        # Weighted overlap-add. A plain mean gives every patch an equal vote on
        # a pixel, so four disagreeing predictions of one sharp peak average
        # into something duller than any of them. Raised-cosine weights make a
        # pixel be decided mostly by the patch that sees it centrally, while
        # still blending across boundaries so no seam appears.
        w = _raised_cosine_window(p, tokens.device, tokens.dtype)   # (p*p,)
        u = (tokens.reshape(B * L, P, d) * w).transpose(1, 2)
        out = F.fold(u, (N, K), kernel_size=p, stride=s)
        weights = w.view(1, d, 1).expand(B * L, d, P)
        cnt = F.fold(weights, (N, K), kernel_size=p, stride=s)
        return (out / cnt).reshape(B, L, N, K)
    if reduction != "mean":
        raise ValueError(f"unknown overlap reduction {reduction!r}")
    u = tokens.reshape(B * L, P, d).transpose(1, 2)          # (B*L, p*p, P)
    out = F.fold(u, (N, K), kernel_size=p, stride=s)
    cnt = F.fold(torch.ones_like(u), (N, K), kernel_size=p, stride=s)
    return (out / cnt).reshape(B, L, N, K)
