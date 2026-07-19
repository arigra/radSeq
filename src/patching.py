"""Overlapping patch extraction and reassembly for RD maps.

Patches of size p with stride s < p overlap; unpatchify averages
contributions in overlapped regions (F.fold divided by hit counts),
so patchify -> unpatchify is the identity on raw maps.
"""
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


def unpatchify(tokens: torch.Tensor, N: int = 64, K: int = 64,
               p: int = 8, s: int = 4) -> torch.Tensor:
    """(B, L, P, p*p) -> (B, L, N, K), averaging overlaps."""
    B, L, P, d = tokens.shape
    u = tokens.reshape(B * L, P, d).transpose(1, 2)          # (B*L, p*p, P)
    out = F.fold(u, (N, K), kernel_size=p, stride=s)
    cnt = F.fold(torch.ones_like(u), (N, K), kernel_size=p, stride=s)
    return (out / cnt).reshape(B, L, N, K)
