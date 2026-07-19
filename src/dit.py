"""Temporal-only Diffusion Transformer for RD sequences.

Tokens: overlapping patches per frame. Attention runs over the L frames
of each spatial site independently ((B*P, L, d) reshape) — no spatial
mixing beyond patch overlap, per the proposal. adaLN-Zero conditioning
on diffusion timestep (+ optional condition vector added to it).
"""
import math

import torch
import torch.nn as nn

from src.patching import patchify, unpatchify, num_patches


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, z, c):
        # z: (B, L, P, d); c: (B, d)
        B, L, P, d = z.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c)[:, None, None].chunk(6, dim=-1)
        h = self.norm1(z) * (1 + sc1) + sh1
        h = h.permute(0, 2, 1, 3).reshape(B * P, L, d)
        a, _ = self.attn(h, h, h, need_weights=False)
        a = a.reshape(B, P, L, d).permute(0, 2, 1, 3)
        z = z + g1 * a
        h = self.norm2(z) * (1 + sc2) + sh2
        return z + g2 * self.mlp(h)


class TemporalDiT(nn.Module):
    def __init__(self, seq_len=16, N=64, K=64, patch=8, stride=4,
                 dim=256, depth=8, heads=8):
        super().__init__()
        self.N, self.K, self.p, self.s = N, K, patch, stride
        pr, pc = num_patches(N, K, patch, stride)
        P = pr * pc
        self.proj = nn.Linear(patch * patch, dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, P, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, seq_len, 1, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                   nn.Linear(dim, dim))
        self.dim = dim
        self.blocks = nn.ModuleList(TemporalBlock(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond=None):
        tokens = patchify(x, self.p, self.s)                  # (B, L, P, p*p)
        z = self.proj(tokens) + self.spatial_pos + self.temporal_pos
        c = self.t_mlp(timestep_embedding(t, self.dim))
        if cond is not None:
            c = c + cond
        for blk in self.blocks:
            z = blk(z, c)
        sh, sc = self.final_adaLN(c)[:, None, None].chunk(2, dim=-1)
        z = self.final_norm(z) * (1 + sc) + sh
        return unpatchify(self.out(z), self.N, self.K, self.p, self.s)
