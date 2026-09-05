"""A small conventional spatial U-Net baseline for radar sequences.

The sequence frames are input/output channels of a 2-D U-Net. This removes
patch extraction and overlap averaging entirely, gives every prediction a
spatial convolutional receptive field, and lets the network mix all frames.
It intentionally shares the DiT call signature so the diffusion objective and
samplers are unchanged.
"""
import torch
import torch.nn as nn

from src.dit import timestep_embedding


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))
        self.act = nn.SiLU()

    def forward(self, x, time_condition):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(time_condition))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SpatialUNet(nn.Module):
    """Two-level 2-D U-Net with diffusion-timestep conditioning."""

    def __init__(self, seq_len=16, base_channels=64, time_dim=256):
        super().__init__()
        b = base_channels
        self.dim = time_dim
        self.in_conv = nn.Conv2d(seq_len, b, 3, padding=1)
        self.t_mlp = nn.Sequential(
            nn.Linear(b, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim))

        self.down1 = nn.ModuleList([
            ResBlock(b, b, time_dim), ResBlock(b, b, time_dim)])
        self.downsample1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.down2 = nn.ModuleList([
            ResBlock(2 * b, 2 * b, time_dim),
            ResBlock(2 * b, 2 * b, time_dim)])
        self.downsample2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)

        self.mid = nn.ModuleList([
            ResBlock(4 * b, 4 * b, time_dim),
            ResBlock(4 * b, 4 * b, time_dim)])

        self.upsample2 = nn.ConvTranspose2d(
            4 * b, 2 * b, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResBlock(4 * b, 2 * b, time_dim),
            ResBlock(2 * b, 2 * b, time_dim)])
        self.upsample1 = nn.ConvTranspose2d(
            2 * b, b, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResBlock(2 * b, b, time_dim), ResBlock(b, b, time_dim)])

        self.out_norm = nn.GroupNorm(_group_count(b), b)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(b, seq_len, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _run(blocks, x, c):
        for block in blocks:
            x = block(x, c)
        return x

    def forward(self, x, t, cond=None):
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("SpatialUNet input dimensions must be divisible by 4")
        c = self.t_mlp(timestep_embedding(t, self.in_conv.out_channels))
        if cond is not None:
            if cond.shape != c.shape:
                raise ValueError(
                    f"condition shape {tuple(cond.shape)} != {tuple(c.shape)}")
            c = c + cond

        d1 = self._run(self.down1, self.in_conv(x), c)
        d2 = self._run(self.down2, self.downsample1(d1), c)
        h = self._run(self.mid, self.downsample2(d2), c)
        h = self._run(self.up2, torch.cat([self.upsample2(h), d2], dim=1), c)
        h = self._run(self.up1, torch.cat([self.upsample1(h), d1], dim=1), c)
        return self.out_conv(self.out_act(self.out_norm(h)))
