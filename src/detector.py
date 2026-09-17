"""Per-frame target detector for research plan step 2: does synthetic data help?

A small U-Net maps one normalised RD frame (64x64) to a heatmap of target
centres. Labels are Gaussian blobs (sigma 1 bin, peak 1) at each target's
rounded bin, trained with the CenterNet focal loss. Detections are 3x3 local
maxima of the sigmoid heatmap; average precision matches them to true targets
within 2 bins, each true target at most once.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blobs(traj, n_targets, size, sigma):
    """Gaussian blob (peak 1) at each target's rounded bin: (B, M, L, size, size), and the (B, M) real-target mask."""
    B, M, L, _ = traj.shape
    device = traj.device
    centres = torch.round(traj.float())
    r = torch.arange(size, device=device, dtype=torch.float32).view(1, 1, 1, size, 1)
    d = torch.arange(size, device=device, dtype=torch.float32).view(1, 1, 1, 1, size)
    blobs = torch.exp(-((r - centres[..., 0, None, None]) ** 2
                        + (d - centres[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    real = (torch.arange(M, device=device)[None] < n_targets.to(device)[:, None]).float()
    return blobs, real


def heatmap_targets(traj, n_targets, size=64, sigma=1.0):
    """traj (B, M, L, 2) bins; n_targets (B,) -> (B, L, size, size), peak 1 at rounded bins."""
    blobs, real = _blobs(traj, n_targets, size, sigma)
    B, M = real.shape
    return (blobs * real.view(B, M, 1, 1, 1)).amax(dim=1)


def class_heatmap_targets(traj, n_targets, cls, n_classes=3, size=64, sigma=1.0):
    """-> (B, n_classes, L, size, size): each target's blob in its class channel only."""
    blobs, real = _blobs(traj, n_targets, size, sigma)
    B, M = real.shape
    onehot = F.one_hot(cls.to(traj.device).long(), n_classes).float() * real[..., None]
    return (blobs[:, :, None] * onehot.view(B, M, n_classes, 1, 1, 1)).amax(dim=1)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class HeatmapDetector(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.enc1 = _block(1, width)
        self.enc2 = _block(width, 2 * width)
        self.enc3 = _block(2 * width, 4 * width)
        self.dec2 = _block(4 * width + 2 * width, 2 * width)
        self.dec1 = _block(2 * width + width, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -2.19)       # sigmoid ~0.1 at start (CenterNet)

    def forward(self, x):
        """x (B, 64, 64) normalised frames -> (B, 64, 64) logits."""
        e1 = self.enc1(x[:, None])
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)[:, 0]


class SequenceClassDetector(nn.Module):
    """All 16 frames in, one heatmap per class per frame out.

    (B, 16, 64, 64) normalised maps -> (B, n_classes, 16, 64, 64) logits. Seeing the
    whole sequence lets it use frame-to-frame amplitude fluctuation (Swerling-1)
    and range extent (extended targets).
    """

    def __init__(self, frames=16, n_classes=3, width=48):
        super().__init__()
        self.frames, self.n_classes = frames, n_classes
        self.enc1 = _block(frames, width)
        self.enc2 = _block(width, 2 * width)
        self.enc3 = _block(2 * width, 4 * width)
        self.dec2 = _block(4 * width + 2 * width, 2 * width)
        self.dec1 = _block(2 * width + width, width)
        self.head = nn.Conv2d(width, n_classes * frames, 1)
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1).view(x.shape[0], self.n_classes, self.frames, *x.shape[-2:])


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet penalty-reduced focal loss, normalised by the number of target centres."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp_min(1)


def detect(prob, threshold=0.05, top_k=10):
    """prob (F, H, W) in [0, 1] -> per frame list of (score, range, doppler)."""
    peaks = (prob == F.max_pool2d(prob[:, None], 3, 1, 1)[:, 0]) & (prob > threshold)
    out = []
    for f in range(prob.shape[0]):
        rs, ds = torch.nonzero(peaks[f], as_tuple=True)
        scores = prob[f, rs, ds]
        order = torch.argsort(scores, descending=True)[:top_k]
        out.append([(float(scores[i]), float(rs[i]), float(ds[i])) for i in order])
    return out


def average_precision(detections, targets, radius=2.0):
    """detections: per frame list of (score, r, d); targets: per frame (m, 2) true bins.

    All-point interpolated AP over all frames; a true target is matched at most once.
    """
    scored, n_true = [], 0
    for frame_dets, tgt in zip(detections, targets):
        n_true += len(tgt)
        used = torch.zeros(len(tgt), dtype=torch.bool)
        for score, r, d in sorted(frame_dets, reverse=True):
            hit = False
            if len(tgt):
                dist = (tgt.float() - torch.tensor([r, d])).norm(dim=1)
                dist[used] = float("inf")
                j = int(dist.argmin())
                if float(dist[j]) <= radius:
                    used[j] = True
                    hit = True
            scored.append((score, hit))
    if n_true == 0 or not scored:
        return 0.0
    scored.sort(key=lambda s: -s[0])
    hits = torch.tensor([float(h) for _, h in scored])
    tp = torch.cumsum(hits, 0)
    precision = tp / torch.arange(1, len(hits) + 1)
    recall = tp / n_true
    envelope = torch.flip(torch.cummax(torch.flip(precision, [0]), 0).values, [0])
    previous = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - previous) * envelope).sum())
