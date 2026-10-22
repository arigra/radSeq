"""Vehicle detector for RADIal-grid range-Doppler sequences (512 x 256).

The step 3 comparison scores every arm with this one detector: real data only,
real plus raw simulator data, and real plus DiT-generated data. It works frame
by frame, like RADIal's own baseline, so no arm can gain from reading a
neighbouring frame.

The Doppler axis is circular (unshifted FFT order), so both the training
targets and the convolutions wrap around it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def scene_heatmap_targets(traj, present, n_range=512, n_doppler=256, sigma=2.0):
    """Gaussian blobs at each present vehicle: (B, L, n_range, n_doppler).

    Doppler distance wraps; a vehicle near bin 255 must light up bin 0 as well,
    or the detector is taught a seam the radar does not have.
    """
    B, M, L, _ = traj.shape
    device = traj.device
    r = torch.arange(n_range, device=device, dtype=torch.float).view(1, 1, 1, n_range, 1)
    d = torch.arange(n_doppler, device=device, dtype=torch.float).view(1, 1, 1, 1, n_doppler)
    # Centres are rounded to a cell: the CenterNet focal loss counts a cell as
    # a vehicle only if its target is exactly 1.0, and RADIal ranges are
    # fractional. Unrounded, 596 vehicles produced a single positive cell.
    tr = torch.round(traj[..., 0].float()).view(B, M, L, 1, 1)
    td = torch.remainder(torch.round(traj[..., 1].float()), n_doppler).view(B, M, L, 1, 1)
    dd = torch.remainder(d - td + n_doppler / 2, n_doppler) - n_doppler / 2
    blobs = torch.exp(-((r - tr) ** 2 + dd ** 2) / (2 * sigma ** 2))
    blobs = blobs * present.to(device).view(B, M, L, 1, 1).float()
    return blobs.amax(dim=1)


def _conv(cin, cout):
    """3x3 convolution that wraps in Doppler and pads normally in range."""
    return nn.Sequential(
        _WrapPad(), nn.Conv2d(cin, cout, 3, padding=0), nn.GroupNorm(1, cout), nn.SiLU(),
        _WrapPad(), nn.Conv2d(cout, cout, 3, padding=0), nn.GroupNorm(1, cout), nn.SiLU())


class _WrapPad(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")     # Doppler: circular
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # range: replicate


class RadialVehicleDetector(nn.Module):
    """U-Net over one frame; sequences are folded into the batch."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = _conv(1, w), _conv(w, 2 * w), _conv(2 * w, 4 * w)
        self.mid = _conv(4 * w, 4 * w)
        self.dec3, self.dec2, self.dec1 = _conv(8 * w, 2 * w), _conv(4 * w, w), _conv(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)
        self.head.bias.data.fill_(-4.0)                 # rare positives

    def forward(self, x):
        b, l = x.shape[:2]
        h = x.reshape(b * l, 1, *x.shape[2:])
        e1 = self.enc1(h)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([up(m, e3), e3], 1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1).reshape(b, l, *x.shape[2:])


def average_precision_wrapped(detections, targets, n_doppler, radius=2.0):
    """All-point interpolated AP with a circular Doppler axis.

    src.detector.average_precision measures plain Euclidean distance, which
    puts bin 255 and bin 1 nearly a full axis apart. Many RADIal vehicles sit
    at that edge, so every arm would be scored against phantom misses.
    """
    scored, n_true = [], 0
    for frame_dets, tgt in zip(detections, targets):
        n_true += len(tgt)
        used = torch.zeros(len(tgt), dtype=torch.bool)
        for score, r, d in sorted(frame_dets, key=lambda t: -t[0]):
            hit = -1
            if len(tgt):
                dr = tgt[:, 0] - r
                dd = torch.remainder(tgt[:, 1] - d + n_doppler / 2, n_doppler) - n_doppler / 2
                dist = torch.sqrt(dr ** 2 + dd ** 2)
                dist[used] = float("inf")
                best = int(torch.argmin(dist))
                if float(dist[best]) <= radius:
                    hit = best
            if hit >= 0:
                used[hit] = True
            scored.append((score, hit >= 0))
    if n_true == 0:
        return float("nan")
    scored.sort(key=lambda t: -t[0])
    tp = fp = 0
    points = []
    for _, is_tp in scored:
        tp, fp = tp + int(is_tp), fp + int(not is_tp)
        points.append((tp / n_true, tp / (tp + fp)))
    # precision envelope, then sum over recall steps (all-point interpolation)
    envelope, best = [], 0.0
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    ap, prev_recall = 0.0, 0.0
    for recall, precision in reversed(envelope):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


class RadialSequenceDetector(RadialVehicleDetector):
    """The same U-Net, reading all L frames of a sequence at once as channels
    and predicting every frame's heatmap.

    The per-frame detector cannot tell a generator that models time from one
    that does not; this one can use persistence (a vehicle stays, clutter
    flickers), so it gains from training data only if the data's temporal
    behaviour is real.
    """

    def __init__(self, width=32, seq_len=8):
        super().__init__(width)
        self.enc1 = _conv(seq_len, width)
        self.head = nn.Conv2d(width, seq_len, 1)
        self.head.bias.data.fill_(-4.0)

    def forward(self, x):
        h = x
        e1 = self.enc1(h)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([up(m, e3), e3], 1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1)
