"""Physics metrics for generated RD sequences (spec §5).

Hard peak detection + greedy nearest-neighbor linking; metrics on the
resulting tracks quantify kinematic plausibility.
"""
import numpy as np
import torch
from scipy.ndimage import maximum_filter


def detect_peaks(frame, threshold_db=12.0, max_peaks=8):
    f = frame.detach().cpu().numpy()
    local_max = (maximum_filter(f, size=3) == f)
    thresh = np.median(f) + threshold_db
    mask = local_max & (f > thresh)
    rs, cs = np.nonzero(mask)
    if len(rs) == 0:
        return torch.zeros(0, 2)
    order = np.argsort(f[rs, cs])[::-1][:max_peaks]
    return torch.tensor(np.stack([rs[order], cs[order]], axis=1), dtype=torch.float)


def link_tracks(peaks_per_frame, gate=3.0):
    tracks = []          # each: list of (frame_idx, (r, d))
    for l, peaks in enumerate(peaks_per_frame):
        unused = list(range(len(peaks)))
        for tr in tracks:
            if tr[-1][0] != l - 1 or not unused:
                continue
            last = torch.tensor(tr[-1][1])
            d = torch.tensor([torch.dist(last, peaks[j]) for j in unused])
            j_best = int(d.argmin())
            if d[j_best] <= gate:
                tr.append((l, tuple(peaks[unused[j_best]].tolist())))
                unused.pop(j_best)
        for j in unused:
            tracks.append([(l, tuple(peaks[j].tolist()))])
    return tracks


def _positions(track):
    return torch.tensor([p for _, p in track])


def velocity_consistency(tracks):
    vals = []
    for tr in tracks:
        if len(tr) < 3:
            continue
        pos = _positions(tr)
        acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]
        vals.append(acc.pow(2).sum(dim=1).mean())
    return float(torch.stack(vals).mean()) if vals else float("nan")


def doppler_drift(tracks):
    vals = []
    for tr in tracks:
        if len(tr) < 2:
            continue
        dop = _positions(tr)[:, 1]
        vals.append((dop[1:] - dop[:-1]).abs().mean())
    return float(torch.stack(vals).mean()) if vals else float("nan")


def persistence(tracks, seq_len):
    if not tracks:
        return 0.0
    spans = torch.tensor([len(tr) for tr in tracks], dtype=torch.float)
    return float((spans >= 0.8 * seq_len).float().mean())


def marginal_l1(x_gen, x_real, bins=64):
    lo = min(x_gen.min().item(), x_real.min().item())
    hi = max(x_gen.max().item(), x_real.max().item())
    hg = torch.histc(x_gen.flatten(), bins=bins, min=lo, max=hi)
    hr = torch.histc(x_real.flatten(), bins=bins, min=lo, max=hi)
    return float((hg / hg.sum() - hr / hr.sum()).abs().sum())


def evaluate_sequences(x_gen, x_real, seq_len=16):
    vc, dd, ps, nt = [], [], [], []
    for seq in x_gen:
        tracks = link_tracks([detect_peaks(f) for f in seq])
        vc.append(velocity_consistency(tracks))
        dd.append(doppler_drift(tracks))
        ps.append(persistence(tracks, seq_len))
        nt.append(len(tracks))
    def _nanmean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")
    return {
        "velocity_consistency": _nanmean(vc),
        "doppler_drift": _nanmean(dd),
        "persistence": _nanmean(ps),
        "marginal_l1": marginal_l1(x_gen, x_real),
        "mean_tracks_per_seq": sum(nt) / len(nt),
    }


def velocity_adherence(x_gen, v0_cmd, frame_interval=0.5,
                       dv=0.2496006389776358, v_min=-7.987220447284345):
    """Pearson correlation between commanded initial velocity and the
    velocity implied by the longest track's mean Doppler coordinate."""
    measured = []
    for seq in x_gen:
        tracks = link_tracks([detect_peaks(f) for f in seq])
        if not tracks:
            measured.append(float("nan"))
            continue
        tr = max(tracks, key=len)
        dop = _positions(tr)[:, 1].mean()
        measured.append(v_min + dv * float(dop))
    m = torch.tensor(measured)
    ok = torch.isfinite(m)
    if ok.sum() < 3:
        return float("nan")
    a, b = m[ok], v0_cmd[ok].float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def cfar_detection_stats(x_seqs, Pfa=1e-3, num_train=4, num_guard=2):
    """Mean CA-CFAR detections per frame over (B, L, N, K) log-mag input."""
    from src.eval.cfar import ca_cfar_2d
    counts = []
    for seq in x_seqs:
        power = (10 ** (seq / 10)).detach().cpu().numpy()
        for frame in power:
            counts.append(ca_cfar_2d(frame, num_train, num_guard, Pfa).sum())
    return float(sum(counts) / len(counts))
