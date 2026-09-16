"""Does a generated sequence contain the targets it was asked for?

Uses the evaluator's own detector and linker (src/eval/metrics): peaks at
least 12 dB above the frame median, the 5 strongest per frame, linked across
frames within 3 bins.
"""
import torch

from src.eval.metrics import detect_peaks, filter_tracks, link_tracks


def trajectory_adherence(x_db, traj, n_targets, radius=2.0, max_peaks=5, min_track_len=8):
    """x_db (B, L, N, K) dB maps; traj (B, M, L, 2) requested bins; n_targets (B,).

    hit_rate: fraction of requested (target, frame) pairs with a detected peak
    within `radius` bins.
    unrequested_lasting_tracks_per_seq: linked tracks of >= min_track_len frames
    whose peaks are within `radius` of a requested target on fewer than half
    of their frames.
    """
    hits = requested = unrequested = 0
    B, L = x_db.shape[:2]
    for i in range(B):
        targets = traj[i, :int(n_targets[i])].float()                    # (m, L, 2)
        peaks = [detect_peaks(x_db[i, l], max_peaks=max_peaks) for l in range(L)]
        for l in range(L):
            for m in range(len(targets)):
                requested += 1
                if len(peaks[l]) and float((peaks[l] - targets[m, l]).norm(dim=1).min()) <= radius:
                    hits += 1
        for track in filter_tracks(link_tracks(peaks), min_track_len):
            near = sum(1 for l, pos in track if len(targets) and
                       float((targets[:, l] - torch.tensor(pos)).norm(dim=1).min()) <= radius)
            unrequested += int(near < len(track) / 2)
    return {"hit_rate": hits / max(requested, 1),
            "unrequested_lasting_tracks_per_seq": unrequested / B}
