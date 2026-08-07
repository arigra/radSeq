"""Are the physics metrics measuring TARGETS, or clutter false alarms?

evaluate_sequences reports ~45 tracks/sequence on real data that contains only
1-5 real targets. If most detections are clutter, then velocity_consistency /
doppler_drift / persistence describe clutter statistics, and 'the backbone
learned target motion' would not follow from them.

Uses real val sequences, which carry ground-truth trajectories, to measure what
fraction of detected peaks and linked tracks correspond to an actual target.
Also compares bright-pixel spatial structure between real and generated, the
signature predicted by the spatially-independent architecture.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import detect_peaks, link_tracks


def load_val_items(cache_dir, limit):
    items = []
    for f in sorted(Path(cache_dir).glob("val_*.pt")):
        for item in torch.load(f):
            items.append(item)
            if len(items) >= limit:
                return items
    return items


def bright_structure(seqs, label, thresh_db=12.0):
    """Fraction of pixels above the detect_peaks threshold, and how many
    connected blobs they form. Spatially-independent generation should give
    a similar or larger bright fraction but far more, smaller, scattered blobs."""
    from scipy.ndimage import label as cc_label
    fracs, comps, sizes = [], [], []
    for seq in seqs:
        for f in seq:
            a = f.numpy()
            m = a > (np.median(a) + thresh_db)
            fracs.append(m.mean())
            lab, n = cc_label(m)
            comps.append(n)
            if n:
                sizes.append(np.bincount(lab.ravel())[1:].mean())
    return {"label": label,
            "bright_frac": round(float(np.mean(fracs)), 6),
            "blobs_per_frame": round(float(np.mean(comps)), 2),
            "mean_blob_px": round(float(np.mean(sizes)) if sizes else 0.0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--ckpt", default="checkpoints/phase1_wandb/best.pt")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="bins; a detection within tol of a GT target counts as a hit")
    ap.add_argument("--out", default="samples/diag_track_validity.json")
    args = ap.parse_args()

    report = {"config": vars(args)}
    items = load_val_items(args.cache, args.n)

    # ---- how many detections are actually targets? ----------------------
    tot_peaks = hit_peaks = 0
    tot_tracks = hit_tracks = 0
    n_targets_total = 0
    per_seq_tracks = []
    for item in items:
        seq = item["x"]                      # raw dB
        m = int(item["n_targets"])
        traj = item["traj"][:m]              # (m, L, 2) in (range_bin, doppler_bin)
        n_targets_total += m
        peaks_per_frame = [detect_peaks(f) for f in seq]
        for l, peaks in enumerate(peaks_per_frame):
            for p in peaks:
                tot_peaks += 1
                d = (traj[:, l] - p).pow(2).sum(dim=1).sqrt().min()
                if float(d) <= args.tol:
                    hit_peaks += 1
        tracks = link_tracks(peaks_per_frame)
        per_seq_tracks.append(len(tracks))
        for tr in tracks:
            tot_tracks += 1
            # a track counts as a target track if its first point is near a GT
            l0, (r0, d0) = tr[0]
            p0 = torch.tensor([r0, d0])
            if float((traj[:, l0] - p0).pow(2).sum(dim=1).sqrt().min()) <= args.tol:
                hit_tracks += 1

    report["real_detection_validity"] = {
        "sequences": len(items),
        "true_targets_total": n_targets_total,
        "mean_true_targets_per_seq": round(n_targets_total / len(items), 2),
        "detected_peaks_total": tot_peaks,
        "peaks_matching_a_target": hit_peaks,
        "peak_precision": round(hit_peaks / tot_peaks, 4) if tot_peaks else None,
        "tracks_total": tot_tracks,
        "tracks_starting_on_a_target": hit_tracks,
        "track_precision": round(hit_tracks / tot_tracks, 4) if tot_tracks else None,
        "mean_tracks_per_seq": round(float(np.mean(per_seq_tracks)), 2),
    }
    print(json.dumps(report["real_detection_validity"], indent=2))

    # ---- bright-pixel spatial structure, real vs generated ---------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.diffusion import GaussianDiffusion
    from src.train import build_model
    stats = torch.load(Path(args.cache) / "stats.pt")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    torch.manual_seed(1)
    gen = diff.ddim_sample(model, (16, cfg["data"]["seq_len"], 64, 64), device,
                           steps=50).cpu() * stats["std"] + stats["mean"]

    real_seqs = [it["x"] for it in items[:16]]
    report["bright_structure"] = {
        "real": bright_structure(real_seqs, "real"),
        "generated": bright_structure(list(gen), "generated"),
    }
    print(json.dumps(report["bright_structure"], indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
