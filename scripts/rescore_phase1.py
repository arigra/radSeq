"""Honest Phase-1 exit check under the corrected metrics.

The original scoring averaged kinematic metrics over tracks that were only
19.9% real targets, so it described clutter. This rescoring:

  1. validates the detector settings on real val data against ground truth,
  2. computes the real-vs-real reference at the plan's protocol
     (evaluate_sequences(xs[:32], xs[32:]), plan Task 10 Step 7),
  3. scores checkpoints/phase1_wandb/best.pt the same way,
  4. evaluates the spec's Phase-1 exit criteria on the result.

Reports old (max_peaks=8, unfiltered) alongside new for comparability.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import (detect_peaks, evaluate_sequences, filter_tracks,
                              link_tracks)


def load_val_items(cache_dir, limit):
    items = []
    for f in sorted(Path(cache_dir).glob("val_*.pt")):
        for it in torch.load(f):
            items.append(it)
            if len(items) >= limit:
                return items
    return items


def detector_validity(items, max_peaks, min_len, tol=2.0):
    """Precision/recall of filtered tracks against ground-truth targets."""
    n_tracks = n_hit = n_targets = n_found = 0
    for it in items:
        m = int(it["n_targets"])
        traj = it["traj"][:m]
        tracks = filter_tracks(
            link_tracks([detect_peaks(f, max_peaks=max_peaks) for f in it["x"]]),
            min_len)
        n_tracks += len(tracks)
        matched = set()
        for tr in tracks:
            dists = [[float((traj[i, l] - torch.tensor(p)).pow(2).sum().sqrt())
                      for l, p in tr] for i in range(m)]
            meds = [float(np.median(d)) for d in dists]
            best = int(np.argmin(meds))
            if meds[best] <= tol:
                n_hit += 1
                matched.add(best)
        n_targets += m
        n_found += len(matched)
    return {"tracks_per_seq": round(n_tracks / len(items), 2),
            "true_targets_per_seq": round(n_targets / len(items), 2),
            "track_precision": round(n_hit / n_tracks, 4) if n_tracks else None,
            "target_recall": round(n_found / n_targets, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase1_wandb/best.pt")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n", type=int, default=32, help="per side, per the plan")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="samples/phase1_rescore.json")
    args = ap.parse_args()

    report = {"config": vars(args)}
    items = load_val_items(args.cache, 2 * args.n)
    real = torch.stack([it["x"] for it in items])
    ref_a, ref_b = real[:args.n], real[args.n:]

    # 1. detector validity, old settings vs new
    report["detector_validity"] = {
        "old_max_peaks8_no_filter": detector_validity(items, 8, 1),
        "new_max_peaks5_min_len8": detector_validity(items, 5, 8),
    }
    print("detector validity:")
    print(json.dumps(report["detector_validity"], indent=2))

    # 2. real-vs-real reference
    report["reference_real_vs_real"] = {
        "new": evaluate_sequences(ref_a, ref_b),
        "old": evaluate_sequences(ref_a, ref_b, max_peaks=8, min_track_len=1),
    }

    # 3. generated
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
    print(f"sampling {args.n} sequences...")
    gen = (diff.ddim_sample(model, (args.n, cfg["data"]["seq_len"], 64, 64),
                            device, steps=args.steps).cpu()
           * stats["std"] + stats["mean"])
    report["generated"] = {
        "new": evaluate_sequences(gen, ref_b),
        "old": evaluate_sequences(gen, ref_b, max_peaks=8, min_track_len=1),
    }
    report["generated_ckpt"] = {"epoch": ckpt.get("epoch"), "step": ckpt.get("step")}

    # 4. exit criteria (spec Phase 1)
    ref, g = report["reference_real_vs_real"]["new"], report["generated"]["new"]
    vc = g["velocity_consistency"]
    report["exit_criteria"] = {
        "persistence_within_0.2_of_reference": {
            "reference": round(ref["persistence"], 4),
            "generated": round(g["persistence"], 4),
            "abs_diff": round(abs(ref["persistence"] - g["persistence"]), 4),
            "pass": abs(ref["persistence"] - g["persistence"]) <= 0.2},
        "velocity_consistency_finite": {
            "value": vc, "pass": bool(vc == vc and abs(vc) != float("inf"))},
        "target_track_count_matches_reference": {
            "reference": round(ref["n_target_tracks_per_seq"], 2),
            "generated": round(g["n_target_tracks_per_seq"], 2),
            "note": "not a spec criterion; the clearest target-level signal"},
    }

    print("\n=== reference (real vs real) ===")
    print(json.dumps(report["reference_real_vs_real"], indent=2))
    print("\n=== generated (best.pt) ===")
    print(json.dumps(report["generated"], indent=2))
    print("\n=== exit criteria ===")
    print(json.dumps(report["exit_criteria"], indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
