"""Is the small-data conditional DiT copying its training sequences? (research plan step 2)

Requests from the generator's own training labels: compare each generated
sequence with the training sequence carrying those labels (same targets, so
only clutter and noise can differ). Same for held-out labels and their
held-out sequences. A memorising generator reproduces training sequences much
more closely than unseen ones.
Pre-set: memorising if median distance (training labels) < 0.8 x median
distance (held-out labels). Distance: RMS difference of normalised maps.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.sample import load_trajectory_conditioned, resolve_cache_dir, sample_trajectory_conditioned


def rms_to_references(model, cfg, items, stats, device, guidance, seed, batch=32):
    dists = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        labels = {k: torch.stack([it[k] for it in chunk]) for k in ("traj", "n_targets", "cls")}
        gen = sample_trajectory_conditioned(model, cfg, labels, device, steps=30,
                                            guidance=guidance, seed=seed + s)
        real = torch.stack([it["x"] for it in chunk]).float()
        z = lambda x: (x - stats["mean"]) / stats["std"]
        dists += (z(gen) - z(real)).pow(2).flatten(1).mean(1).sqrt().tolist()
    return torch.tensor(dists)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--out", default="samples/memorization_n2000.json")
    ap.add_argument("--note")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_trajectory_conditioned(args.ckpt, device, weights="ema")
    cache = resolve_cache_dir(cfg["data"]["cache_dir"])
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    subset = cfg["data"].get("train_subset") or len(RadarSequenceDataset(cache, "train").items)
    train_items = RadarSequenceDataset(cache, "train").items[:min(args.n, subset)]
    val_items = RadarSequenceDataset(cache, "val").items[1000:1000 + args.n]
    d_train = rms_to_references(model, cfg, train_items, stats, device, args.guidance, seed=7000)
    d_val = rms_to_references(model, cfg, val_items, stats, device, args.guidance, seed=9000)
    real_pairs = torch.stack([it["x"] for it in val_items]).float()
    z = (real_pairs - stats["mean"]) / stats["std"]
    unrelated = (z - z.roll(1, 0)).pow(2).flatten(1).mean(1).sqrt()
    ratio = float(d_train.median() / d_val.median())
    report = {"checkpoint": args.ckpt, "n": args.n, "guidance": args.guidance,
              "median_rms_training_labels": float(d_train.median()),
              "median_rms_heldout_labels": float(d_val.median()),
              "median_rms_unrelated_real_pairs": float(unrelated.median()),
              "ratio_training_over_heldout": ratio, "memorising": ratio < 0.8}
    print(json.dumps(report, indent=2), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.note:
        with open(args.note, "a") as fh:
            fh.write(f"\n## Memorisation check (`{args.out}`)\n\n"
                     f"{args.n} requests each, EMA, DDIM 30, w={args.guidance}. RMS difference (normalised units) "
                     f"between a generated sequence and the real sequence with the same labels: training labels "
                     f"{report['median_rms_training_labels']:.3f}, held-out labels "
                     f"{report['median_rms_heldout_labels']:.3f} (unrelated real pairs "
                     f"{report['median_rms_unrelated_real_pairs']:.3f}). Ratio {ratio:.2f} (memorising if < 0.8).\n\n"
                     "**Verdict (pre-set rule): "
                     + ("the generator IS copying its training sequences.**" if report["memorising"] else
                        "not memorising: training sequences are reproduced no more closely than unseen ones.**")
                     + "\n")


if __name__ == "__main__":
    main()
