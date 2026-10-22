"""Render labelled RADIal-grid sequences from a fine-tuned scene DiT.

Conditioned on the vehicle labels of a cache's sequences: the real training
sequences (the generated set then carries the same labels as the real data it
augments) or, for new scene layouts, simulator sequences (`--cache
data/scene_<variant> --limit N`). Output: <out>/part_<r>.pt with x (dB,
float16), traj, present and the sampler settings. Existing parts are skipped,
so the script is resumable.
"""
import argparse
from pathlib import Path

import torch

from src.sample import load_trajectory_conditioned, sample_trajectory_conditioned
from src.scene_data import SceneSequenceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/radial_seq", help="labels come from its split")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, help="use only the first N label sets")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_trajectory_conditioned(a.ckpt, device, weights=a.weights)
    ds = SceneSequenceDataset(a.cache, a.split, normalise=False)
    n = len(ds.traj) if a.limit is None else min(a.limit, len(ds.traj))
    labels = {"traj": ds.traj[:n], "present": ds.present[:n]}
    for r in range(a.repeats):
        path = out / f"part_{r}.pt"
        if path.exists():
            print(f"skip {path}", flush=True)
            continue
        maps = []
        for s in range(0, n, a.batch):
            chunk = {k: v[s:s + a.batch] for k, v in labels.items()}
            maps.append(sample_trajectory_conditioned(
                model, cfg, chunk, device, steps=a.steps, guidance=a.guidance,
                seed=100_000 * (r + 1) + s, eta=a.eta).half())
            print(f"part {r}: {min(s + a.batch, n)}/{n}", flush=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"x": torch.cat(maps), **labels, "repeat": r,
                    "guidance": a.guidance, "steps": a.steps, "eta": a.eta,
                    "weights": a.weights, "labels_from": f"{a.cache}:{a.split}",
                    "ckpt": a.ckpt}, tmp)
        tmp.replace(path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
