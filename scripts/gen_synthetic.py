"""Render labelled synthetic sequences with a trajectory-conditioned DiT (step 2).

Requests are the labels of the first N training sequences (the same N the
generator was trained on), each rendered `repeats` times with different
sampling seeds. Output: <out>/part_<r>.pt with x (dB, float16), traj,
n_targets, cls. Existing parts are skipped, so the script is resumable.
"""
import argparse
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.sample import load_trajectory_conditioned, resolve_cache_dir, sample_trajectory_conditioned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-labels", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="data/synthetic/cond_traj_n2000")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_trajectory_conditioned(args.ckpt, device, weights="ema")
    items = RadarSequenceDataset(resolve_cache_dir(cfg["data"]["cache_dir"]), "train").items[:args.n_labels]
    labels = {key: torch.stack([it[key] for it in items]) for key in ("traj", "n_targets", "cls")}
    del items
    for r in range(args.repeats):
        path = out / f"part_{r}.pt"
        if path.exists():
            print(f"skip {path}", flush=True)
            continue
        maps = []
        for s in range(0, args.n_labels, args.batch):
            chunk = {k: v[s:s + args.batch] for k, v in labels.items()}
            maps.append(sample_trajectory_conditioned(
                model, cfg, chunk, device, steps=args.steps, guidance=args.guidance,
                seed=100_000 * (r + 1) + s).half())
        tmp = path.with_suffix(".tmp")
        torch.save({"x": torch.cat(maps), **labels, "repeat": r, "guidance": args.guidance,
                    "ckpt": args.ckpt}, tmp)
        tmp.replace(path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
