"""Is the RADIal-fine-tuned generator copying its 387 training sequences?

Asks the generator for the label sets it trained on, and for held-out label
sets it never saw, then measures how close each output lands to the real
sequence that carries those labels:

  ratio = median RMS(training labels) / median RMS(held-out labels)

Well below 1 means the generator reproduces sequences it memorised rather than
generating from the labels. Unrelated real pairs give the scale of "no
relationship at all".

This matters for reading the detector arms: a memorising generator adds little
beyond the real data the detector already has, which would explain a weak
result without the method itself being wrong.
"""
import argparse
import json

import numpy as np
import torch

from src.sample import load_trajectory_conditioned, sample_trajectory_conditioned
from src.scene_data import SceneSequenceDataset

MEMORISING_BELOW = 0.8


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)))


def generated_distances(model, cfg, ds, n, device, steps, guidance, batch=4):
    out = []
    for s in range(0, n, batch):
        idx = range(s, min(s + batch, n))
        labels = {"traj": ds.traj[list(idx)], "present": ds.present[list(idx)]}
        x = sample_trajectory_conditioned(model, cfg, labels, device, steps=steps,
                                          guidance=guidance, seed=7 + s).numpy()
        for k, i in enumerate(idx):
            out.append(rms(x[k], np.asarray(ds.x[i], np.float32)))
        print(f"  {min(s + batch, n)}/{n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/radial_seq")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_trajectory_conditioned(a.ckpt, device, weights="ema")
    train = SceneSequenceDataset(a.cache, "train", normalise=False)
    held = SceneSequenceDataset(a.cache, "val", normalise=False)
    n_held = min(a.n, len(held))

    print("training labels:", flush=True)
    d_train = generated_distances(model, cfg, train, a.n, device, a.steps, a.guidance)
    print("held-out labels:", flush=True)
    d_held = generated_distances(model, cfg, held, n_held, device, a.steps, a.guidance)
    pairs = [rms(np.asarray(train.x[i], np.float32), np.asarray(train.x[i + 1], np.float32))
             for i in range(0, min(2 * a.n, len(train) - 1), 2)]

    ratio = float(np.median(d_train) / np.median(d_held))
    report = {"checkpoint": a.ckpt, "n_train_labels": a.n, "n_heldout_labels": n_held,
              "guidance": a.guidance,
              "median_rms_training_labels": float(np.median(d_train)),
              "median_rms_heldout_labels": float(np.median(d_held)),
              "median_rms_unrelated_real_pairs": float(np.median(pairs)),
              "ratio_training_over_heldout": ratio,
              "memorising": bool(ratio < MEMORISING_BELOW)}
    open(a.out, "w").write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
