"""Is the RADIal-fine-tuned generator copying its real training sequences?

Asks the generator for the label sets it trained on, and for held-out label
sets it never saw, then measures how close each output lands to the real
sequence that carries those labels:

  ratio = median RMS(training labels) / median RMS(held-out labels)

Well below 1 means the generator reproduces sequences it memorised rather than
generating from the labels. Unrelated real pairs give the scale of "no
relationship at all".

The RMS is measured on the vehicle windows (+/-6 range, +/-3 Doppler bins
around every labelled vehicle in every frame), which is what a detector learns
from; it decides the verdict. The whole-map RMS, which the step 4 check used,
is dominated by background and is kept only for reference.

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
WIN_R, WIN_D = 6, 3


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)))


def vehicle_cells(traj, present, shape):
    """Boolean mask of the vehicle windows, (L, n_range, n_doppler)."""
    mask = np.zeros(shape, bool)
    for m in range(traj.shape[0]):
        for t in range(traj.shape[1]):
            if not present[m, t]:
                continue
            r, d = int(round(float(traj[m, t, 0]))), int(round(float(traj[m, t, 1])))
            rs = slice(max(r - WIN_R, 0), min(r + WIN_R + 1, shape[1]))
            mask[t, rs, [(d + k) % shape[2] for k in range(-WIN_D, WIN_D + 1)]] = True
    return mask


def vehicle_rms(a, b, mask):
    if not mask.any():
        return float("nan")
    return rms(np.asarray(a, np.float32)[mask], np.asarray(b, np.float32)[mask])


def generated_distances(model, cfg, ds, n, device, steps, guidance, eta, batch=4):
    whole, vehicle = [], []
    for s in range(0, n, batch):
        idx = list(range(s, min(s + batch, n)))
        labels = {"traj": ds.traj[idx], "present": ds.present[idx]}
        x = sample_trajectory_conditioned(model, cfg, labels, device, steps=steps,
                                          guidance=guidance, seed=7 + s, eta=eta).numpy()
        for k, i in enumerate(idx):
            real = np.asarray(ds.x[i], np.float32)
            whole.append(rms(x[k], real))
            vehicle.append(vehicle_rms(x[k], real,
                                       vehicle_cells(ds.traj[i], ds.present[i], real.shape)))
        print(f"  {idx[-1] + 1}/{n}", flush=True)
    return whole, vehicle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/radial_seq")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_trajectory_conditioned(a.ckpt, device, weights=a.weights)
    train = SceneSequenceDataset(a.cache, "train", normalise=False)
    held = SceneSequenceDataset(a.cache, "val", normalise=False)
    n_train, n_held = min(a.n, len(train)), min(a.n, len(held))

    print("training labels:", flush=True)
    w_train, v_train = generated_distances(model, cfg, train, n_train, device, a.steps,
                                           a.guidance, a.eta)
    print("held-out labels:", flush=True)
    w_held, v_held = generated_distances(model, cfg, held, n_held, device, a.steps,
                                         a.guidance, a.eta)
    pairs = [rms(np.asarray(train.x[i], np.float32), np.asarray(train.x[i + 1], np.float32))
             for i in range(0, min(2 * n_train, len(train) - 1), 2)]

    ratio = float(np.nanmedian(v_train) / np.nanmedian(v_held))
    report = {"checkpoint": a.ckpt, "n_train_labels": n_train, "n_heldout_labels": n_held,
              "guidance": a.guidance, "steps": a.steps, "eta": a.eta, "weights": a.weights,
              "median_rms_vehicle_training_labels": float(np.nanmedian(v_train)),
              "median_rms_vehicle_heldout_labels": float(np.nanmedian(v_held)),
              "ratio_vehicle_training_over_heldout": ratio,
              "median_rms_training_labels": float(np.median(w_train)),
              "median_rms_heldout_labels": float(np.median(w_held)),
              "median_rms_unrelated_real_pairs": float(np.median(pairs)),
              "ratio_training_over_heldout": float(np.median(w_train) / np.median(w_held)),
              "memorising": bool(ratio < MEMORISING_BELOW)}
    open(a.out, "w").write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
