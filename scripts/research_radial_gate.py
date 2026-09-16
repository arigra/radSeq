"""Gate: temporal background detection paths in measured RADIal sequences.

Pre-registered in docs/notes/2026-09-13-measured-temporal-gate.md.
Compares ordered background detections with a within-window frame
permutation that mimics a frame-wise, vehicle-conditioned generator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

RANGE_RES_M = 0.201  # matched to label/peak alignment on 200 labelled vehicles


def ca_cfar_ratio(power_db, num_train=4, num_guard=2):
    """Cell power over mean training-cell power, per frame, linear units."""
    power = 10 ** (power_db.astype(np.float64) / 10)
    outer, inner = 2 * (num_train + num_guard) + 1, 2 * num_guard + 1
    size = (1, outer, outer)
    total = uniform_filter(power, size=size, mode="nearest") * outer ** 2
    guard = uniform_filter(power, size=(1, inner, inner), mode="nearest") * inner ** 2
    return power / ((total - guard) / (outer ** 2 - inner ** 2))


def paths(det, k):
    size = (2 * k + 1, 2 * k + 1)
    linked = det[1] & maximum_filter(det[0], size=size)
    total = 0
    for t in range(2, len(det)):
        total += int((det[t] & maximum_filter(linked, size=size)).sum())
        linked = det[t] & maximum_filter(det[t - 1], size=size)
    return total


def vehicle_mask(vehicles, shape, margin=8):
    mask = np.zeros(shape, dtype=bool)
    for t, frame in enumerate(vehicles):
        for r_m, _, _ in frame:
            b = int(round(r_m / RANGE_RES_M))
            mask[t, max(b - margin, 0):b + margin + 1] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/measured/radial")
    parser.add_argument("--out", default="samples/research_radial_gate.json")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--perms", type=int, default=20)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
    rows = []
    for run in manifest:
        maps = np.load(Path(args.cache) / f"run_{run['run']:03d}.npy")
        for s in range(0, len(maps) - args.window + 1, args.window):
            ratio = ca_cfar_ratio(maps[s:s + args.window])
            mask = vehicle_mask(run["vehicles"][s:s + args.window], ratio.shape)
            for occ in (1e-3, 1e-2):
                det = ratio > np.quantile(ratio, 1 - occ)
                bg = det & ~mask
                for k in (1, 4):
                    ordered = paths(bg, k)
                    null = [paths(bg[rng.permutation(len(bg))], k)
                            for _ in range(args.perms)]
                    rows.append({"run": run["run"], "record": run["record"],
                                 "start": s, "occupancy": occ, "k": k,
                                 "background_detections": int(bg.sum()),
                                 "ordered": ordered,
                                 "permuted_mean": float(np.mean(null))})
        print(run["run"], flush=True)
    summary = {}
    for occ in (1e-3, 1e-2):
        for k in (1, 4):
            sel = [r for r in rows if r["occupancy"] == occ and r["k"] == k]
            ratio = np.array([(r["ordered"] + 1) / (r["permuted_mean"] + 1)
                              for r in sel])
            summary[f"occ={occ},k={k}"] = {
                "windows": len(sel), "median_ratio": float(np.median(ratio)),
                "q25": float(np.quantile(ratio, 0.25)),
                "q75": float(np.quantile(ratio, 0.75)),
                "frac_ratio_gt_1": float((ratio > 1).mean()),
                "ordered_total": int(sum(r["ordered"] for r in sel)),
                "permuted_total": float(sum(r["permuted_mean"] for r in sel))}
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows},
                                         indent=2))


if __name__ == "__main__":
    main()
