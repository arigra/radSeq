"""Check temporal detection-path dependence in public measured RD sequences.

Uses Rad-R training_cache.h5, grouped by capture. These paths are NOT labelled
false tracks: real objects may be present. Per-frame rank thresholds fix the
cell occupancy, isolating temporal relationships from marginal calibration.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import maximum_filter


def paths(detection):
    linked = detection[1] & maximum_filter(detection[0], size=3)
    total = 0
    for t in range(2, len(detection)):
        total += int((detection[t] & maximum_filter(linked, size=3)).sum())
        linked = detection[t] & maximum_filter(detection[t - 1], size=3)
    return total


def run(path, occupancy=0.005, seed=201):
    rng = np.random.default_rng(seed)
    rows = []
    with h5py.File(path) as f:
        capture = np.asarray(f["capture"]).astype(str)
        frames = np.asarray(f["frame_idx"])
        for name in np.unique(capture):
            indices = np.flatnonzero(capture == name)
            indices = indices[np.argsort(frames[indices])]
            # Read each frame once; keep full resolution for 1-cell paths.
            maps = np.stack([f["rd_map"][int(i)] for i in indices])
            if not np.isfinite(maps).all():
                raise ValueError(f"non-finite RD map in {name}")
            cut = np.quantile(maps.reshape(len(maps), -1), 1 - occupancy, axis=1)
            det = maps > cut[:, None, None]
            real = paths(det)
            nulls = [paths(det[rng.permutation(len(det))]) for _ in range(20)]
            row = {"capture": name, "frames": len(det), "frame_idx_gap":
                   float(np.median(np.diff(frames[indices]))),
                   "occupancy": float(det.mean()), "ordered_paths": real,
                   "shuffled_mean": float(np.mean(nulls)),
                   "shuffled_min": min(nulls), "shuffled_max": max(nulls)}
            rows.append(row)
            print(row, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/measured/radr/training_cache.h5")
    parser.add_argument("--out", default="archive/codex_research/results/research_real_track_gap.json")
    args = parser.parse_args()
    rows = run(args.input)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
