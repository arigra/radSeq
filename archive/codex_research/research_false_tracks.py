"""Do matched cell false-alarm rates imply matched false-track rates?

Run from repository root: python -m archive.codex_research.research_false_tracks
This is a target-free synthetic diagnostic, not a trained generator result.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from archive.codex_research.research_tail_risk import scores
from src.simulator import TemporalRadarSimulator


def false_tracks(cube, threshold):
    """Count 3-frame paths allowing a one-cell step in range and Doppler."""
    detection = cube > threshold
    linked2 = detection[1] & maximum_filter(detection[0], size=3)
    starts = 0
    for frame in range(2, len(cube)):
        linked3 = detection[frame] & maximum_filter(linked2, size=3)
        starts += int(linked3.sum())
        linked2 = detection[frame] & maximum_filter(detection[frame - 1], size=3)
    return starts


def run(train_n=8, test_n=12, pfa=0.01):
    sim = TemporalRadarSimulator(seq_len=8)
    rows = []
    for rho in (0.1, 0.5, 0.9):
        for nu in (0.1, 1.3):
            train = scores(sim, rho, nu, train_n, 101)
            test = scores(sim, rho, nu, test_n, 202).reshape(test_n, sim.L, 56, 56)
            threshold = float(np.quantile(train, 1 - pfa))
            observed = float(np.mean(test > threshold))
            actual = sum(false_tracks(cube, threshold) for cube in test)
            # Break temporal dependence while preserving every score frame.
            rng = np.random.default_rng(319)
            shuffled = test.copy()
            for frame in range(sim.L):
                shuffled[:, frame] = test[rng.permutation(test_n), frame]
            null = sum(false_tracks(cube, threshold) for cube in shuffled)
            row = {"rho": rho, "nu": nu, "target_pfa": pfa,
                   "observed_pfa": observed, "false_3frame_paths": actual,
                   "frame_shuffled_paths": null, "test_sequences": test_n}
            rows.append(row)
            print(row, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--out", default="archive/codex_research/results/research_false_tracks.json")
    args = parser.parse_args()
    result = run(args.train_n, args.test_n, args.pfa)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
