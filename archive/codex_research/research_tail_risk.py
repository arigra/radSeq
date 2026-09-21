"""Synthetic-only screen of extreme-tail CFAR calibration.

Run with ``python -m archive.codex_research.research_tail_risk``. Compare a Gaussian score
model, empirical quantiles, and a peaks-over-threshold
model on independent clutter realizations. This evaluates a necessary component
of detector-risk generation, not real-data transfer or a new radar generator.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import genpareto, norm

from src.simulator import TemporalRadarSimulator, create_rd_map


def scores(sim, rho, nu, n, seed):
    torch.manual_seed(seed)
    out = []
    for _ in range(n):
        clutter = sim._clutter_frames(rho, nu)
        noise = torch.randn_like(clutter) * math.sqrt(sim.sigma2)
        rd = torch.stack([create_rd_map(frame) for frame in clutter + noise])
        power = rd.abs().square().float()
        local = F.avg_pool2d(power, 9, stride=1, padding=4)
        ratio = power / local.clamp_min(1e-9)
        out.append(ratio[:, 4:-4, 4:-4].flatten().numpy())
    return np.concatenate(out)


def thresholds(train, pfa):
    empirical = float(np.quantile(train, 1 - pfa))
    gaussian = float(train.mean() + train.std() * norm.isf(pfa))
    u = float(np.quantile(train, 0.95))
    excess = train[train > u] - u
    shape, _, scale = genpareto.fit(excess, floc=0)
    pot = float(u + genpareto.isf(pfa / 0.05, shape, loc=0, scale=scale))
    return {"gaussian": gaussian, "empirical": empirical, "pot": pot}


def run(train_n=6, test_n=12):
    sim = TemporalRadarSimulator(seq_len=8)
    rows = []
    for rho, nu in [(0.3, 1.3), (0.7, 0.5), (0.9, 0.1)]:
        train = scores(sim, rho, nu, train_n, 101)
        test = scores(sim, rho, nu, test_n, 202)
        for pfa in (1e-3, 1e-4):
            t = thresholds(train, pfa)
            row = {"rho": rho, "nu": nu, "nominal_pfa": pfa,
                   "train_cells": len(train), "test_cells": len(test),
                   "observed_pfa": {k: float(np.mean(test > v)) for k, v in t.items()}}
            rows.append(row)
            print(row, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=6)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--out", default="archive/codex_research/results/research_tail_risk.json")
    args = parser.parse_args()
    result = run(args.train_n, args.test_n)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
