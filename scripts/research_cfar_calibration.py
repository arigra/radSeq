"""Pilot: can detector statistics recover clutter correlation and texture?

Run from the repository root. The conventional raw-map moment baseline uses
mean, std and lag-one correlation; the candidate uses soft CA-CFAR occupancy
and detection correlation at three operating points. Both see the same draws.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from src.research_losses import soft_cfar_map
from src.simulator import TemporalRadarSimulator, create_rd_map


def draw_clutter(sim, rho, nu, seed, n):
    torch.manual_seed(seed)
    maps = []
    for _ in range(n):
        clutter = sim._clutter_frames(rho, nu)
        noise = torch.randn_like(clutter) * math.sqrt(sim.sigma2)
        rd = torch.stack([create_rd_map(frame) for frame in clutter + noise])
        maps.append(20 * torch.log10(rd.abs() + 1e-6))
    return torch.stack(maps)


def cfar_features(x):
    features = []
    for pfa in (1e-2, 1e-3, 1e-4):
        det = soft_cfar_map(x, pfa)
        mean = det.mean()
        lag = ((det[:, 1:] - mean) * (det[:, :-1] - mean)).mean()
        lag = lag / det.var().clamp_min(1e-8)
        features.extend((mean.log(), lag))
    return torch.stack(features)


def raw_features(x):
    mean, std = x.mean(), x.std()
    lag = ((x[:, 1:] - mean) * (x[:, :-1] - mean)).mean() / x.var()
    return torch.stack((mean, std, lag))


def gain_invariant_features(x):
    z = (x - x.mean()) / x.std()
    lag = (z[:, 1:] * z[:, :-1]).mean()
    return torch.stack((z.pow(3).mean(), z.pow(4).mean(), lag))


def select(reference, candidates, scale):
    distances = {key: (((value - reference) / scale) ** 2).mean().item()
                 for key, value in candidates.items()}
    return min(distances, key=distances.get), distances


def run(n=16, gain_shift=False):
    sim = TemporalRadarSimulator(seq_len=8)
    grid = [(rho, nu) for rho in (0.1, 0.3, 0.5, 0.7, 0.9)
            for nu in (0.1, 0.3, 0.5, 0.8, 1.3)]
    truths = [(0.7, 0.5), (0.3, 1.3), (0.9, 0.1)]
    rows = []
    for truth in truths:
        for seed, shift in ((11, 8), (22, -6), (33, 12)):
            reference = draw_clutter(sim, *truth, seed=seed, n=n)
            if gain_shift:
                reference = reference + shift
            ref_c, ref_r = cfar_features(reference), raw_features(reference)
            ref_s = gain_invariant_features(reference)
            cand_c, cand_r, cand_s = {}, {}, {}
            for candidate in grid:
                x = draw_clutter(sim, *candidate, seed=seed + 1, n=n)
                cand_c[candidate] = cfar_features(x)
                cand_r[candidate] = raw_features(x)
                cand_s[candidate] = gain_invariant_features(x)
            # Scale by the spread across parameter settings, fixed within an
            # experiment; prevents dB mean from overwhelming correlation.
            sc = torch.stack(list(cand_c.values())).std(0).clamp_min(1e-3)
            sr = torch.stack(list(cand_r.values())).std(0).clamp_min(1e-3)
            ss = torch.stack(list(cand_s.values())).std(0).clamp_min(1e-3)
            best_c, _ = select(ref_c, cand_c, sc)
            best_r, _ = select(ref_r, cand_r, sr)
            best_s, _ = select(ref_s, cand_s, ss)
            rows.append({"truth": truth, "seed": seed,
                         "gain_shift_db": shift if gain_shift else 0,
                         "cfar": best_c, "raw_moments": best_r,
                         "gain_invariant_moments": best_s})
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--gain-shift", action="store_true")
    parser.add_argument("--out", default="samples/research_cfar_calibration.json")
    args = parser.parse_args()
    result = run(args.n, gain_shift=args.gain_shift)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
