"""Fit the scene simulator's uncertain knobs to RADIal -- the "fitted" variant.

The study compares two simulators on purpose:
  engineer's  Scenario() as written: physics plus general road knowledge,
              never tuned against the recordings. The honest baseline.
  fitted      the same simulator with its uncertain knobs tuned to RADIal
              TRAINING recordings. The upper bound on what calibration buys.

Only knobs that are genuinely uncertain are fitted: the rough-surface
scattering law, scene density, ground return, the receiver filter shapes and
the link-budget anchor. The radar itself (grid, DDMA code, antennas) is
published and stays fixed.

Validation and test recordings are never read: fitting on them would leak the
data the whole comparison is evaluated on.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from src import radial
from src.scene_sim import Scenario, SceneSimulator

KNOBS = {   # name: (low, high) for the random search
    "surface_incidence_power": (0.0, 3.0),
    "structure_coverage_scale": (0.5, 2.0),
    "tree_density_scale": (0.5, 8.0),
    "ground_rcs_mean": (-35.0, -15.0),
    "highpass_corner_m": (1.0, 8.0),
    "lowpass_edge": (0.85, 0.97),
    "lowpass_order": (4.0, 16.0),
    "thermal_over_adc_db": (10.0, 30.0),
    "snr_offset_db": (-10.0, 10.0),
}
SCALE_DB = 5.0   # every statistic is in dB; errors are counted in units of 5 dB


def vehicle_prominence(maps, labels):
    out = []
    for m, frame in zip(maps, labels):
        for rb, db in frame:
            ri, di = int(round(rb)), int(round(db))
            if 2 <= ri < m.shape[0] - 2 and 2 <= di < m.shape[1] - 2:
                out.append(m[ri-2:ri+3, di-2:di+3].max() - np.median(m[ri-2:ri+3, :]))
    return float(np.median(out)) if out else float("nan")


def describe(maps, labels):
    stats = radial.background_stats(maps)
    stats["vehicle prominence"] = vehicle_prominence(maps, labels)
    return stats


def real_reference(n_frames):
    split = json.loads(Path("data/radial/split.json").read_text())
    recs = radial.recordings()
    by_frame = radial.read_labels()
    train = sorted(s for name in split["train"] for s in recs[name])
    chosen = train[::max(1, len(train) // n_frames)][:n_frames]
    maps = np.stack([radial.power_map(s) for s in chosen])
    labels = [radial.targets(s, by_frame) for s in chosen]
    return describe(maps, labels)


def scenario_from(knobs):
    k = dict(knobs)
    ground = k.pop("ground_rcs_mean")
    k["lowpass_order"] = int(round(k["lowpass_order"]))
    return Scenario(ground_rcs_dbsm=(ground, 6.0), **k)


def simulate(scenario, n_frames, device, seed):
    maps, labels = [], []
    for i in range(n_frames):
        sim = SceneSimulator(scenario=scenario, seq_len=1, device=device,
                             generator=torch.Generator().manual_seed(seed + i))
        o = sim.gen_sequence()
        maps.append(o["x"][0].numpy())
        labels.append([(l["range_bin"], l["doppler_bin"]) for l in o["labels"][0]])
    return describe(np.stack(maps), labels)


def distance(sim, real):
    errs = [abs(sim[k] - real[k]) / SCALE_DB for k in real if not np.isnan(sim[k])]
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-frames", type=int, default=150)
    ap.add_argument("--sim-frames", type=int, default=24)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--refine", type=int, default=60,
                    help="local trials around the best random-search result")
    ap.add_argument("--out", default="samples/scene_sim_fit.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = real_reference(a.real_frames)
    engineer = simulate(Scenario(), a.sim_frames, device, seed=1000)
    print("real (train recordings):", {k: round(v, 1) for k, v in real.items()})
    print(f"engineer's variant: distance {distance(engineer, real):.3f}")

    rng = np.random.default_rng(0)
    trials = []
    for t in range(a.trials):
        knobs = {k: float(rng.uniform(*bounds)) for k, bounds in KNOBS.items()}
        stats = simulate(scenario_from(knobs), a.sim_frames, device, seed=1000)
        d = distance(stats, real)
        trials.append({"knobs": knobs, "stats": stats, "distance": d})
        print(f"trial {t:3d}: distance {d:.3f}", flush=True)
    trials.sort(key=lambda r: r["distance"])

    # local refinement: random search is coarse in 9 dimensions; move the
    # centre whenever a trial beats the best so far
    centre, best_d = trials[0]["knobs"], trials[0]["distance"]
    for t in range(a.refine):
        knobs = {}
        for k, (lo, hi) in KNOBS.items():
            v = centre[k] + rng.normal(0, 0.12 * (hi - lo))
            knobs[k] = float(np.clip(v, lo, hi))
        stats = simulate(scenario_from(knobs), a.sim_frames, device, seed=1000)
        d = distance(stats, real)
        trials.append({"knobs": knobs, "stats": stats, "distance": d, "refine": True})
        if d < best_d:
            centre, best_d = knobs, d
        print(f"refine {t:3d}: distance {d:.3f}", flush=True)
    trials.sort(key=lambda r: r["distance"])

    # re-check the best few on fresh scenes, so the winner is not a lucky draw
    for r in trials[:5]:
        r["distance_fresh"] = distance(
            simulate(scenario_from(r["knobs"]), a.sim_frames, device, seed=5000), real)
    best = min(trials[:5], key=lambda r: r["distance_fresh"])

    print("\nfitted knobs:", json.dumps({k: round(v, 3) for k, v in best["knobs"].items()}))
    print(f"\n{'statistic (dB)':<28}{'real':>9}{'engineer':>10}{'fitted':>9}")
    for k in real:
        print(f"{k:<28}{real[k]:>9.1f}{engineer[k]:>10.1f}{best['stats'][k]:>9.1f}")
    print(f"{'distance':<28}{'':>9}{distance(engineer, real):>10.3f}{best['distance_fresh']:>9.3f}")
    Path(a.out).write_text(json.dumps(
        {"fitted_on": "train recordings only", "real": real,
         "engineer": {"stats": engineer, "distance": distance(engineer, real)},
         "fitted": best, "trials": trials}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
