"""Measure the sim-to-real gap of the specification-driven simulator.

Nothing here is tuned. The simulator is configured entirely from published
radar parameters (src/radar_physics.py) and the residual against real RADIal
recordings is reported as a measurement. That residual is the thing the
generator has to close, so minimising it by fitting would destroy the very
quantity the study is about.

Evaluated on held-out VALIDATION recordings, so the number is not even
incidentally tied to what any model trains on.
"""
import argparse
import json

import numpy as np
import torch

from src import radial
from src.radar_physics import RADIAL_SPEC
from src.simulator import spec_simulator
from scripts.fit_simulator_to_radial import DR_M, prominence, stats


def real_reference(split_name, seq_len, max_frames):
    recs = radial.recordings()
    split = json.load(open("data/radial/split.json"))
    sub = {k: recs[k] for k in split[split_name]}
    seqs = radial.sequences(sub, seq_len=seq_len)
    by_frame = radial.read_labels()
    samples = [s for _, run in seqs for s in run][:max_frames]
    maps, peaks = [], []
    for s in samples:
        p = radial.power_map(s)
        maps.append(p)
        for lab in by_frame.get(s, []):
            v = prominence(p, int(round(float(lab["radar_R_m"]) / DR_M)),
                           int(round(float(lab["radar_D_mps"]))) % p.shape[1])
            if v is not None:
                peaks.append(v)
    return stats(maps, peaks), len(maps), len(peaks)


def sim_reference(seq_len, n_seq, seed=0):
    sim = spec_simulator(RADIAL_SPEC, seq_len=seq_len)
    torch.manual_seed(seed)
    maps, peaks = [], []
    for _ in range(n_seq):
        n = int(torch.randint(1, 4, (1,)).item())
        out = sim.gen_sequence(n_targets=n)
        x = out["x"].numpy()
        maps.extend(x)
        for t in range(n):
            for l in range(0, seq_len, 2):
                rb, vb = out["traj"][t, l]
                v = prominence(x[l], int(round(float(rb))), int(round(float(vb))))
                if v is not None:
                    peaks.append(v)
    return stats(maps, peaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", default="samples/sim2real_gap.json")
    a = ap.parse_args()

    real, n_maps, n_peaks = real_reference("val", a.seq_len, a.max_frames)
    sim = sim_reference(a.seq_len, a.n_seq)
    print(f"real: held-out val, {n_maps} frames, {n_peaks} vehicles")
    print(f"sim : specification-driven, {a.n_seq} sequences, nothing fitted\n")
    print(f"{'statistic':32s} {'real':>9} {'sim':>9} {'gap':>9}")
    gap = {}
    for k in real:
        gap[k] = sim[k] - real[k]
        print(f"{k:32s} {real[k]:9.2f} {sim[k]:9.2f} {gap[k]:+9.2f}")
    json.dump({"real": real, "sim": sim, "gap": gap,
               "spec": RADIAL_SPEC.name, "seq_len": a.seq_len,
               "n_real_frames": n_maps, "n_real_vehicles": n_peaks,
               "n_sim_sequences": a.n_seq,
               "note": "Nothing fitted to recordings; this is the measured gap."},
              open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
