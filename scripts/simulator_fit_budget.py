"""How much real data does the simulator fit actually need?

The step 3 comparison is only honest if the simulator arm consumes no more real
data than the generator it is compared against. This measures the trade: fit the
simulator on k real sequences, then score that fit against HELD-OUT validation
recordings. The resulting curve says how much of the fit quality survives a
small budget, and fixes what "matched budget" must mean at each point of the
size curve.
"""
import argparse, csv, collections, dataclasses, itertools, json
import numpy as np
import torch

from src import radial
from src.simulator import RADIAL_GEOMETRY, TemporalRadarSimulator
from scripts.fit_simulator_to_radial import (ROOT, DR_M, stats, prominence,
                                             fit_range_gain, score)


def reference_from(samples, by_frame):
    maps, peaks = [], []
    for s in samples:
        p = radial.power_map(s)
        maps.append(p)
        for lab in by_frame.get(s, []):
            v = prominence(p, int(round(float(lab["radar_R_m"]) / DR_M)),
                           int(round(float(lab["radar_D_mps"]))) % p.shape[1])
            if v is not None:
                peaks.append(v)
    if not peaks:
        return None, None
    return stats(maps, peaks), np.median(np.stack(maps), axis=(0, 2))


def sim_eval(n_looks, sigma_f, lo, hi, gain, n_seq=6, seq_len=8, seed=0):
    geom = dataclasses.replace(RADIAL_GEOMETRY, CNR_DB=15.0)
    sim = TemporalRadarSimulator(seq_len=seq_len, geometry=geom, sigma_f=sigma_f,
                                 n_looks=n_looks, range_gain_db=gain)
    torch.manual_seed(seed)
    maps, peaks = [], []
    for _ in range(n_seq):
        n = int(torch.randint(1, 4, (1,)).item())
        out = sim.gen_sequence(n_targets=n, gain_db=torch.empty(n).uniform_(lo, hi))
        x = out["x"].numpy()
        maps.extend(x)
        for t in range(n):
            for l in range(0, seq_len, 2):
                rb, vb = out["traj"][t, l]
                v = prominence(x[l], int(round(float(rb))), int(round(float(vb))))
                if v is not None:
                    peaks.append(v)
    return stats(maps, peaks), np.median(np.stack(maps), axis=(0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--out", default="samples/simulator_fit_budget.json")
    a = ap.parse_args()

    by_frame = radial.read_labels()
    recs = radial.recordings()
    split = json.load(open("data/radial/split.json"))
    train_seqs = radial.sequences({k: recs[k] for k in split["train"]}, seq_len=a.seq_len)
    val_seqs = radial.sequences({k: recs[k] for k in split["val"]}, seq_len=a.seq_len)
    val_samples = [s for _, run in val_seqs for s in run]
    held_out, _ = reference_from(val_samples, by_frame)
    print(f"held-out reference: {len(val_seqs)} val sequences, "
          f"{len(val_samples)} frames")
    for k, v in held_out.items():
        print(f"  {k:28s} {v:8.2f}")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_seqs))
    budgets = [1, 2, 5, 10, 25, 50, 100, len(train_seqs)]
    grid = list(itertools.product((4, 8), (0.1, 0.3),
                                  ((-40.0, -25.0), (-35.0, -20.0),
                                   (-30.0, -15.0), (-25.0, -10.0))))
    rows = []
    for k in budgets:
        picked = [train_seqs[i] for i in order[:k]]
        samples = [s for _, run in picked for s in run]
        fit_ref, fit_prof = reference_from(samples, by_frame)
        if fit_ref is None:
            print(f"k={k}: no labelled vehicles in the sample, skipped")
            continue
        best = None
        for n_looks, sigma_f in {(g[0], g[1]) for g in grid}:
            _, sim_prof = sim_eval(n_looks, sigma_f, -30.0, -15.0, None,
                                   n_seq=3, seq_len=a.seq_len)
            gain = fit_range_gain(fit_prof, sim_prof)
            for lo, hi in {(g[2][0], g[2][1]) for g in grid}:
                st, _ = sim_eval(n_looks, sigma_f, lo, hi, torch.from_numpy(
                    gain.astype(np.float32)), n_seq=6, seq_len=a.seq_len)
                sc_fit = score(st, fit_ref)         # against what it was fitted on
                if best is None or sc_fit < best[0]:
                    best = (sc_fit, n_looks, sigma_f, lo, hi, st)
        sc_fit, n_looks, sigma_f, lo, hi, st = best
        sc_held = score(st, held_out)
        rows.append({"k_sequences": k, "n_frames": len(samples),
                     "n_looks": n_looks, "sigma_f": sigma_f, "gain_db": [lo, hi],
                     "score_on_fit_data": sc_fit, "score_on_heldout": sc_held,
                     "stats": st})
        print(f"  k={k:3d} ({len(samples):4d} frames): looks {n_looks} sf {sigma_f} "
              f"gain [{lo:.0f},{hi:.0f}] | fit {sc_fit:.3f}  HELD-OUT {sc_held:.3f}",
              flush=True)
    json.dump({"held_out_reference": held_out, "seq_len": a.seq_len,
               "n_train_sequences": len(train_seqs), "rows": rows},
              open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
