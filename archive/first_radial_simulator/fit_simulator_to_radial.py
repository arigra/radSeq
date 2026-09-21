"""Fit the simulator's clutter/noise/target parameters to RADIal.

The premise of step 3 is that the simulator is a *good but imperfect* stand-in
for real radar: if it matched perfectly the generator would be pointless, and
if it matched badly the comparison would be a strawman. So it is fitted here
on four statistics that are computed identically on real and simulated maps,
and the residual is reported rather than hidden.

Only training recordings may be used -- fitting on the test split would leak
real data into the "simulator" arm.
"""
import argparse, csv, collections, itertools, json
import numpy as np
import torch

from src.simulator import RADIAL_GEOMETRY, TemporalRadarSimulator

ROOT = "/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal"
DR_M = 0.2


# ---------------------------------------------------------------- statistics
def stats(maps, peaks):
    """The four quantities the fit matches. `maps` is (frames, N, K) in dB,
    `peaks` the per-target prominence over its range ring in dB."""
    m = np.asarray(maps)
    prof_r = np.median(m, axis=(0, 2))
    prof_d = np.median(m, axis=(0, 1))
    return {
        "target_prominence_db": float(np.median(peaks)),
        "dynamic_range_db": float(m.max() - m.min()),
        "doppler_profile_spread_db": float(prof_d.max() - prof_d.min()),
        "range_profile_spread_db": float(prof_r.max() - prof_r.min()),
    }


def prominence(frame, ri, di, half=2):
    n, k = frame.shape
    if not (half <= ri < n - half and half <= di < k - half):
        return None
    return float(frame[ri - half:ri + half + 1, di - half:di + half + 1].max()
                 - np.median(frame[ri - half:ri + half + 1, :]))


# ------------------------------------------------------------------- RADIal
def training_samples(limit=None):
    """Frames from the pinned TRAINING recordings only.

    Fitting on val/test recordings would leak real data into the arm that is
    supposed to represent "simulator only".
    """
    import json
    from src import radial
    recs = radial.recordings()
    split = json.load(open("data/radial/split.json"))
    samples = sorted(s for name in split["train"] for s in recs[name])
    return samples[:limit] if limit else samples


def fit_range_gain(radial_prof, sim_prof):
    """dB gain per range bin carrying RADIal's receiver response.

    RADIal has a near blind zone, a peak near 12 m, a gentle falloff and a
    sharp roll-off past ~96 m; the simulator's clutter is uniform in range.
    Both profiles are referenced to their own median so only the *shape* is
    transferred, never the absolute level.
    """
    import numpy as np
    r = np.asarray(radial_prof) - np.median(radial_prof)
    s = np.asarray(sim_prof) - np.median(sim_prof)
    return r - s


def radial_reference(n_frames, recordings=None):
    by_frame = collections.defaultdict(list)
    allowed = set(training_samples())
    for r in csv.DictReader(open(f"{ROOT}/labels.csv")):
        if float(r["radar_R_m"]) < 0:
            continue                      # invalid label rows
        if int(r["numSample"]) in allowed:
            by_frame[int(r["numSample"])].append(r)
    maps, peaks = [], []
    chosen = sorted(by_frame)[::max(1, len(by_frame) // max(n_frames, 1))][:n_frames]
    for sample in chosen:
        x = np.load(f"{ROOT}/radar_FFT/fft_{sample:06d}.npy")
        p = 10 * np.log10((np.abs(x) ** 2).sum(axis=-1) + 1e-12)
        maps.append(p)
        for lab in by_frame[sample]:
            v = prominence(p, int(round(float(lab["radar_R_m"]) / DR_M)),
                           int(round(float(lab["radar_D_mps"]))) % p.shape[1])
            if v is not None:
                peaks.append(v)
    prof = np.median(np.stack(maps), axis=(0, 2))
    return stats(maps, peaks), len(maps), len(peaks), prof


# ---------------------------------------------------------------- simulator
def sim_stats(cnr_db, sigma_f, gain_lo, gain_hi, n_seq=8, seq_len=8, seed=0,
              n_looks=1, range_gain_db=None, want_profile=False):
    import dataclasses
    geom = dataclasses.replace(RADIAL_GEOMETRY, CNR_DB=cnr_db)
    sim = TemporalRadarSimulator(seq_len=seq_len, geometry=geom, sigma_f=sigma_f,
                                 n_looks=n_looks, range_gain_db=range_gain_db)
    torch.manual_seed(seed)
    maps, peaks = [], []
    for _ in range(n_seq):
        n = int(torch.randint(1, 4, (1,)).item())
        gains = torch.empty(n).uniform_(gain_lo, gain_hi)
        out = sim.gen_sequence(n_targets=n, gain_db=gains)
        x = out["x"].numpy()
        maps.extend(x)
        for t in range(n):
            for l in range(0, seq_len, 2):
                rb, vb = out["traj"][t, l]
                v = prominence(x[l], int(round(float(rb))), int(round(float(vb))))
                if v is not None:
                    peaks.append(v)
    if want_profile:
        return stats(maps, peaks), np.median(np.stack(maps), axis=(0, 2))
    return stats(maps, peaks)


def score(sim, ref):
    """Relative error per statistic, averaged. Doppler/range spreads are near
    zero in RADIal so they are scored on an absolute dB scale instead."""
    s = 0.0
    s += abs(sim["target_prominence_db"] - ref["target_prominence_db"]) / 10.0
    s += abs(sim["dynamic_range_db"] - ref["dynamic_range_db"]) / 30.0
    s += abs(sim["doppler_profile_spread_db"] - ref["doppler_profile_spread_db"]) / 10.0
    s += abs(sim["range_profile_spread_db"] - ref["range_profile_spread_db"]) / 10.0
    return s / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--n-seq", type=int, default=6)
    ap.add_argument("--out", default="archive/first_radial_simulator/results/simulator_fit_radial.json")
    a = ap.parse_args()

    ref, n_maps, n_peaks, radial_prof = radial_reference(a.frames)
    print(f"RADIal reference, TRAIN recordings only "
          f"({n_maps} frames, {n_peaks} vehicles):")
    for k, v in ref.items():
        print(f"  {k:28s} {v:8.2f}")

    results = []
    for n_looks in (1, 2, 4, 8):
        for sigma_f in (0.1, 0.3, 0.6):
            # fit the receiver range response for this clutter setting, then
            # search target gain with that response held fixed
            _, sim_prof = sim_stats(15.0, sigma_f, -20.0, -5.0, n_seq=3,
                                    n_looks=n_looks, want_profile=True)
            gain = fit_range_gain(radial_prof, sim_prof)
            for lo, hi in ((-45.0, -30.0), (-40.0, -25.0), (-35.0, -20.0),
                           (-30.0, -15.0), (-25.0, -10.0), (-15.0, 0.0)):
                st = sim_stats(15.0, sigma_f, lo, hi, n_seq=a.n_seq,
                               n_looks=n_looks, range_gain_db=gain)
                sc = score(st, ref)
                results.append({"n_looks": n_looks, "sigma_f": sigma_f,
                                "cnr_db": 15.0, "gain_db": [lo, hi],
                                "stats": st, "score": sc})
                print(f"  looks {n_looks} sf {sigma_f:4.2f} gain [{lo:6.1f},{hi:6.1f}]"
                      f" -> prom {st['target_prominence_db']:6.1f}"
                      f" dyn {st['dynamic_range_db']:6.1f}"
                      f" dop {st['doppler_profile_spread_db']:5.1f}"
                      f" rng {st['range_profile_spread_db']:6.1f} | {sc:.3f}", flush=True)

    results.sort(key=lambda r: r["score"])
    best = results[0]
    _, sim_prof = sim_stats(15.0, best["sigma_f"], -20.0, -5.0, n_seq=3,
                            n_looks=best["n_looks"], want_profile=True)
    gain = fit_range_gain(radial_prof, sim_prof)
    print("\nbest:", json.dumps({k: v for k, v in best.items()}, indent=2))
    print("\nresidual (sim - RADIal):")
    for k in ref:
        print(f"  {k:28s} {best['stats'][k] - ref[k]:+8.2f}")
    json.dump({"reference": ref, "n_frames": n_maps, "n_vehicles": n_peaks,
               "range_gain_db": gain.tolist(), "results": results,
               "best": best}, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
