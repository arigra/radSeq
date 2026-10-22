"""Statistics that tell a generated RADIal map from a real one, as a detector
would see them.

Measured 2026-10-22 on the step 4 generator: its maps had ~40% less speckle,
~40% less frame-to-frame change, ~9 dB lower brightest cells and a quarter of
the strong clutter peaks of real maps. A detector trained on such clean maps is
swamped by clutter on real ones.

All inputs are dB maps, x (L, n_range, n_doppler) per sequence, with labels
traj (slots, L, 2) and present (slots, L).
"""
import numpy as np

DDMA_SLOT = 16
PEAK_DB = 15.0      # a strong peak: this far above its range ring


def speckle_db(x):
    """Cell-to-cell texture: std of the map minus its 4-neighbour mean."""
    hp = x - (np.roll(x, 1, -2) + np.roll(x, -1, -2) + np.roll(x, 1, -1) + np.roll(x, -1, -1)) / 4
    return float(hp.std())


def frame_change_db(x):
    return float(np.abs(np.diff(x, axis=0)).mean())


def brightest_db(x):
    return float(np.percentile(x, 99.9))


def off_label_peaks(x, traj, present):
    """Strong local maxima per frame that are not a labelled vehicle (or one of
    its DDMA copies): the clutter a detector must learn to ignore."""
    total = 0
    for t in range(x.shape[0]):
        e = x[t] - np.median(x[t], axis=1, keepdims=True)
        peak = ((e > PEAK_DB) & (e >= np.roll(e, 1, 0)) & (e >= np.roll(e, -1, 0))
                & (e >= np.roll(e, 1, 1)) & (e >= np.roll(e, -1, 1)))
        rr, dd = np.nonzero(peak)
        keep = np.ones(len(rr), bool)
        for m in range(traj.shape[0]):
            if not present[m, t]:
                continue
            r, d = float(traj[m, t, 0]), float(traj[m, t, 1])
            off = (dd - d) % DDMA_SLOT
            keep &= ~((np.abs(rr - r) <= 3) & ((off <= 2) | (off >= DDMA_SLOT - 2)))
        total += int(keep.sum())
    return total / x.shape[0]


STATS = {"speckle_db": speckle_db, "frame_change_db": frame_change_db,
         "brightest_db": brightest_db}


def realism(items):
    """Mean of each statistic over a list of {"x", "traj", "present"} items."""
    out = {k: [] for k in (*STATS, "off_label_peaks")}
    for it in items:
        x = np.asarray(it["x"], dtype=np.float32)
        for k, f in STATS.items():
            out[k].append(f(x))
        out["off_label_peaks"].append(off_label_peaks(x, np.asarray(it["traj"]),
                                                      np.asarray(it["present"])))
    return {k: float(np.mean(v)) for k, v in out.items()}


def mismatch(gen, real):
    """Per-statistic relative error and their mean (the sampler-selection score).
    brightest_db is compared above the map median, so the 97 dB level does not
    hide a 9 dB shortfall."""
    rel = {}
    for k in real:
        if k == "brightest_db":
            g, r = gen[k] - gen["_median"], real[k] - real["_median"]
        elif k.startswith("_"):
            continue
        else:
            g, r = gen[k], real[k]
        rel[k] = abs(g - r) / abs(r)
    return rel, float(np.mean(list(rel.values())))


def with_median(stats, items):
    stats = dict(stats)
    stats["_median"] = float(np.median(np.stack([np.asarray(it["x"], np.float32)
                                                 for it in items])))
    return stats


def vehicle_persistence(x, traj, present, win_r=4, win_d=2):
    """Temporal fidelity of the vehicles themselves: Pearson correlation of each
    vehicle's window (around its label) between consecutive frames, averaged.
    Real vehicles change slowly; a generator that renders each frame on its
    own, or smooths them in time, sits below or above real."""
    vals = []
    for m in range(traj.shape[0]):
        for t in range(x.shape[0] - 1):
            if not (present[m, t] and present[m, t + 1]):
                continue
            wins = []
            for f in (t, t + 1):
                r, d = int(round(float(traj[m, f, 0]))), int(round(float(traj[m, f, 1])))
                if not win_r <= r < x.shape[1] - win_r:
                    break
                cols = [(d + k) % x.shape[2] for k in range(-win_d, win_d + 1)]
                wins.append(x[f, r - win_r:r + win_r + 1][:, cols].ravel())
            if len(wins) == 2 and wins[0].std() > 0 and wins[1].std() > 0:
                vals.append(float(np.corrcoef(wins[0], wins[1])[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def temporal(items):
    """Temporal statistics over items: frame-to-frame change of the whole map
    and of the vehicles."""
    fc, vp = [], []
    for it in items:
        x = np.asarray(it["x"], dtype=np.float32)
        fc.append(frame_change_db(x))
        v = vehicle_persistence(x, np.asarray(it["traj"]), np.asarray(it["present"]))
        if v == v:
            vp.append(v)
    return {"frame_change_db": float(np.mean(fc)),
            "vehicle_persistence": float(np.mean(vp)) if vp else float("nan"),
            "n_vehicle_pairs_seqs": len(vp)}
