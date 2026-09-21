"""Measure RADIal RD-map statistics that a simulator must reproduce.

Grid (established by scripts/radial_calibrate.py): axis0 = range at 0.2 m/bin,
axis1 = Doppler in unshifted FFT order.  labels.csv's `radar_D_mps` column
holds an integer Doppler *bin index*, not metres per second.
"""
import csv, collections, sys
import numpy as np

ROOT = "/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal"
DR_M = 0.2


def labels_by_frame():
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{ROOT}/labels.csv")):
        d[int(r["numSample"])].append(r)
    return d


def power_db(sample):
    x = np.load(f"{ROOT}/radar_FFT/fft_{sample:06d}.npy")
    p = (np.abs(x) ** 2).sum(axis=-1)
    return 10 * np.log10(p + 1e-12)


def check_grid(labels, n=60):
    """A labelled vehicle should sit on a local maximum at (R/0.2, D_bin)."""
    hits = []
    for sample, labs in list(labels.items())[:n]:
        p = power_db(sample)
        for lab in labs:
            ri = int(round(float(lab["radar_R_m"]) / DR_M))
            di = int(round(float(lab["radar_D_mps"]))) % p.shape[1]
            if not (2 <= ri < p.shape[0] - 2):
                continue
            win = p[ri - 2:ri + 3, (di - 2) % p.shape[1]:(di + 3) % p.shape[1]] \
                if 2 <= di < p.shape[1] - 2 else None
            if win is None or win.size == 0:
                continue
            local = win.max()
            hits.append(local - np.median(p[ri - 2:ri + 3, :]))
    a = np.array(hits)
    print(f"grid check: {len(a)} labelled vehicles")
    print(f"  peak above that range ring's median, dB: "
          f"p10 {np.percentile(a,10):.1f}  median {np.median(a):.1f}  "
          f"p90 {np.percentile(a,90):.1f}")


def scene_stats(labels, n=40):
    maps = []
    for sample in list(labels)[:n]:
        maps.append(power_db(sample))
    m = np.stack(maps)
    print(f"\nRD power (dB) over {len(maps)} frames, shape {m.shape[1:]}")
    print(f"  global   min {m.min():.1f}  median {np.median(m):.1f}  max {m.max():.1f}")
    zero_d = m[:, :, 0]
    far = m[:, :, 8:248]
    print(f"  zero-Doppler column median {np.median(zero_d):.1f} dB")
    print(f"  non-clutter Doppler median {np.median(far):.1f} dB")
    print(f"  clutter ridge above background: {np.median(zero_d)-np.median(far):.1f} dB")
    prof = np.median(m, axis=(0, 2))
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 350), (350, 512)):
        print(f"  range bins {lo:3d}-{hi:3d} ({lo*DR_M:5.1f}-{hi*DR_M:5.1f} m): "
              f"median {np.median(prof[lo:hi]):.1f} dB")
    dop = np.median(m, axis=(0, 1))
    width = int((dop > np.median(far) + 3).sum())
    print(f"  Doppler bins more than 3 dB above background: {width} of 256")


if __name__ == "__main__":
    lab = labels_by_frame()
    check_grid(lab, int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    scene_stats(lab)
