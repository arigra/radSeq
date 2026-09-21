"""Empirically calibrate RADIal's range-Doppler grid.

The dataset ships range-Doppler tensors (512, 256, 16) but the axis convention
and the metres/bin, (m/s)/bin scaling are not stated in a form we can trust for
matching a simulator to them.  labels.csv gives radar_R_m and radar_D_mps per
vehicle, so the grid can be read off the data: locate each labelled vehicle's
peak and regress bin index against the physical value.
"""
import csv, collections, sys
import numpy as np

ROOT = "/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal"


def load_labels():
    by_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{ROOT}/labels.csv")):
        by_frame[int(r["numSample"])].append(r)
    return by_frame


def power_map(sample):
    """|.|^2 summed over the 16 receive channels -> (512, 256)."""
    x = np.load(f"{ROOT}/radar_FFT/fft_{sample:06d}.npy")
    return (np.abs(x) ** 2).sum(axis=-1)


def main(n_frames=40):
    labels = load_labels()
    single = [(s, r[0]) for s, r in labels.items() if len(r) == 1]
    single.sort(key=lambda sr: -float(sr[1]["radar_P_db"]))
    rows = []
    for sample, lab in single[:n_frames]:
        p = power_map(sample)
        i, j = np.unravel_index(np.argmax(p), p.shape)
        rows.append((i, j, float(lab["radar_R_m"]), float(lab["radar_D_mps"]),
                     float(lab["radar_P_db"])))
    a = np.array(rows)
    print(f"{'axis0':>6} {'axis1':>6} {'R_m':>8} {'D_mps':>8}")
    for r in a[:15]:
        print(f"{r[0]:6.0f} {r[1]:6.0f} {r[2]:8.2f} {r[3]:8.2f}")
    for name, axis in (("axis0", 0), ("axis1", 1)):
        for phys, pname in ((2, "R_m"), (3, "D_mps")):
            c = np.corrcoef(a[:, axis], a[:, phys])[0, 1]
            if abs(c) > 0.9:
                slope, intercept = np.polyfit(a[:, phys], a[:, axis], 1)
                print(f"{name} vs {pname}: corr {c:+.3f}  "
                      f"bin = {slope:.4f}*{pname} + {intercept:.2f}  "
                      f"-> {1/slope:.4f} per bin")
            else:
                print(f"{name} vs {pname}: corr {c:+.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
