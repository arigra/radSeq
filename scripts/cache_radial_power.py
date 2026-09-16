"""Cache measured RADIal RD power maps for runs of consecutive labelled frames.

Reads RADIal ready-to-use `radar_FFT/fft_*.npy` (512 range x 256 Doppler x 16
complex channels), sums |.|^2 over channels, and stores 10*log10 power as
float32. Only runs of >= `min_run` consecutive labelled frames (same record,
index step 1) are kept, because the research question is temporal.
Vehicle labels (range metres, Doppler label) are stored per frame.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal")


def power_db(num_sample):
    fft = np.load(ROOT / "radar_FFT" / f"fft_{num_sample:06d}.npy")
    power = (fft.real ** 2 + fft.imag ** 2).sum(-1)
    return (10 * np.log10(np.maximum(power, 1e-12))).astype(np.float32)


def runs(labels, min_run):
    frames = labels.drop_duplicates("numSample")
    out = []
    for record, x in frames.groupby("dataset"):
        x = x.sort_values("index")
        breaks = np.flatnonzero(np.diff(x["index"].values) != 1) + 1
        for chunk in np.split(x["numSample"].values, breaks):
            if len(chunk) >= min_run:
                out.append((record, [int(n) for n in chunk]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/measured/radial")
    parser.add_argument("--min-run", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "labels.csv")
    manifest = []
    with Pool(args.workers) as pool:
        for i, (record, samples) in enumerate(runs(labels, args.min_run)):
            maps = np.stack(pool.map(power_db, samples))
            vehicles = [labels.loc[(labels.numSample == n) & (labels.radar_R_m >= 0),
                                   ["radar_R_m", "radar_A_deg", "radar_D_mps"]]
                        .values.tolist() for n in samples]
            np.save(out / f"run_{i:03d}.npy", maps)
            manifest.append({"run": i, "record": record, "samples": samples,
                             "vehicles": vehicles})
            print(i, record, maps.shape, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    main()
