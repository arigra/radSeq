"""RADIal real-radar access: splits, sequence extraction, and RD maps.

Grid facts established in docs/notes/2026-09-21-radial-sim2real-design.md:
`radar_FFT/fft_NNNNNN.npy` is (512 range, 256 Doppler, 16 Rx) complex64, range
is 0.2 m per bin, Doppler is in unshifted FFT order, and labels.csv's
`radar_D_mps` column holds an integer Doppler *bin index* despite its name.
"""
import collections
import csv
import hashlib
import os

import numpy as np
import torch

DEFAULT_ROOT = "/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal"
DR_M = 0.2
N_RANGE, N_DOPPLER = 512, 256


def read_labels(root=DEFAULT_ROOT):
    """Valid label rows grouped by frame. Rows with radar_R_m < 0 are dropped."""
    by_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(root, "labels.csv"))):
        if float(r["radar_R_m"]) < 0:
            continue
        by_frame[int(r["numSample"])].append(r)
    return by_frame


def recordings(root=DEFAULT_ROOT):
    by_frame = read_labels(root)
    recs = collections.defaultdict(set)
    for sample, labs in by_frame.items():
        recs[labs[0]["dataset"]].add(sample)
    return {k: sorted(v) for k, v in sorted(recs.items())}


def split_recordings(recs, test_frac=0.2, val_frac=0.1, seed="radseq-step3",
                     seq_len=16):
    """Split whole recordings, balanced by how many sequences each contributes.

    Splitting by frame would leak: consecutive frames of one drive are nearly
    identical, so a frame-level test set measures memorisation, not transfer.

    Recordings differ ~20x in length, so assigning them independently (e.g. by
    hashing the name) balances *recordings* while leaving the sequence counts
    badly skewed -- it put 6 of 219 sequences in val, too few to early-stop on.
    Instead recordings are walked in a fixed hashed order and each is given to
    whichever split is furthest below its target share.

    The result is deterministic for a given set of recordings, but is not
    stable if recordings are added. The split is therefore generated once and
    pinned to data/radial/split.json, which is the source of truth.
    """
    weights = {name: sum(len(w[1]) // seq_len for w in
                         _runs(name, recs[name], seq_len))
               for name in recs}
    total = sum(weights.values())
    remaining = {"train": (1.0 - test_frac - val_frac) * total,
                 "val": val_frac * total, "test": test_frac * total}
    # Largest recording first: a 320-frame drive placed late cannot be
    # balanced against, so it has to be assigned while every split is empty.
    order = sorted(recs, key=lambda n: (-weights[n], hashlib.sha256(
        f"{seed}/{n}".encode()).hexdigest()))
    out = {k: [] for k in remaining}
    for name in order:
        pick = max(remaining, key=lambda k: (remaining[k], k))
        out[pick].append(name)
        remaining[pick] -= weights[name]
    return {k: sorted(v) for k, v in out.items()}


def _runs(name, samples, seq_len):
    """Maximal runs of consecutive frames, as (name, run) pairs."""
    out, run = [], [samples[0]] if samples else []
    for prev, cur in zip(samples, samples[1:]):
        if cur == prev + 1:
            run.append(cur)
        else:
            out.append((name, run))
            run = [cur]
    if run:
        out.append((name, run))
    return out


def sequences(recs, seq_len=16, stride=None):
    """Runs of `seq_len` consecutive labelled frames within one recording.

    Default stride is `seq_len`, i.e. non-overlapping: overlapping windows from
    one drive are near-duplicates and would inflate any count of "independent"
    training data.
    """
    stride = stride or seq_len
    out = []
    for name, samples in recs.items():
        run = [samples[0]]
        for prev, cur in zip(samples, samples[1:]):
            if cur == prev + 1:
                run.append(cur)
            else:
                out += _windows(name, run, seq_len, stride)
                run = [cur]
        out += _windows(name, run, seq_len, stride)
    return out


def _windows(name, run, seq_len, stride):
    return [(name, run[i:i + seq_len])
            for i in range(0, len(run) - seq_len + 1, stride)]


def power_map(sample, root=DEFAULT_ROOT, channels=None):
    """RD power in dB, summed over receive channels (non-coherent looks)."""
    x = np.load(os.path.join(root, "radar_FFT", f"fft_{sample:06d}.npy"))
    if channels is not None:
        x = x[..., :channels]
    return 10 * np.log10((np.abs(x) ** 2).sum(axis=-1) + 1e-12)


def load_sequence(samples, root=DEFAULT_ROOT, channels=None):
    return torch.from_numpy(
        np.stack([power_map(s, root, channels) for s in samples])).float()


def targets(sample, by_frame):
    """(range_bin, doppler_bin) per labelled vehicle in a frame."""
    out = []
    for lab in by_frame.get(sample, []):
        out.append((float(lab["radar_R_m"]) / DR_M,
                    float(lab["radar_D_mps"]) % N_DOPPLER))
    return out


def range_profile(samples, root=DEFAULT_ROOT):
    """Median dB per range bin: the receiver response the simulator is fitted to."""
    m = np.stack([power_map(s, root) for s in samples])
    return np.median(m, axis=(0, 2))


def fitted_simulator(seq_len=16, root=None, **kwargs):
    """The simulator as fitted to RADIal's training recordings.

    Parameters come from configs/radial_sim.yaml; see
    docs/notes/2026-09-21-radial-sim2real-design.md for the fit and residuals.
    """
    import yaml
    from src.simulator import RADIAL_GEOMETRY, TemporalRadarSimulator
    import dataclasses
    cfg = yaml.safe_load(open("configs/radial_sim.yaml"))["simulator"]
    gain = torch.from_numpy(np.load(cfg["range_gain_db"])).float()
    geom = dataclasses.replace(RADIAL_GEOMETRY, CNR_DB=float(cfg["cnr_db"]))
    return TemporalRadarSimulator(
        seq_len=seq_len, geometry=geom, sigma_f=float(cfg["sigma_f"]),
        n_looks=int(cfg["n_looks"]), range_gain_db=gain, **kwargs), cfg


def target_gain_draw(n, cfg, generator=None):
    lo, hi = cfg["target_gain_db"]
    return torch.empty(n).uniform_(float(lo), float(hi), generator=generator)


def background_stats(maps):
    """Scene-level statistics of RD maps (dB), used to compare sources.

    They describe the background (level shape over range and Doppler, spread of
    values) rather than targets, so they apply to any map, labelled or not.
    """
    m = np.asarray(maps)
    prof_r = np.median(m, axis=(0, 2))
    prof_d = np.median(m, axis=(0, 1))
    lo, hi = np.percentile(m, (0.1, 99.9))
    mid = np.median(prof_r[100:300])
    return {
        "dynamic range (0.1-99.9%)": float(hi - lo),
        "range-profile spread": float(prof_r.max() - prof_r.min()),
        "Doppler-profile spread": float(prof_d.max() - prof_d.min()),
        "near range (0-4 m) vs mid": float(np.median(prof_r[:20]) - mid),
        "far range (>98 m) vs mid": float(np.median(prof_r[490:]) - mid),
    }


CAMERA_SCALE = 0.5   # boxes are in the 1920x1080 sensor frame; stored images are 960x540


def camera_image(sample, root=DEFAULT_ROOT):
    from PIL import Image
    return np.asarray(Image.open(os.path.join(root, "camera", f"image_{sample:06d}.jpg")))


def vehicles(sample, by_frame):
    """Labelled vehicles of one frame, in both views.

    box: (x1, y1, x2, y2) in the stored camera image; range_bin / doppler_bin
    in the RD map (doppler_bin is the Tx0 copy -- see scene_sim.DDMA_OFFSETS).
    """
    out = []
    for lab in by_frame.get(sample, []):
        box = tuple(float(lab[k]) * CAMERA_SCALE
                    for k in ("x1_pix", "y1_pix", "x2_pix", "y2_pix"))
        out.append({"box": box, "range_m": float(lab["radar_R_m"]),
                    "range_bin": float(lab["radar_R_m"]) / DR_M,
                    "doppler_bin": float(lab["radar_D_mps"]) % N_DOPPLER})
    return out
