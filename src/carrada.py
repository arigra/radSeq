"""CARRADA real-radar access: recordings, 8-frame sequences, RD maps, labels.

Conventions measured in docs/notes/2026-10-26-carrada-raddet-facts.md:
`range_doppler_raw/NNNNNN.npy` is linear power (256 range x 64 Doppler), range
row 0 is near (0.2 m/bin), Doppler bin 32 is zero velocity (0.42 m/s/bin) and
approaching objects sit above it; frames are 10 Hz. Frames without annotations
carry no moving object, so they are kept as empty frames.

A vehicle's label is the centre of its RD box; instances keep their CARRADA
id across frames, so no linking is needed (unlike RADIal).
"""
import json
import os

import numpy as np
import torch

DEFAULT_ROOT = "/truenas/home/arigra/permuter/ariGranevich/radar data/Carrada"
N_RANGE, N_DOPPLER = 256, 64
DR_M, DV_MPS, ZERO_DOPPLER = 0.2, 0.42, 32
CLASSES = {1: "pedestrian", 2: "cyclist", 3: "car"}


def recordings(root=DEFAULT_ROOT):
    """All frames per recording, in time order: {name: ['000000', ...]}."""
    light = json.load(open(os.path.join(root, "light_dataset_frame_oriented.json")))
    return {k: sorted(v, key=int) for k, v in sorted(light.items())}


def official_split(root=DEFAULT_ROOT):
    """CARRADA's own recording-level split: {name: 'Train'|'Validation'|'Test'}."""
    ref = json.load(open(os.path.join(root, "data_seq_ref.json")))
    return {k: v["split"] for k, v in ref.items()}


def annotations(root=DEFAULT_ROOT):
    return json.load(open(os.path.join(root, "annotations_frame_oriented.json")))


def sequences(recs, seq_len=8):
    """Non-overlapping windows of `seq_len` consecutive frames per recording."""
    out = []
    for name, frames in recs.items():
        run = [frames[0]]
        for prev, cur in zip(frames, frames[1:]):
            if int(cur) == int(prev) + 1:
                run.append(cur)
            else:
                out += [(name, run[i:i + seq_len]) for i in range(0, len(run) - seq_len + 1, seq_len)]
                run = [cur]
        out += [(name, run[i:i + seq_len]) for i in range(0, len(run) - seq_len + 1, seq_len)]
    return out


def power_db(name, frame, root=DEFAULT_ROOT):
    x = np.load(os.path.join(root, name, "range_doppler_raw", f"{frame}.npy"))
    return 10 * np.log10(x + 1e-12)


def load_sequence(name, frames, root=DEFAULT_ROOT):
    return torch.from_numpy(np.stack([power_db(name, f, root) for f in frames]).astype(np.float32))


def frame_labels(name, frame, ann):
    """[{id, range_bin, doppler_bin, cls}] for one frame (empty when no object)."""
    out = []
    for iid, a in sorted(ann.get(name, {}).get(frame, {}).items()):
        (r0, d0), (r1, d1) = a["range_doppler"]["box"]
        out.append({"iid": iid, "range_bin": (r0 + r1) / 2, "doppler_bin": (d0 + d1) / 2,
                    "cls": int(a["range_doppler"]["label"])})
    return out


def pack(frames_labels, seq_len, max_objects=8):
    """Per-frame label lists -> traj (M, L, 2), present (M, L), cls (M,), slot by instance id."""
    ids = sorted({lab["iid"] for fl in frames_labels for lab in fl})[:max_objects]
    traj = torch.zeros(max_objects, seq_len, 2)
    present = torch.zeros(max_objects, seq_len, dtype=torch.bool)
    cls = torch.zeros(max_objects, dtype=torch.long)
    for t, fl in enumerate(frames_labels):
        for lab in fl:
            if lab["iid"] not in ids:
                continue
            m = ids.index(lab["iid"])
            traj[m, t] = torch.tensor([lab["range_bin"], lab["doppler_bin"]])
            present[m, t] = True
            cls[m] = lab["cls"]
    return traj, present, cls
