"""Datasets of 8-frame RADIal-grid sequences, simulated or real.

At 512 x 256 a sequence is 2 MB in float16, so a few thousand of them do not
fit the ~15 GB memory of an interactive session. Maps are therefore stored in
one memory-mapped file per split and read on demand; labels sit beside them in
a small .pt file.

Layout of a cache directory:
  <split>_x.npy        (n, L, 512, 256) float16, dB
  <split>_labels.pt    traj (n, MAX_VEHICLES, L, 2) range/Doppler bins,
                       present (n, MAX_VEHICLES, L) bool, meta (list of dicts)
  stats.pt             mean/std of the dB maps over the training split
"""
from pathlib import Path

import numpy as np
import torch

MAX_VEHICLES = 8


def pack_labels(frame_labels, seq_len):
    """Per-frame label lists -> fixed-size traj/present tensors, keyed by id.

    Keyed by the simulator's stable vehicle id, not by position in each
    frame's list: when one vehicle leaves the field of view the others would
    otherwise slide into its slot and swap identities mid-sequence. Moving
    vehicles come first in the simulator's list, so their ids are small.
    """
    traj = torch.zeros(MAX_VEHICLES, seq_len, 2)
    present = torch.zeros(MAX_VEHICLES, seq_len, dtype=torch.bool)
    for t, frame in enumerate(frame_labels):
        for lab in frame:
            i = lab["id"]
            if i >= MAX_VEHICLES:
                continue
            traj[i, t, 0] = lab["range_bin"]
            traj[i, t, 1] = lab["doppler_bin"]
            present[i, t] = True
    return traj, present


class SceneSequenceDataset(torch.utils.data.Dataset):
    """Normalised sequences from a memory-mapped cache."""

    def __init__(self, cache_dir, split, normalise=True):
        cache = Path(cache_dir)
        self.x = np.load(cache / f"{split}_x.npy", mmap_mode="r")
        labels = torch.load(cache / f"{split}_labels.pt")
        self.traj, self.present = labels["traj"], labels["present"]
        self.meta = labels["meta"]
        self.stats = torch.load(cache / "stats.pt")
        self.normalise = normalise

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32))
        if self.normalise:
            x = (x - self.stats["mean"]) / self.stats["std"]
        return {"x": x, "traj": self.traj[i], "present": self.present[i]}
