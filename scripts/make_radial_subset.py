"""Nested random subsets of the real RADIal training sequences (step 5).

data/radial_seq_n<N> holds the first N sequences of one fixed permutation of
the 387 training sequences, so every smaller subset is contained in every
larger one. Its normalisation statistics are computed on the subset alone;
val and test are links to the full cache, unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

PERMUTATION_SEED = 0


def subset_indices(n_total, n, seed=PERMUTATION_SEED):
    return sorted(np.random.default_rng(seed).permutation(n_total)[:n].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--full", default="data/radial_seq")
    a = ap.parse_args()
    full = Path(a.full)
    out = Path(f"{a.full}_n{a.n}")
    if (out / "DONE").exists():
        print(f"skip {out}")
        return
    out.mkdir(exist_ok=True)
    x = np.load(full / "train_x.npy", mmap_mode="r")
    idx = subset_indices(len(x), a.n)
    sub = open_memmap(out / "train_x.npy", mode="w+", dtype=x.dtype,
                      shape=(a.n,) + x.shape[1:])
    for k, i in enumerate(idx):
        sub[k] = x[i]
    sub.flush()
    labels = torch.load(full / "train_labels.pt")
    sub_labels = {"traj": labels["traj"][idx], "present": labels["present"][idx],
                  "meta": [labels["meta"][i] for i in idx]}
    if "cls" in labels:                      # CARRADA: per-object class
        sub_labels["cls"] = labels["cls"][idx]
    torch.save(sub_labels, out / "train_labels.pt")
    arr = np.asarray(sub, dtype=np.float32)
    torch.save({"mean": float(arr.mean()), "std": float(arr.std())}, out / "stats.pt")
    for split in ("val", "test"):
        for name in (f"{split}_x.npy", f"{split}_labels.pt"):
            link = out / name
            if not link.exists():
                link.symlink_to((full / name).resolve())
    (out / "DONE").write_text(json.dumps({"n": a.n, "indices": idx,
                                          "permutation_seed": PERMUTATION_SEED}))
    print(f"wrote {out}: {a.n} sequences", flush=True)


if __name__ == "__main__":
    main()
