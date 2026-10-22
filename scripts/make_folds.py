"""Recording-level cross-validation folds for the RADIal study (step 7).

docs/design/2026-10-22-evaluation-redesign.md: the old test recordings were
seen once (step 5's sim-only AP prompted the car redesign), so instead of one
fixed split every recording is held out in turn. All 91 recordings (vehicle-
free frames included) go into K = 5 folds balanced by the 8-frame sequences
they contribute -- recordings differ ~20x in length. Fold k: test = fold k,
val = fold k+1, train = the other three.

  --plan       write data/radial/folds.json and data/radial/all.json
  --build k    cut data/radial_fold<k> out of data/radial_all (built first by
               build_radial_cache.py --split-json data/radial/all.json --out data/radial_all)
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src import radial

K = 5
SEQ_LEN = 8


def plan():
    recs = radial.recordings()
    counts = {k: len(radial.sequences({k: v}, seq_len=SEQ_LEN)) for k, v in recs.items()}
    # largest first, ties broken by a hash of the name: deterministic, not alphabetical
    order = sorted(recs, key=lambda k: (-counts[k], hashlib.sha256(k.encode()).hexdigest()))
    folds, load = [[] for _ in range(K)], [0] * K
    for k in order:
        i = int(np.argmin(load))
        folds[i].append(k)
        load[i] += counts[k]
    Path("data/radial/folds.json").write_text(json.dumps(
        {"k": K, "seq_len": SEQ_LEN, "folds": [sorted(f) for f in folds],
         "sequences_per_fold": load}, indent=2))
    Path("data/radial/all.json").write_text(json.dumps({"all": sorted(recs)}, indent=2))
    print("sequences per fold", load, "recordings per fold", [len(f) for f in folds])


def build(k):
    out = Path(f"data/radial_fold{k}")
    if (out / "DONE").exists():
        print(f"skip {out}")
        return
    folds = json.loads(Path("data/radial/folds.json").read_text())["folds"]
    role = {"test": set(folds[k]), "val": set(folds[(k + 1) % K])}
    role["train"] = set(sum(folds, [])) - role["test"] - role["val"]
    src = Path("data/radial_all")
    x = np.load(src / "all_x.npy", mmap_mode="r")
    labels = torch.load(src / "all_labels.pt")
    out.mkdir(exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        idx = [i for i, m in enumerate(labels["meta"]) if m["recording"] in role[split]]
        sub = open_memmap(out / f"{split}_x.npy", mode="w+", dtype=x.dtype,
                          shape=(len(idx),) + x.shape[1:])
        for j, i in enumerate(idx):
            sub[j] = x[i]
        sub.flush()
        torch.save({"traj": labels["traj"][idx], "present": labels["present"][idx],
                    "meta": [labels["meta"][i] for i in idx]}, out / f"{split}_labels.pt")
        counts[split] = len(idx)
        if split == "train":
            arr = np.asarray(sub[:: max(1, len(idx) // 200)], dtype=np.float32)
            torch.save({"mean": float(arr.mean()), "std": float(arr.std())}, out / "stats.pt")
    (out / "DONE").write_text(json.dumps({"fold": k, "counts": counts,
                                          "recordings": {s: sorted(r) for s, r in role.items()}}))
    print(f"wrote {out}: {counts}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", type=int)
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.build is not None:
        build(a.build)


if __name__ == "__main__":
    main()
