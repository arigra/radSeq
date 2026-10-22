"""Build 8-frame RADIal sequence caches (real data) in the scene-cache format.

Same layout as the simulated caches (src/scene_data), so one training path
reads both: a memory-mapped float16 array of dB maps plus linked vehicle
labels. Splits come from the pinned data/radial/split.json, by whole recording.

Normalisation: each cache carries its own mean/std, computed on its training
split only. Real and simulated maps sit at very different absolute levels
(real ~70-146 dB), so normalising each by its own statistics is what lets a
model pretrained on one be fine-tuned on the other.

Resumable: progress is recorded per chunk, like the simulated caches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src import radial
from src.radar_physics import RADIAL_SPEC


def build_split(out, split, sequences, by_frame, seq_len, chunk=20):
    x_path = out / f"{split}_x.npy"
    progress = out / f"{split}_progress.json"
    n = len(sequences)
    shape = (n, seq_len, RADIAL_SPEC.n_range, RADIAL_SPEC.n_doppler)
    done = json.loads(progress.read_text())["done"] if progress.exists() else 0
    x = (open_memmap(x_path, mode="r+") if done else
         open_memmap(x_path, mode="w+", dtype=np.float16, shape=shape))
    parts = out / f"{split}_labelparts"
    parts.mkdir(exist_ok=True)
    for part in parts.glob("*.pt"):
        if int(part.stem) >= done:
            part.unlink()
    while done < n:
        stop = min(done + chunk, n)
        trajs, presents, meta = [], [], []
        for i in range(done, stop):
            name, samples = sequences[i]
            x[i] = radial.load_sequence(samples).numpy().astype(np.float16)
            frames = [[(v["range_bin"], v["doppler_bin"])
                       for v in radial.vehicles(s, by_frame)] for s in samples]
            traj, present = radial.link_vehicles(frames)
            trajs.append(traj)
            presents.append(present)
            meta.append({"recording": name, "samples": samples})
        x.flush()
        torch.save({"traj": torch.stack(trajs), "present": torch.stack(presents),
                    "meta": meta}, parts / f"{done:06d}.pt")
        done = stop
        progress.write_text(json.dumps({"done": done, "n": n}))
        print(f"{split}: {done}/{n}", flush=True)
    chunks = [torch.load(p) for p in sorted(parts.glob("*.pt"))]
    torch.save({"traj": torch.cat([c["traj"] for c in chunks]),
                "present": torch.cat([c["present"] for c in chunks]),
                "meta": [m for c in chunks for m in c["meta"]]},
               out / f"{split}_labels.pt")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=None,
                    help="window stride for TRAIN only (default: non-overlapping)")
    ap.add_argument("--out", default="data/radial_seq")
    ap.add_argument("--split-json", default="data/radial/split.json",
                    help="{split name: [recordings]}; every split in it is built")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = radial.recordings()
    split = json.loads(Path(a.split_json).read_text())
    by_frame = radial.read_labels()
    counts = {}
    for name in split:
        sub = {k: recs[k] for k in split[name]}
        stride = a.stride if name == "train" else None
        seqs = radial.sequences(sub, seq_len=a.seq_len, stride=stride)
        counts[name] = len(seqs)
        print(f"{name}: {len(seqs)} sequences from {len(sub)} recordings", flush=True)
        arr = build_split(out, name, seqs, by_frame, a.seq_len)
        if name == ("train" if "train" in split else next(iter(split))) \
                and not (out / "stats.pt").exists():
            sample = np.asarray(arr[:: max(1, len(seqs) // 200)], dtype=np.float32)
            torch.save({"mean": float(sample.mean()), "std": float(sample.std())},
                       out / "stats.pt")
    (out / "DONE").write_text(json.dumps({"seq_len": a.seq_len, "counts": counts,
                                          "stride": a.stride}))
    print("done", out, torch.load(out / "stats.pt"), flush=True)


if __name__ == "__main__":
    main()
