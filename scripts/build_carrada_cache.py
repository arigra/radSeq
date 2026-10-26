"""8-frame CARRADA caches in the scene-cache format (src/scene_data.py), plus a
per-object class: real recordings (`--real`, CARRADA's own recording-level
Train/Validation/Test split) or the engineer's simulator (`--sim`).

Maps are dB of power summed over channels, float16; each cache carries mean/std
of its training split. Resumable by chunk, like build_scene_cache.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src import carrada
from src.carrada_sim import CarradaSimulator


def write_split(out, split, n, make, chunk=50):
    """make(i) -> (x (8,256,64), traj, present, cls, meta)."""
    progress = out / f"{split}_progress.json"
    done = json.loads(progress.read_text())["done"] if progress.exists() else 0
    shape = (n, 8, carrada.N_RANGE, carrada.N_DOPPLER)
    x = open_memmap(out / f"{split}_x.npy", mode="r+" if done else "w+", dtype=np.float16, shape=shape)
    parts = out / f"{split}_labelparts"
    parts.mkdir(exist_ok=True)
    for p in parts.glob("*.pt"):
        if int(p.stem) >= done:
            p.unlink()
    while done < n:
        stop = min(done + chunk, n)
        tr, pr, cl, me = [], [], [], []
        for i in range(done, stop):
            xi, t, p, c, m = make(i)
            x[i] = xi.astype(np.float16)
            tr.append(t); pr.append(p); cl.append(c); me.append(m)
        x.flush()
        torch.save({"traj": torch.stack(tr), "present": torch.stack(pr), "cls": torch.stack(cl),
                    "meta": me}, parts / f"{done:06d}.pt")
        done = stop
        progress.write_text(json.dumps({"done": done, "n": n}))
        print(f"{split}: {done}/{n}", flush=True)
    ch = [torch.load(p) for p in sorted(parts.glob("*.pt"))]
    torch.save({k: torch.cat([c[k] for c in ch]) for k in ("traj", "present", "cls")}
               | {"meta": [m for c in ch for m in c["meta"]]}, out / f"{split}_labels.pt")
    return x


def stats(out, x):
    a = np.asarray(x[:: max(1, len(x) // 200)], dtype=np.float32)
    torch.save({"mean": float(a.mean()), "std": float(a.std())}, out / "stats.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.real:
        recs, split, ann = carrada.recordings(), carrada.official_split(), carrada.annotations()
        seqs = carrada.sequences(recs)
        for name, tag in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
            mine = [s for s in seqs if split[s[0]] == tag]
            def make(i, mine=mine):
                rec, frames = mine[i]
                t, p, c = carrada.pack([carrada.frame_labels(rec, f, ann) for f in frames], 8)
                return carrada.load_sequence(rec, frames).numpy(), t, p, c, {"recording": rec, "samples": frames}
            arr = write_split(out, name, len(mine), make)
            if name == "train":
                stats(out, arr)
    if a.sim:
        for name, n, seed0 in (("train", a.n_train, 10_000_000), ("val", a.n_val, 20_000_000)):
            def make(i, seed0=seed0):
                o = CarradaSimulator(generator=torch.Generator().manual_seed(seed0 + i)).gen_sequence()
                fl = [[{"iid": lab["id"], "range_bin": lab["range_bin"], "doppler_bin": lab["doppler_bin"],
                        "cls": lab["cls"]} for lab in f] for f in o["labels"]]
                t, p, c = carrada.pack(fl, 8)
                return o["x"].numpy(), t, p, c, {"classes": o["classes"]}
            arr = write_split(out, name, n, make)
            if name == "train":
                stats(out, arr)
    (out / "DONE").write_text(json.dumps({"real": a.real, "sim": a.sim}))
    print("done", out, torch.load(out / "stats.pt"))


if __name__ == "__main__":
    main()
