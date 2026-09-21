"""Render a training cache of 8-frame RADIal-grid sequences from one simulator variant.

Resumable: sequences are written into a pre-allocated memory-mapped file in
chunks, and a progress file records how many are done, so a killed session
continues where it stopped. Each sequence has its own seed, so the cache is
identical however many times it is interrupted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.radar_physics import RADIAL_SPEC
from src.scene_data import pack_labels
from src.scene_sim import VARIANTS, SceneSimulator


def render_split(out, split, n, seq_len, scenario, device, seed0, chunk=50):
    x_path = out / f"{split}_x.npy"
    progress = out / f"{split}_progress.json"
    shape = (n, seq_len, RADIAL_SPEC.n_range, RADIAL_SPEC.n_doppler)
    done = json.loads(progress.read_text())["done"] if progress.exists() else 0
    x = (open_memmap(x_path, mode="r+") if done else
         open_memmap(x_path, mode="w+", dtype=np.float16, shape=shape))
    parts_dir = out / f"{split}_labelparts"
    parts_dir.mkdir(exist_ok=True)
    # Chunks at or beyond the resume point are re-rendered below; drop their
    # old label parts so the assembled labels match the maps one-to-one.
    for part in parts_dir.glob("*.pt"):
        start = int(part.stem)
        if start >= done:
            part.unlink()
        else:
            p = torch.load(part)
            if start + len(p["meta"]) > done:          # straddles the resume point
                keep = done - start
                torch.save({"traj": p["traj"][:keep], "present": p["present"][:keep],
                            "meta": p["meta"][:keep]}, part)
    while done < n:
        stop = min(done + chunk, n)
        trajs, presents, metas = [], [], []
        for i in range(done, stop):
            sim = SceneSimulator(scenario=scenario, seq_len=seq_len, device=device,
                                 generator=torch.Generator().manual_seed(seed0 + i))
            o = sim.gen_sequence()
            x[i] = o["x"].numpy().astype(np.float16)
            t, p = pack_labels(o["labels"], seq_len)
            trajs.append(t); presents.append(p)
            metas.append({"road_type": o["road_type"], "ego_speed": o["ego_speed"]})
        x.flush()
        torch.save({"traj": torch.stack(trajs), "present": torch.stack(presents),
                    "meta": metas}, parts_dir / f"{done:06d}.pt")
        done = stop
        progress.write_text(json.dumps({"done": done, "n": n}))
        print(f"{split}: {done}/{n}", flush=True)
    parts = [torch.load(p) for p in sorted(parts_dir.glob("*.pt"))]
    torch.save({"traj": torch.cat([p["traj"] for p in parts]),
                "present": torch.cat([p["present"] for p in parts]),
                "meta": [m for p in parts for m in p["meta"]]},
               out / f"{split}_labels.pt")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out or f"data/scene_{a.variant}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario = VARIANTS[a.variant]()
    # disjoint seed ranges so train and val never share a scene
    train = render_split(out, "train", a.n_train, a.seq_len, scenario, device, 10_000_000)
    render_split(out, "val", a.n_val, a.seq_len, scenario, device, 20_000_000)
    if not (out / "stats.pt").exists():
        sample = np.asarray(train[:: max(1, a.n_train // 500)], dtype=np.float32)
        torch.save({"mean": float(sample.mean()), "std": float(sample.std())},
                   out / "stats.pt")
    (out / "DONE").write_text(json.dumps({"variant": a.variant,
                                          "n_train": a.n_train, "n_val": a.n_val,
                                          "seq_len": a.seq_len}))
    print("done", out, torch.load(out / "stats.pt"), flush=True)


if __name__ == "__main__":
    main()
