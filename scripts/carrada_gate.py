"""CARRADA simulator gate (docs/design/2026-10-26-carrada-simulator-design.md):
a detector trained on simulator sequences alone must reach AP >= 0.1 on CARRADA's
real validation recordings before any generator work. A real-only detector
(real train -> real val) is trained the same way as a reference, not a gate.

Same protocol as the RADIal car gate (scripts/run_car_gate.sh): per-frame U-Net,
3,000 steps of 8 frames, seeds 1-3, each source normalised by its own
statistics, class-agnostic AP within 2 bins, Doppler wrapping. Grid 256 x 64.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.radial_detector_arms import dataset_pool, normalised
from src.detector import detect, focal_loss
from src.detector_scene import RadialVehicleDetector, average_precision_wrapped, scene_heatmap_targets

NR, ND = 256, 64


def batch(pool, n, device, g):
    idx = torch.randint(len(pool), (n,), generator=g).tolist()
    fr = torch.randint(8, (n,), generator=g).tolist()
    x = torch.stack([normalised(pool[i], t) for i, t in zip(idx, fr)])[:, None]
    traj = torch.stack([pool[i]["traj"][:, t] for i, t in zip(idx, fr)])[:, :, None]
    pres = torch.stack([pool[i]["present"][:, t] for i, t in zip(idx, fr)])[:, :, None]
    return x.to(device), scene_heatmap_targets(traj.to(device), pres.to(device), n_range=NR, n_doppler=ND)


@torch.no_grad()
def ap(model, pool, device):
    model.eval()
    dets, tgts = [], []
    for it in pool:
        probs = torch.sigmoid(model(torch.stack([normalised(it, t) for t in range(8)])[:, None].to(device)).float())[:, 0].cpu()
        dets += detect(probs, threshold=0.05, top_k=10, circular_doppler=True)
        tgts += [it["traj"][it["present"][:, t], t].float() for t in range(8)]
    return average_precision_wrapped(dets, tgts, ND, radius=2.0)


def train(pool, device, seed, steps=3000):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = RadialVehicleDetector(width=32).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        x, h = batch(pool, 8, device, g)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = focal_loss(m(x), h)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sim", default="data/carrada_sim_gate")
    a.add_argument("--real", default="data/carrada_seq")
    a.add_argument("--out", default="samples/carrada_gate.json")
    a = a.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim, real, val = dataset_pool(a.sim, "train"), dataset_pool(a.real, "train"), dataset_pool(a.real, "val")
    out = Path(a.out)
    rep = json.loads(out.read_text()) if out.exists() else {"n_sim": len(sim), "n_real_train": len(real), "n_val": len(val), "arms": {}}
    for name, pool in (("sim_only", sim), ("real_only_reference", real)):
        aps = rep["arms"].setdefault(name, [])
        for seed in (1, 2, 3)[len(aps):]:
            aps.append(ap(train(pool, device, seed), val, device))
            print(name, seed, f"AP {aps[-1]:.3f}", flush=True)
            out.write_text(json.dumps(rep, indent=2))
    s = rep["arms"]["sim_only"]
    rep["gate"] = {"sim_only_mean": sum(s) / len(s), "bar": 0.1, "passed": sum(s) / len(s) >= 0.1}
    out.write_text(json.dumps(rep, indent=2))
    print(rep["gate"])


if __name__ == "__main__":
    main()
