"""Step 3's question, on real radar: does a generator beat its own simulator?

Three arms train the same vehicle detector on the same budget and are scored on
held-out RADIal TEST recordings:

  A  real            the real training sequences alone
  B  real + sim      plus raw sequences from the simulator
  D  real + DiT      plus sequences from a DiT pretrained on that simulator and
                     fine-tuned on the same real sequences

B and D add the same number of sequences, so the comparison isolates where the
extra data came from, not how much of it there is.

Claim under test: D > B. If a generator only reproduces its simulator, D = B.
A gate refuses to score a generator whose vehicles do not land on their labels.

Every source is normalised by its own statistics, as in DiT training: real maps
sit near 97 dB and simulated ones near 12 dB, so normalising simulated frames
by the real statistics put them ~15 std off the real data. Generated maps come
out in real dB and use the real statistics.

Resumable: the report is rewritten after every detector, and a rerun skips the
seeds it already holds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.detector import detect, focal_loss
from src.detector_scene import (RadialVehicleDetector, average_precision_wrapped,
                                scene_heatmap_targets)
from src.scene_data import SceneSequenceDataset

MIN_HIT_RATE = 0.5


def synthetic_pool(folder, stats):
    """Generated sequences, in the dB of the real data they were fine-tuned on."""
    pool = []
    for path in sorted(Path(folder).glob("part_*.pt")):
        part = torch.load(path, map_location="cpu")
        for i in range(len(part["x"])):
            pool.append({"x": part["x"][i], "traj": part["traj"][i],
                         "present": part["present"][i], "stats": stats})
    return pool


def dataset_pool(cache, split, limit=None):
    ds = SceneSequenceDataset(cache, split, normalise=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    return [{"x": torch.from_numpy(np.array(ds.x[i], dtype=np.float16)),
             "traj": ds.traj[i], "present": ds.present[i], "stats": ds.stats}
            for i in range(n)]


def normalised(item, t):
    return (item["x"][t].float() - item["stats"]["mean"]) / item["stats"]["std"]


def hit_rate(pool, radius=2.0, n=64):
    """Share of requested vehicles that are actually a local peak in the map."""
    hits = total = 0
    for item in pool[:n]:
        x = item["x"].float().numpy()
        for m in range(item["traj"].shape[0]):
            for t in range(item["traj"].shape[1]):
                if not item["present"][m, t]:
                    continue
                total += 1
                r, d = [int(round(float(v))) for v in item["traj"][m, t]]
                if not (2 <= r < x.shape[1] - 2):
                    continue
                win = x[t, r - 2:r + 3][:, [(d + k) % x.shape[2] for k in range(-2, 3)]]
                ring = np.median(x[t, r - 2:r + 3, :])
                hits += int(win.max() > ring + 3.0)
    return hits / max(total, 1)


def batch_from(pool, batch, device, generator):
    """One random frame per batch element, normalised by its own source.

    The detector is per-frame, and a batch of whole 8-frame 512x256 sequences
    does not fit in GPU memory.
    """
    idx = torch.randint(len(pool), (batch,), generator=generator).tolist()
    n_frames = pool[0]["x"].shape[0]
    frames = torch.randint(n_frames, (batch,), generator=generator).tolist()
    x = torch.stack([normalised(pool[i], t) for i, t in zip(idx, frames)])[:, None]
    traj = torch.stack([pool[i]["traj"][:, t] for i, t in zip(idx, frames)])[:, :, None]
    present = torch.stack([pool[i]["present"][:, t] for i, t in zip(idx, frames)])[:, :, None]
    return x.to(device), scene_heatmap_targets(traj.to(device), present.to(device))


def train_detector(pool, device, seed, steps, batch, width):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RadialVehicleDetector(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for step in range(steps):
        x, heat = batch_from(pool, batch, device, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = focal_loss(logits, heat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"    step {step}: loss {float(loss.detach()):.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, test_pool, device):
    model.eval()
    dets, tgts = [], []
    for item in test_pool:
        probs = []
        for t in range(item["x"].shape[0]):  # frame by frame: the whole sequence
            frame = normalised(item, t)[None, None].to(device)  # does not fit at once
            probs.append(torch.sigmoid(model(frame).float())[0, 0].cpu())
        dets += detect(torch.stack(probs), threshold=0.05, top_k=10)
        for t in range(item["traj"].shape[1]):
            keep = item["present"][:, t]
            tgts.append(item["traj"][keep, t].float())
    return average_precision_wrapped(dets, tgts, 256, radius=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("engineer", "fitted"))
    ap.add_argument("--real", default="data/radial_seq")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8, help="frames per step")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(Path(a.real) / "stats.pt")

    real = dataset_pool(a.real, "train")
    test = dataset_pool(a.real, "test")
    n_extra = len(real) * a.repeats
    sim = dataset_pool(f"data/scene_{a.variant}", "train", limit=n_extra)
    synth = synthetic_pool(a.synthetic, stats)
    print(f"real {len(real)}  sim {len(sim)}  synthetic {len(synth)}  test {len(test)}",
          flush=True)

    rate = hit_rate(synth)
    print(f"generated hit rate: {rate:.2f}", flush=True)
    if rate < MIN_HIT_RATE:
        raise SystemExit(f"generator places only {rate:.0%} of vehicles on their "
                         f"labels (< {MIN_HIT_RATE:.0%}); refusing to score it")

    arms = {"real": real, "real_sim": real + sim, "real_dit": real + synth}
    seeds = [int(s) for s in a.seeds.split(",")]
    settings = {"variant": a.variant, "n_real": len(real), "n_sim": len(sim),
                "n_synthetic": len(synth), "n_test": len(test), "steps": a.steps,
                "batch": a.batch, "width": a.width, "seeds": seeds,
                "normalisation": "per source"}
    out = Path(a.out)
    report = json.loads(out.read_text()) if out.exists() else {**settings, "arms": {}}
    stale = {k: (report.get(k), v) for k, v in settings.items() if report.get(k) != v}
    if stale:
        raise SystemExit(f"{out} holds a different experiment {stale}; move it away first")
    report["generated_hit_rate"] = rate

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out)

    for name, pool in arms.items():
        arm = report["arms"].setdefault(name, {"pool": len(pool), "ap": []})
        for seed in seeds[len(arm["ap"]):]:
            model = train_detector(pool, device, seed, a.steps, a.batch, a.width)
            ap_value = evaluate(model, test, device)
            arm["ap"].append(ap_value)
            arm["mean"] = sum(arm["ap"]) / len(arm["ap"])
            arm["range"] = max(arm["ap"]) - min(arm["ap"])
            print(f"{name} seed {seed}: AP {ap_value:.4f}", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    A, B, D = (report["arms"][k] for k in ("real", "real_sim", "real_dit"))
    bar = max(A["range"], B["range"], D["range"])
    report["rule"] = {
        "bar (largest seed range)": bar,
        "sim helps (B - A)": B["mean"] - A["mean"],
        "dit helps (D - A)": D["mean"] - A["mean"],
        "dit beats sim (D - B)": D["mean"] - B["mean"],
        "verdict": ("D > B: the generator beats its own simulator"
                    if D["mean"] - B["mean"] > bar else
                    "D = B within the seed spread" if abs(D["mean"] - B["mean"]) <= bar
                    else "B > D: raw simulator data is better")}
    save()
    print(json.dumps(report["rule"], indent=2), flush=True)


if __name__ == "__main__":
    main()
