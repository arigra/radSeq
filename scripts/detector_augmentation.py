"""Research plan step 2: does conditional-DiT synthetic data help a detector?

Arms (3 detector seeds each, identical training budget):
  real_n        first N real training sequences
  real_n_synth  the same N real + synthetic sequences rendered from their labels
  synth_only    synthetic sequences only (reported, not in the rule)
  real_full     all 20k real training sequences (upper bound)
Detector: src/detector.HeatmapDetector on single normalised frames. Metric:
average precision within 2 bins on held-out validation sequences.
Rule (docs/notes/2026-09-17-step2-synthetic-augmentation.md): synthetic data
helps if mean AP(real_n_synth) - mean AP(real_n) exceeds the larger of the two
arms' seed ranges (max - min).
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.detector import HeatmapDetector, average_precision, detect, focal_loss, heatmap_targets
from src.sample import resolve_cache_dir


def pool_from_items(items):
    return [{"x": it["x"], "traj": it["traj"], "n": it["n_targets"]} for it in items]


def pool_from_synthetic(folder):
    pool = []
    for path in sorted(Path(folder).glob("part_*.pt")):
        part = torch.load(path, map_location="cpu")
        for i in range(len(part["x"])):
            pool.append({"x": part["x"][i], "traj": part["traj"][i], "n": part["n_targets"][i]})
    return pool


def batch_from_pool(pool, batch, stats, device, generator):
    seq = torch.randint(len(pool), (batch,), generator=generator)
    frame = torch.randint(16, (batch,), generator=generator)
    x = torch.stack([pool[i]["x"][l].float() for i, l in zip(seq.tolist(), frame.tolist())])
    traj = torch.stack([pool[i]["traj"] for i in seq.tolist()])
    n = torch.stack([pool[i]["n"] for i in seq.tolist()])
    heat = heatmap_targets(traj.to(device), n.to(device))                 # (B, 16, 64, 64)
    heat = heat[torch.arange(batch, device=device), frame.to(device)]
    return ((x - stats["mean"]) / stats["std"]).to(device), heat


def train_detector(pool, stats, device, seed, steps, batch):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = HeatmapDetector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        x, heat = batch_from_pool(pool, batch, stats, device, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = focal_loss(logits, heat)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, test_items, stats, device):
    model.eval()
    detections, targets = [], []
    for it in test_items:
        x = ((it["x"].float() - stats["mean"]) / stats["std"]).to(device)   # (16, 64, 64)
        prob = torch.sigmoid(model(x).float()).cpu()
        detections += detect(prob)
        m = int(it["n_targets"])
        targets += [torch.round(it["traj"][:m, l].float()) for l in range(16)]
    return average_precision(detections, targets, radius=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="data/synthetic/cond_traj_n2000")
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--test-offset", type=int, default=1000)
    ap.add_argument("--test-seqs", type=int, default=512)
    ap.add_argument("--out", default="samples/detector_augmentation.json")
    ap.add_argument("--note")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = resolve_cache_dir(args.cache)
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    train_items = RadarSequenceDataset(cache, "train").items
    test_items = RadarSequenceDataset(cache, "val").items[args.test_offset:args.test_offset + args.test_seqs]
    real_n = pool_from_items(train_items[:args.n_real])
    synth = pool_from_synthetic(args.synthetic)
    arms = {"real_n": real_n, "real_n_synth": real_n + synth, "synth_only": synth,
            "real_full": pool_from_items(train_items)}
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"n_real": args.n_real, "n_synthetic": len(synth), "steps": args.steps,
              "batch_frames": args.batch, "test": f"val {args.test_offset}-"
              f"{args.test_offset + args.test_seqs - 1}", "arms": {}}
    for name, pool in arms.items():
        aps = []
        for seed in seeds:
            model = train_detector(pool, stats, device, seed, args.steps, args.batch)
            aps.append(evaluate(model, test_items, stats, device))
            print(f"{name} seed {seed}: AP {aps[-1]:.4f}", flush=True)
            del model
        report["arms"][name] = {"pool_sequences": len(pool), "ap": aps,
                                "mean": sum(aps) / len(aps), "range": max(aps) - min(aps)}
        Path(args.out).write_text(json.dumps(report, indent=2))
    a, b = report["arms"]["real_n"], report["arms"]["real_n_synth"]
    gain, bar = b["mean"] - a["mean"], max(a["range"], b["range"])
    report["rule"] = {"gain": gain, "bar": bar, "helps": gain > bar}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("RULE", json.dumps(report["rule"]), flush=True)

    if args.note:
        rows = "\n".join(
            f"| {name} | {r['pool_sequences']} | {', '.join(f'{v:.3f}' for v in r['ap'])} | "
            f"{r['mean']:.3f} | {r['range']:.3f} |" for name, r in report["arms"].items())
        text = (f"\n## Result (`{args.out}`)\n\nDetector AP within 2 bins on {report['test']} "
                f"({args.test_seqs * 16} frames), {args.steps} steps x {args.batch} frames per arm.\n\n"
                "| arm | training sequences | AP per seed | mean AP | seed range |\n"
                "|---|---:|---|---:|---:|\n" + rows + "\n\n"
                f"Gain of real_n_synth over real_n: {gain:+.3f}; bar (larger seed range): {bar:.3f}.\n\n"
                "**Verdict (pre-set rule): "
                + ("synthetic data from the conditional DiT HELPS the detector at this data size.**"
                   if report["rule"]["helps"] else
                   "synthetic data does NOT measurably help the detector at this data size.**") + "\n")
        with open(args.note, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
