"""Research plan step 2 (redesigned): does synthetic data help a target-CLASS detector?

The position-only task was saturated (brightness alone scores AP 0.94). Here
the detector must also say which class each target is (steady, Swerling-1,
extended), which needs the whole sequence and is where synthetic data could
help or hurt (its class fidelity was never tested).

Arms (3 seeds, identical budget): real_n, real_n_synth, synth_only, real_full.
Detector: src/detector.SequenceClassDetector. Metric: mean over classes of AP,
a detection correct only in its class channel within 2 bins of a true target
of that class, on held-out val sequences.
Rule (docs/notes/2026-09-17-step2-synthetic-augmentation.md):
  informative only if mean mAP(real_full) - mean mAP(real_n) > larger seed range of those two;
  if informative, synthetic data helps if mean mAP(real_n_synth) - mean mAP(real_n)
  > larger seed range of those two.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.detector import SequenceClassDetector, average_precision, class_heatmap_targets, detect, focal_loss
from src.sample import resolve_cache_dir

N_CLASSES = 3
CLASS_NAMES = ("steady", "swerling1", "extended")


def pool_from_items(items):
    return [{"x": it["x"], "traj": it["traj"], "n": it["n_targets"], "cls": it["cls"]} for it in items]


def pool_from_synthetic(folder):
    pool = []
    for path in sorted(Path(folder).glob("part_*.pt")):
        part = torch.load(path, map_location="cpu")
        for i in range(len(part["x"])):
            pool.append({"x": part["x"][i], "traj": part["traj"][i], "n": part["n_targets"][i],
                         "cls": part["cls"][i]})
    return pool


def batch_from_pool(pool, batch, stats, device, generator):
    seq = torch.randint(len(pool), (batch,), generator=generator).tolist()
    x = torch.stack([pool[i]["x"].float() for i in seq])
    traj = torch.stack([pool[i]["traj"] for i in seq]).to(device)
    n = torch.stack([pool[i]["n"] for i in seq]).to(device)
    cls = torch.stack([pool[i]["cls"] for i in seq]).to(device)
    return ((x - stats["mean"]) / stats["std"]).to(device), class_heatmap_targets(traj, n, cls)


def train_detector(pool, stats, device, seed, steps, batch):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = SequenceClassDetector().to(device)
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
    dets = {c: [] for c in range(N_CLASSES)}
    tgts = {c: [] for c in range(N_CLASSES)}
    for it in test_items:
        x = ((it["x"].float() - stats["mean"]) / stats["std"]).to(device)[None]
        prob = torch.sigmoid(model(x).float())[0].cpu()                 # (3, 16, 64, 64)
        m = int(it["n_targets"])
        for c in range(N_CLASSES):
            dets[c] += detect(prob[c])
            keep = it["cls"][:m] == c
            tgts[c] += [torch.round(it["traj"][:m][keep, l].float()) for l in range(16)]
    per_class = {CLASS_NAMES[c]: average_precision(dets[c], tgts[c], radius=2.0)
                 for c in range(N_CLASSES) if sum(len(t) for t in tgts[c])}
    return sum(per_class.values()) / len(per_class), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="data/synthetic/cond_traj_n2000")
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--test-offset", type=int, default=1000)
    ap.add_argument("--test-seqs", type=int, default=512)
    ap.add_argument("--out", default="samples/detector_class_augmentation.json")
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
    report = {"task": "target class (steady / swerling1 / extended), mean AP over classes",
              "n_real": args.n_real, "n_synthetic": len(synth), "steps": args.steps,
              "batch_sequences": args.batch,
              "test": f"val {args.test_offset}-{args.test_offset + args.test_seqs - 1}", "arms": {}}
    for name, pool in arms.items():
        maps, per_class = [], []
        for seed in seeds:
            model = train_detector(pool, stats, device, seed, args.steps, args.batch)
            m, pc = evaluate(model, test_items, stats, device)
            maps.append(m)
            per_class.append(pc)
            print(f"{name} seed {seed}: mAP {m:.4f} {json.dumps({k: round(v, 3) for k, v in pc.items()})}",
                  flush=True)
            del model
        report["arms"][name] = {"pool_sequences": len(pool), "map": maps, "per_class": per_class,
                                "mean": sum(maps) / len(maps), "range": max(maps) - min(maps)}
        Path(args.out).write_text(json.dumps(report, indent=2))
    A, B, C = (report["arms"][k] for k in ("real_n", "real_n_synth", "real_full"))
    headroom, headroom_bar = C["mean"] - A["mean"], max(A["range"], C["range"])
    gain, gain_bar = B["mean"] - A["mean"], max(A["range"], B["range"])
    informative = headroom > headroom_bar
    report["rule"] = {"headroom": headroom, "headroom_bar": headroom_bar, "informative": informative,
                      "gain": gain, "gain_bar": gain_bar, "helps": informative and gain > gain_bar}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("RULE", json.dumps(report["rule"]), flush=True)

    if args.note:
        rows = "\n".join(
            f"| {name} | {r['pool_sequences']} | {', '.join(f'{v:.3f}' for v in r['map'])} | "
            f"{r['mean']:.3f} | {r['range']:.3f} |" for name, r in report["arms"].items())
        rule = report["rule"]
        verdict = ("UNINFORMATIVE: 20k real sequences do not beat 2,000 by more than the seed spread, "
                   "so this task has no headroom either." if not informative else
                   "synthetic data from the conditional DiT HELPS the class detector." if rule["helps"] else
                   "synthetic data does NOT measurably help the class detector.")
        text = (f"\n## Result, target-class task (`{args.out}`)\n\nMean AP over classes (within 2 bins, "
                f"correct class) on {report['test']}, {args.steps} steps x {args.batch} sequences per arm.\n\n"
                "| arm | training sequences | mAP per seed | mean mAP | seed range |\n|---|---:|---|---:|---:|\n"
                + rows + "\n\n"
                f"Headroom (real_full - real_n): {headroom:+.3f} vs bar {headroom_bar:.3f}. "
                f"Gain (real_n_synth - real_n): {gain:+.3f} vs bar {gain_bar:.3f}.\n\n"
                f"**Verdict (pre-set rule): {verdict}**\n")
        with open(args.note, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
