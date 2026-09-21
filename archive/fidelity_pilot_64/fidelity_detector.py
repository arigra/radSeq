"""Detector arms for the fidelity pilot (archive/fidelity_pilot_64/fidelity_pilot.py).

Three arms trained on an identical budget and evaluated on held-out REALITY
sequences, so any difference is in the training pool only:

  real        the small real set alone
  real_sim    real + the same number of raw sequences from the wrong simulator
  real_synth  real + the same number from a DiT pretrained on that simulator
              and fine-tuned on the same real set

real_sim and real_synth are matched in pool size, so the comparison isolates
where the extra sequences came from rather than how many there are.

Task and metric follow experiments/step2_detector/detector_class_augmentation.py: target class
within 2 bins, mean AP over classes.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.detector import (SequenceClassDetector, average_precision,
                          class_heatmap_targets, detect, focal_loss)
from experiments.step2_detector.detector_class_augmentation import (CLASS_NAMES, N_CLASSES,
                                                 pool_from_items,
                                                 pool_from_synthetic,
                                                 train_detector, evaluate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reality", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--test-seqs", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reality = Path(args.reality)
    stats = torch.load(reality / "stats.pt", map_location="cpu")

    real_items = RadarSequenceDataset(reality, "train").items[:args.n_real]
    test_items = RadarSequenceDataset(reality, "val").items[:args.test_seqs]
    real = pool_from_items(real_items)

    n_extra = args.n_real * args.repeats
    sim_items = RadarSequenceDataset(Path(args.sim), "train").items[:n_extra]
    raw_sim = pool_from_items(sim_items)
    synth = pool_from_synthetic(args.synthetic)

    arms = {"real": real,
            "real_sim": real + raw_sim,
            "real_synth": real + synth}
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"task": "target class, mean AP over classes",
              "n_real": args.n_real, "n_extra_sim": len(raw_sim),
              "n_extra_synth": len(synth), "steps": args.steps,
              "test_sequences": len(test_items), "arms": {}}
    for name, pool in arms.items():
        maps, per_class = [], []
        for seed in seeds:
            model = train_detector(pool, stats, device, seed, args.steps, args.batch)
            m, pc = evaluate(model, test_items, stats, device)
            maps.append(m)
            per_class.append(pc)
            print(f"{name} seed {seed}: mAP {m:.4f} "
                  f"{json.dumps({k: round(v, 3) for k, v in pc.items()})}", flush=True)
            del model
        report["arms"][name] = {"pool_sequences": len(pool), "map": maps,
                                "per_class": per_class,
                                "mean": sum(maps) / len(maps),
                                "range": max(maps) - min(maps)}
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
