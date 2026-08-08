"""Control experiment: does spatial attention restore the bright tail?

Compares, at a matched 12500-step budget and matched data/seed:

  baseline      temporal-only  depth 8   9.83M   checkpoints/phase1_wandb/epoch_0010.pt
  control d5    factorized     depth 5   8.58M   (-12.7% params vs baseline)
  control d6    factorized     depth 6  10.22M   (+4.0%  params vs baseline)

The depth-5 arm carries the argument: it has FEWER parameters than the
baseline, so if it restores the tail the result cannot be attributed to
capacity.

Pre-registered decision rule (fixed before the runs, see the notes doc):
  generated std > 0.85 (from the baseline's 0.655, against real 1.003)
  in either arm  ->  diagnosis confirmed.
  Both arms near 0.66  ->  diagnosis incomplete; reopen the investigation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.diffusion import GaussianDiffusion
from src.eval.metrics import evaluate_sequences
from src.train import build_model

DECISION_THRESHOLD = 0.85


def load_raw_val(cache_dir, limit):
    seqs = []
    for f in sorted(Path(cache_dir).glob("val_*.pt")):
        for it in torch.load(f):
            seqs.append(it["x"])
            if len(seqs) >= limit:
                return torch.stack(seqs)
    return torch.stack(seqs)


def tail_stats(xn):
    f = xn.flatten().float()
    sub = f if f.numel() <= 16_000_000 else f[:: f.numel() // 16_000_000 + 1]
    q = torch.quantile(sub, torch.tensor([0.5, 0.75, 0.99, 0.999]))
    return {"std": round(float(f.std()), 4), "mean": round(float(f.mean()), 4),
            "p50": round(float(q[0]), 4), "p75": round(float(q[1]), 4),
            "p99": round(float(q[2]), 4), "p999": round(float(q[3]), 4)}


def score(ckpt_path, n, steps, real_raw, mean, std, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    torch.manual_seed(1)                      # same noise draw for every arm
    gn = diff.ddim_sample(model, (n, cfg["data"]["seq_len"], 64, 64), device,
                          steps=steps).cpu()
    out = {
        "checkpoint": str(ckpt_path),
        "attn_mode": cfg["model"].get("attn_mode", "temporal"),
        "depth": cfg["model"]["depth"],
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "epoch": ckpt.get("epoch"), "step": ckpt.get("step"),
        "generated": tail_stats(gn),
        "metrics": evaluate_sequences(gn * std + mean, real_raw),
    }
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="samples/control_comparison.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(Path(args.cache) / "stats.pt")
    mean, std = stats["mean"], stats["std"]
    real_raw = load_raw_val(args.cache, 2 * args.n)
    ref_a, ref_b = real_raw[:args.n], real_raw[args.n:]
    real_norm = (ref_b - mean) / std

    report = {"decision_threshold_std": DECISION_THRESHOLD,
              "real": tail_stats(real_norm),
              "real_reference_metrics": evaluate_sequences(ref_a, ref_b)}
    print("real:", report["real"])

    arms = [("baseline_temporal_d8", "checkpoints/phase1_wandb/epoch_0010.pt"),
            ("control_factorized_d5", "checkpoints/ctrl_factorized_d5/last.pt"),
            ("control_factorized_d6", "checkpoints/ctrl_factorized_d6/last.pt")]
    report["arms"] = {}
    for name, path in arms:
        if not Path(path).exists():
            print(f"  {name}: MISSING ({path}) — skipped")
            continue
        print(f"  sampling {name}...")
        report["arms"][name] = score(path, args.n, args.steps, ref_b,
                                     mean, std, device)
        a = report["arms"][name]
        print(f"    {a['params_M']:.2f}M  step {a['step']}  "
              f"std {a['generated']['std']:.3f}  p99 {a['generated']['p99']:.3f}  "
              f"L1 {a['metrics']['marginal_l1']:.4f}")

    base = report["arms"].get("baseline_temporal_d8", {}).get("generated", {}).get("std")
    passed = {k: v["generated"]["std"] > DECISION_THRESHOLD
              for k, v in report["arms"].items() if k.startswith("control")}
    report["verdict"] = {
        "baseline_std": base,
        "real_std": report["real"]["std"],
        "arms_over_threshold": passed,
        "diagnosis_confirmed": any(passed.values()) if passed else None,
        "d5_confirms_not_capacity": passed.get("control_factorized_d5"),
    }

    print("\n=== VERDICT ===")
    print(json.dumps(report["verdict"], indent=2))
    print(f"\n{'arm':<26}{'params':>9}{'std':>8}{'p99':>8}{'L1':>9}"
          f"{'vel_cons':>10}{'persist':>9}")
    r = report["real"]; rm = report["real_reference_metrics"]
    print(f"{'REAL':<26}{'-':>9}{r['std']:>8.3f}{r['p99']:>8.3f}"
          f"{'-':>9}{rm['velocity_consistency']:>10.3f}{rm['persistence']:>9.3f}")
    for k, v in report["arms"].items():
        g, m = v["generated"], v["metrics"]
        print(f"{k:<26}{v['params_M']:>8.2f}M{g['std']:>8.3f}{g['p99']:>8.3f}"
              f"{m['marginal_l1']:>9.4f}{m['velocity_consistency']:>10.3f}"
              f"{m['persistence']:>9.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
