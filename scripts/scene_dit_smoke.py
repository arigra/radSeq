"""Step 3 recipe check: does the DiT recipe hold at RADIal's 512x256 grid?

Everything so far trained at 64x64. Three short runs on the engineer's scene
cache isolate the two settings most likely to break at 32x more pixels:

  A  patch 16, schedule shift 16   resolution-scaled noise schedule
  B  patch 16, schedule shift 4    the 64x64 setting, as a control
  C  patch 32, schedule shift 16   3x faster, 4x coarser tokens

Diffusion needs more noise at a given step as resolution grows (Hoogeboom et
al. 2023): scaling the side length by ~5.7 suggests a shift near 4 x 5.7 ~ 23,
so 16 is a conservative step from the 64x64 value.

For each run: validation loss, samples conditioned on held-out vehicle
labels, their background statistics against held-out simulated maps, and a
figure. Resumable: finished runs are skipped.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "A_p16_s16": {"patch": 16, "shift": 16.0},
    "B_p16_s4": {"patch": 16, "shift": 4.0},
    "C_p32_s16": {"patch": 32, "shift": 16.0},
    # After 2,250 steps none had learned the DDMA copies; C showed 32x32 block
    # artifacts and was dropped. The two patch-16 settings get a long run each.
    "B_long": {"patch": 16, "shift": 4.0, "epochs": 20, "ema": 0.9995},
    "A_long": {"patch": 16, "shift": 16.0, "epochs": 20, "ema": 0.9995},
}
DEFAULT_RUNS = "A_p16_s16,B_p16_s4,C_p32_s16,B_long,A_long"


def config(name, run, cache, epochs):
    return {
        "data": {"kind": "scene", "cache_dir": cache, "seq_len": 8,
                 "n_range": 512, "n_doppler": 256, "num_workers": 4, "seed": 1234},
        "model": {"patch": run["patch"], "stride": run["patch"], "dim": 384,
                  "depth": 12, "heads": 6, "attn_mode": "factorized",
                  "cond_channels": 2},
        "diffusion": {"timesteps": 1000, "parameterization": "v",
                      "schedule_shift": run["shift"], "x0_clamp": "off",
                      "terminal_x0": "mean"},
        "train": {"seed": 2026, "batch_size": 8, "lr": 1.0e-4, "weight_decay": 0.0,
                  "epochs": run.get("epochs", epochs), "lambda_smooth": 0.0,
                  "lambda_traj": 0.0, "lambda_doppler": 0.0, "amp": "bf16",
                  "ema_decay": run.get("ema", 0.998),
                  "phase": 1, "cond_dropout": 0.1,
                  "ckpt_dir": f"checkpoints/scene_smoke_{name}",
                  "log_file": f"logs/scene_smoke_{name}.log",
                  "log_every_steps": 100, "save_every_steps": 500,
                  "val_every_epochs": 1, "val_batch_size": 8, "val_seed": 4321,
                  "wandb": False},
        # short runs: EMA not warmed up, sample raw; long runs: EMA
        "sample": {"ddim_steps": 30, "weights": "ema" if "ema" in run else "raw"},
    }


def evaluate(name, cache, device, n=8):
    from src import radial
    from src.sample import load_trajectory_conditioned, sample_trajectory_conditioned
    from src.scene_data import SceneSequenceDataset
    weights = "ema" if "ema" in RUNS[name] else "raw"
    model, cfg = load_trajectory_conditioned(
        f"checkpoints/scene_smoke_{name}/last.pt", device, weights=weights)
    val = SceneSequenceDataset(cache, "val", normalise=False)
    labels = {"traj": val.traj[:n], "present": val.present[:n]}
    x = sample_trajectory_conditioned(model, cfg, labels, device, steps=30,
                                      guidance=1.0, seed=1).numpy()
    ref = np.stack([np.asarray(val.x[i], dtype=np.float32) for i in range(n, 2 * n)])
    gen_stats = radial.background_stats(x.reshape(-1, 512, 256))
    ref_stats = radial.background_stats(ref.reshape(-1, 512, 256))
    np.save(ROOT / f"samples/scene_smoke_{name}_samples.npy", x[:4, :4].astype(np.float16))
    return {"generated": gen_stats, "simulated": ref_stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/scene_engineer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="samples/scene_dit_smoke.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    out = ROOT / a.out
    report = json.loads(out.read_text()) if out.exists() else {}
    for name in a.runs.split(","):
        if name in report:
            print(f"{name} done, skipping", flush=True)
            continue
        cfg_path = ROOT / f"configs/scene_smoke_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(config(name, RUNS[name], a.cache, a.epochs),
                                           sort_keys=False))
        ckpt = ROOT / f"checkpoints/scene_smoke_{name}"
        cmd = [sys.executable, "-m", "src.train", "--config", str(cfg_path)]
        if (ckpt / "last.pt").exists() and not (ckpt / "STAGE_DONE").exists():
            cmd += ["--resume", str(ckpt / "last.pt")]
        if not (ckpt / "STAGE_DONE").exists():
            subprocess.run(cmd, check=True, cwd=ROOT)
            (ckpt / "STAGE_DONE").touch()
        vals = [l for l in (ROOT / f"logs/scene_smoke_{name}.log").read_text().splitlines()
                if "validation" in l]
        report[name] = {"run": RUNS[name], "last_validation": vals[-1] if vals else None,
                        **evaluate(name, a.cache, device)}
        out.write_text(json.dumps(report, indent=2))
        print(name, json.dumps(report[name]["generated"]), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
