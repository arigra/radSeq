"""Sim-to-sim rehearsal: does generative fine-tuning substitute for simulator fidelity?

One simulator configuration is designated "reality". Others are deliberately
wrong by a controlled amount (a dB offset on target brightness -- the dominant
mismatch measured against RADIal, where the specification-driven simulator's
targets were +39 dB too prominent). For each mismatch delta we compare:

  A          real only                                (independent of delta)
  B(delta)   real + raw simulator data, no generator
  D(delta)   real + data from a DiT pretrained on the wrong simulator and
             fine-tuned on the same small real set

Prediction under test: B degrades strongly with delta while D stays flat,
i.e. the generator absorbs the sim-to-real gap. The interaction is the result;
if D(delta) tracks B(delta) the mechanism is dead.

Rehearsing sim-to-sim lets the mismatch be dialled continuously and the effect
size measured before committing days of GPU time to RADIal.

Every stage is skipped when its output exists, so the run resumes after a
killed session.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def ensure_cache(path, n_train, n_val, seed, gain_offset_db):
    """Generate a dataset, or keep the one already there."""
    if (Path(path) / "stats.pt").exists():
        print(f"cache {path} exists, skipping", flush=True)
        return
    from src.dataset import generate_cache
    print(f"generating {path} (offset {gain_offset_db:+.1f} dB)", flush=True)
    generate_cache(path, n_train, n_val, seq_len=16, seed=seed,
                   shard_size=1000, gain_offset_db=gain_offset_db)


def train_stage(cfg_path, ckpt_dir):
    """Train once, resuming a partial run and recording completion.

    `last.pt` existing does NOT mean the run finished -- it is written
    periodically -- so a separate marker records completion and an existing
    checkpoint is resumed from rather than skipped.
    """
    ckpt_dir = Path(ckpt_dir)
    done = ckpt_dir / "STAGE_DONE"
    if done.exists():
        print(f"{ckpt_dir} already complete, skipping", flush=True)
        return
    cmd = [PY, "-m", "src.train", "--config", cfg_path]
    last = ckpt_dir / "last.pt"
    if last.exists():
        print(f"resuming {ckpt_dir} from {last}", flush=True)
        cmd += ["--resume", str(last)]
    run(cmd)
    done.touch()


MIN_HIT_RATE = 0.7


def synthetic_hit_rate(folder, n=128):
    """Share of requested targets the rendered sequences actually contain."""
    from src.eval.adherence import trajectory_adherence
    part = torch.load(Path(folder) / "part_0.pt", map_location="cpu")
    return trajectory_adherence(part["x"][:n].float(), part["traj"][:n],
                                part["n_targets"][:n])["hit_rate"]


def write_config(path, **over):
    """A conditional-DiT config sized for a pilot rather than a final model."""
    cfg = {
        "data": {"n_train": 20000, "n_val": 2000, "seq_len": 16,
                 "frame_interval": 0.5, "cache_dir": "data/cache",
                 "shard_size": 1000, "seed": 1234},
        "model": {"patch": 8, "stride": 8, "dim": 256, "depth": 8, "heads": 8,
                  "attn_mode": "factorized", "cond_channels": 4},
        "diffusion": {"timesteps": 1000, "parameterization": "v",
                      "schedule_shift": 4.0, "x0_clamp": "off",
                      "terminal_x0": "model"},
        "train": {"seed": 2026, "batch_size": 32, "lr": 1.0e-4,
                  "weight_decay": 0.0, "epochs": 40, "lambda_smooth": 0.0,
                  "lambda_traj": 0.01, "lambda_doppler": 0.01, "amp": "bf16",
                  # 0.999 suits a ~7k-step pilot; 0.9999 left 47% of the
                  # random init in the EMA and broke every sample drawn from it
                  "ema_decay": 0.999, "phase": 1, "cond_dropout": 0.1,
                  "log_every_steps": 200, "save_every_steps": 1000,
                  "val_every_epochs": 10, "val_batch_size": 32,
                  "val_seed": 4321, "wandb": False},
        "sample": {"ddim_steps": 30, "weights": "ema"},
    }
    for section, values in over.items():
        cfg[section].update(values)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="0,4,8,16",
                    help="simulator mismatch in dB of target brightness")
    ap.add_argument("--n-real", type=int, default=200,
                    help="real sequences for fine-tuning and for the detector")
    ap.add_argument("--n-sim", type=int, default=6000)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--detector-steps", type=int, default=2500)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="samples/fidelity_pilot.json")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    reality = "data/pilot/reality"
    ensure_cache(reality, 2000, 800, seed=99, gain_offset_db=0.0)

    results = {}
    if Path(args.out).exists():
        results = json.loads(Path(args.out).read_text()).get("deltas", {})

    for delta in deltas:
        key = f"{delta:g}"
        if key in results and results[key].get("done"):
            print(f"delta {key} dB already done, skipping", flush=True)
            continue
        tag = f"d{key.replace('-', 'm')}"
        sim_cache = f"data/pilot/sim_{tag}"
        ensure_cache(sim_cache, args.n_sim, 400, seed=7, gain_offset_db=delta)

        # 1. pretrain the conditional DiT on the WRONG simulator
        pre_ckpt = Path(f"checkpoints/pilot_pre_{tag}")
        cfg = write_config(
            f"configs/pilot_pre_{tag}.yaml",
            data={"cache_dir": sim_cache, "n_train": args.n_sim, "n_val": 400},
            train={"epochs": args.pretrain_epochs,
                   "ckpt_dir": str(pre_ckpt),
                   "log_file": f"logs/pilot_pre_{tag}.log"})
        train_stage(cfg, pre_ckpt)

        # 2. fine-tune on the small REAL set
        ft_ckpt = Path(f"checkpoints/pilot_ft_{tag}")
        cfg = write_config(
            f"configs/pilot_ft_{tag}.yaml",
            data={"cache_dir": reality, "n_train": 2000, "n_val": 800,
                  "train_subset": args.n_real},
            train={"epochs": args.finetune_epochs,
                   "init_from": str(pre_ckpt / "last.pt"),
                   # raw weights: a short pretraining's EMA is not warmed up
                   "init_weights": "raw",
                   # ~1,800 fine-tuning steps need a short EMA memory
                   "ema_decay": 0.995,
                   "ckpt_dir": str(ft_ckpt),
                   "log_file": f"logs/pilot_ft_{tag}.log"})
        train_stage(cfg, ft_ckpt)

        # 3. render synthetic sequences from the fine-tuned generator
        synth = f"data/pilot/synth_{tag}"
        run([PY, "scripts/gen_synthetic.py", "--ckpt", ft_ckpt / "last.pt",
             "--n-labels", args.n_real, "--repeats", args.repeats,
             "--guidance", 1.0, "--out", synth])

        # 3b. gate: the synthetic labels must be right before a detector learns
        # from them. The first pilot run fed the D arm sequences whose targets
        # landed on their labels 8-16% of the time, and it looked like a result.
        hit = synthetic_hit_rate(synth)
        print(f"synthetic hit rate for delta {key}: {hit:.2f}", flush=True)
        if hit < MIN_HIT_RATE:
            raise SystemExit(f"generator for delta {key} puts only {hit:.0%} of "
                             f"targets where requested (< {MIN_HIT_RATE:.0%}); "
                             "refusing to train detectors on its output")

        # 4. detector arms
        out = f"samples/pilot_detector_{tag}.json"
        if not Path(out).exists():
            run([PY, "scripts/fidelity_detector.py", "--reality", reality,
                 "--sim", sim_cache, "--synthetic", synth,
                 "--n-real", args.n_real, "--repeats", args.repeats,
                 "--steps", args.detector_steps, "--seeds", args.seeds,
                 "--out", out])
        results[key] = {"delta_db": delta, "done": True,
                        "synthetic_hit_rate": hit,
                        "detector": json.loads(Path(out).read_text())}
        Path(args.out).write_text(json.dumps(
            {"knob": "target brightness offset (dB)", "n_real": args.n_real,
             "n_sim": args.n_sim, "deltas": results}, indent=2))
        print(f"delta {key} dB complete", flush=True)

    print("\n=== fidelity pilot summary ===", flush=True)
    print(f"{'delta dB':>9} {'A real':>9} {'B raw sim':>10} {'D dit':>9} {'D-B':>7}")
    for key in sorted(results, key=float):
        d = results[key]["detector"]["arms"]
        a, b, dd = d["real"]["mean"], d["real_sim"]["mean"], d["real_synth"]["mean"]
        print(f"{key:>9} {a:9.3f} {b:10.3f} {dd:9.3f} {dd - b:+7.3f}")


if __name__ == "__main__":
    main()
