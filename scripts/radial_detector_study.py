"""Step 6 detector study: with 50 real sequences, does DiT data beat simulator data?

Pre-registered in docs/notes/2026-10-22-step6-fixed-protocol.md. It replaces
step 4's protocol (scripts/radial_detector_arms.py, kept for reproducibility),
which was biased against any extra data:
- step 4 mixed extra data uniformly (80% of every batch); here mix arms draw
  half of each batch from real data and half from the extra source
- pretrain-then-fine-tune arms train on the extra source, then on real data
- every stage trains until AP on real VAL stops improving (val every 250
  steps, patience 4, at most 6,000 steps) and keeps its best weights, instead
  of a fixed 3,000 steps that gave larger pools fewer passes over real data
- the DiT renders the simulator's own label sets (`dit`), so it adds new scene
  layouts and the DiT and simulator arms differ only in how the same scenes
  look; `dit_rl` (the real training labels, as in step 4) is a secondary arm
- test detections are saved per arm and seed, for bootstrap confidence
  intervals over test recordings (scripts/step6_score.py)

--detector sequence (step 7) swaps the per-frame detector for one that reads
all 8 frames at once (src.detector_scene.RadialSequenceDetector), so the study
also tests whether the generated sequences behave like real ones over time.
Batches are then whole sequences: --batch / 4 of them (8 -> 2 sequences, 16
frames).

Resumable per (arm, seed). Real-only detections do not depend on the
simulator variant and are shared between the variants' reports.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.radial_detector_arms import batch_from, dataset_pool, normalised, synthetic_pool
from src.detector import detect, focal_loss
from src.detector_scene import (RadialSequenceDetector, RadialVehicleDetector,
                                average_precision_wrapped, scene_heatmap_targets)

# arm -> stages; a stage is (kind, source): "real" = real batches, "mix" =
# half real + half source, "only" = source alone, "ft" = real batches at the
# fine-tuning learning rate, from the previous stage's best weights
ARMS = {
    "real":      [("real", None)],
    "sim_mix":   [("mix", "sim")],
    "dit_mix":   [("mix", "dit")],
    "ditrl_mix": [("mix", "dit_rl")],
    "sim_pt":    [("only", "sim"), ("ft", None)],
    "dit_pt":    [("only", "dit"), ("ft", None)],
    "sim_only":  [("only", "sim")],
    "dit_only":  [("only", "dit")],
}
PRIMARY = ("real", "sim_mix", "dit_mix")
LR, LR_FT = 1e-3, 3e-4
VAL_EVERY, PATIENCE, MAX_STEPS = 250, 4, 6000
N_VAL = 32


SEQUENCE = False      # set by --detector sequence


def seq_batch_from(pool, n, device, generator):
    """n whole sequences, each normalised by its own source."""
    idx = torch.randint(len(pool), (n,), generator=generator).tolist()
    x = torch.stack([torch.stack([normalised(pool[i], t) for t in range(pool[i]["x"].shape[0])])
                     for i in idx])
    traj = torch.stack([pool[i]["traj"] for i in idx])
    present = torch.stack([pool[i]["present"] for i in idx])
    return x.to(device), scene_heatmap_targets(traj.to(device), present.to(device))


def draw_from(pool, n, device, generator):
    return (seq_batch_from(pool, max(n // 4, 1), device, generator) if SEQUENCE
            else batch_from(pool, n, device, generator))


@torch.no_grad()
def detections(model, pool, device):
    model.eval()
    dets, tgts, recs = [], [], []
    for item in pool:
        if SEQUENCE:
            x = torch.stack([normalised(item, t) for t in range(item["x"].shape[0])])
            probs = list(torch.sigmoid(model(x[None].to(device)).float())[0].cpu())
        else:
            probs = [torch.sigmoid(model(normalised(item, t)[None, None].to(device)).float())[0, 0].cpu()
                     for t in range(item["x"].shape[0])]
        dets += detect(torch.stack(probs), threshold=0.05, top_k=10, circular_doppler=True)
        for t in range(item["traj"].shape[1]):
            tgts.append(item["traj"][item["present"][:, t], t].float())
            recs.append(item.get("recording"))
    model.train()
    return dets, tgts, recs


def ap(model, pool, device):
    dets, tgts, _ = detections(model, pool, device)
    return average_precision_wrapped(dets, tgts, 256, radius=2.0)


def run_stage(model, lr, draw, val, device, gen, log):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    best, best_state, best_step, bad = -1.0, None, 0, 0
    for step in range(1, MAX_STEPS + 1):
        x, heat = draw(gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = focal_loss(logits, heat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % VAL_EVERY == 0:
            v = ap(model, val, device)
            if v > best:
                best, best_step, bad = v, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
            log(f"    step {step}: loss {float(loss):.4f}  val AP {v:.4f}  best {best:.4f}@{best_step}")
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"val_ap": best, "best_step": best_step, "stopped_at": step}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--variant", required=True, choices=("engineer", "fitted"))
    ap_.add_argument("--real", required=True, help="real cache (its train, val, test)")
    ap_.add_argument("--sim-cache", help="default data/scene_<variant>")
    ap_.add_argument("--dit", required=True, help="DiT renderings of the simulator labels")
    ap_.add_argument("--dit-rl", help="DiT renderings of the real training labels")
    ap_.add_argument("--repeats", type=int, default=4, help="extra sequences per real one")
    ap_.add_argument("--arms", default=",".join(ARMS))
    ap_.add_argument("--seeds-primary", default="1,2,3,4,5")
    ap_.add_argument("--seeds-other", default="1,2,3")
    ap_.add_argument("--batch", type=int, default=8)
    ap_.add_argument("--width", type=int, default=32)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--dets-dir", default="samples/step6_dets")
    ap_.add_argument("--detector", default="frame", choices=("frame", "sequence"))
    a = ap_.parse_args()
    global SEQUENCE
    SEQUENCE = a.detector == "sequence"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(Path(a.real) / "stats.pt")
    names = a.arms.split(",")

    real = dataset_pool(a.real, "train")
    val = dataset_pool(a.real, "val", limit=N_VAL)
    test = dataset_pool(a.real, "test")
    n_extra = len(real) * a.repeats
    sources = {"sim": dataset_pool(a.sim_cache or f"data/scene_{a.variant}", "train", limit=n_extra),
               "dit": synthetic_pool(a.dit, stats)}
    if a.dit_rl:
        sources["dit_rl"] = synthetic_pool(a.dit_rl, stats)
    for k, v in sources.items():
        if k != "sim" and len(v) != n_extra:
            raise SystemExit(f"{k}: {len(v)} sequences, expected {n_extra}")
    print({k: len(v) for k, v in {"real": real, "val": val, "test": test, **sources}.items()},
          flush=True)

    out, dets_dir = Path(a.out), Path(a.dets_dir)
    dets_dir.mkdir(parents=True, exist_ok=True)
    settings = {"variant": a.variant, "real_cache": a.real, "n_real": len(real),
                "n_extra": n_extra, "batch": a.batch, "width": a.width, "lr": LR,
                "lr_ft": LR_FT, "val_every": VAL_EVERY, "patience": PATIENCE,
                "max_steps": MAX_STEPS, "n_val": len(val), "n_test": len(test),
                "detector": a.detector}
    report = json.loads(out.read_text()) if out.exists() else {**settings, "arms": {}}
    stale = {k: (report.get(k), v) for k, v in settings.items() if report.get(k) != v}
    if stale:
        raise SystemExit(f"{out} holds a different experiment {stale}; move it away first")

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out)

    for name in names:
        seeds = [int(s) for s in (a.seeds_primary if name in PRIMARY else a.seeds_other).split(",")]
        arm = report["arms"].setdefault(name, {"stages": ARMS[name], "runs": {}})
        for seed in seeds:
            # real-only does not depend on the variant: one set of detections for both
            tag = ("seq_" if SEQUENCE else "") + ("" if name == "real" else f"{a.variant}_")
            det_path = dets_dir / f"{tag}{Path(a.real).name}_{name}_s{seed}.pt"
            if str(seed) in arm["runs"] and det_path.exists():
                continue
            if det_path.exists():
                saved = torch.load(det_path)
                arm["runs"][str(seed)] = saved["run"]
                save()
                continue
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = (RadialSequenceDetector(width=a.width) if SEQUENCE
                     else RadialVehicleDetector(width=a.width)).to(device)
            run = {"stages": []}
            for kind, src in ARMS[name]:
                half = a.batch // 2
                if kind in ("real", "ft"):
                    draw = lambda g: draw_from(real, a.batch, device, g)
                elif kind == "only":
                    draw = lambda g, s=sources[src]: draw_from(s, a.batch, device, g)
                else:
                    def draw(g, s=sources[src]):
                        x1, h1 = draw_from(real, half, device, g)
                        x2, h2 = draw_from(s, a.batch - half, device, g)
                        return torch.cat([x1, x2]), torch.cat([h1, h2])
                lr = LR_FT if kind == "ft" else LR
                run["stages"].append(run_stage(
                    model, lr, draw, val, device, gen,
                    lambda m: print(f"{name} seed {seed} {kind}{m}", flush=True)))
            dets, tgts, recs = detections(model, test, device)
            run["test_ap"] = average_precision_wrapped(dets, tgts, 256, radius=2.0)
            torch.save({"dets": dets, "tgts": tgts, "recs": recs, "run": run}, det_path)
            arm["runs"][str(seed)] = run
            aps = [r["test_ap"] for r in arm["runs"].values()]
            arm["test_ap_mean"], arm["test_ap_range"] = float(np.mean(aps)), float(np.ptp(aps))
            print(f"{name} seed {seed}: test AP {run['test_ap']:.4f}", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
        aps = [r["test_ap"] for r in arm["runs"].values()]
        arm["test_ap_mean"], arm["test_ap_range"] = float(np.mean(aps)), float(np.ptp(aps))
    def n_seeds(n):
        return len((a.seeds_primary if n in PRIMARY else a.seeds_other).split(","))
    report["complete"] = all(len(report["arms"][n]["runs"]) == n_seeds(n) for n in names)
    save()


if __name__ == "__main__":
    main()
