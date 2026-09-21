"""Why does generated full-data persistence fall short (0.096 vs 0.144)?

1. Noise floor: persistence of disjoint 32-sequence real validation subsets,
   and per-seed generated values, so the failure is judged against sampling noise.
2. Track anatomy: tracks from the evaluator's own detector/linker
   (src/eval/metrics: 12 dB over frame median, top-5 peaks, 3-bin gate),
   bucketed by length, with mean peak dB and Doppler bin; real tracks are also
   matched to ground-truth target trajectories (<= 2 bins on >= half the frames).
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.eval.metrics import detect_peaks, link_tracks, persistence
from src.sample import select_checkpoint_state
from src.train import build_model

BUCKETS = ((1, 2), (3, 7), (8, 12), (13, 16))


def tracks_of(seq):
    return link_tracks([detect_peaks(f, max_peaks=5) for f in seq])


def anatomy(x_db, trajs=None, n_targets=None):
    out = {f"{a}-{b}": {"count": 0, "db": [], "doppler": [], "matched": 0} for a, b in BUCKETS}
    pers = []
    for i, seq in enumerate(x_db):
        trs = tracks_of(seq)
        pers.append(persistence(trs, len(seq)))
        for tr in trs:
            L = len(tr)
            key = next(f"{a}-{b}" for a, b in BUCKETS if a <= L <= b)
            vals = [float(seq[l, int(r), int(d)]) for l, (r, d) in tr]
            out[key]["count"] += 1
            out[key]["db"].append(sum(vals) / L)
            out[key]["doppler"].append(sum(d for _, (_, d) in tr) / L)
            if trajs is not None:
                hits = 0
                for l, (r, d) in tr:
                    tgt = trajs[i, :int(n_targets[i]), l]          # (M, 2) bins
                    if len(tgt) and float(((tgt - torch.tensor([r, d])).norm(dim=1)).min()) <= 2.0:
                        hits += 1
                out[key]["matched"] += int(hits >= L / 2)
    n = len(x_db)
    summary = {}
    for k, v in out.items():
        db = torch.tensor(v["db"]) if v["db"] else torch.zeros(1)
        dop = torch.tensor(v["doppler"]) if v["doppler"] else torch.zeros(1)
        summary[k] = {"per_seq": v["count"] / n,
                      "target_matched_per_seq": (v["matched"] / n) if trajs is not None else None,
                      "mean_db": float(db.mean()), "doppler_bin_q10_50_90":
                      [float(q) for q in torch.quantile(dop, torch.tensor([0.1, 0.5, 0.9]))]}
    return summary, pers


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e3_standard_bs32/last.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="experiments/dit_64/results/diag_persistence.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg, dc = ck["config"], ck["config"]["diffusion"]
    diff = GaussianDiffusion(dc["timesteps"], x0_clamp=dc.get("x0_clamp", DEFAULT_X0_CLAMP),
                             parameterization=dc.get("parameterization", "eps"),
                             terminal_x0=dc.get("terminal_x0", "model"),
                             schedule_shift=dc.get("schedule_shift", 1.0))
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    st = val.stats
    items = val.items[:args.n]
    real = torch.stack([it["x"] for it in items]).float()
    trajs = torch.stack([it["traj"] for it in items]).float()
    n_t = torch.stack([it["n_targets"] for it in items])

    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ck, weights="ema"))
    model.eval()
    gen = []
    torch.manual_seed(11)
    for s in range(0, args.n, 64):
        gen.append(denormalize(diff.ddim_sample(model, (min(64, args.n - s), 16, 64, 64), device,
                                                steps=args.steps).float().cpu(), st))
    gen = torch.cat(gen)

    real_anat, real_pers = anatomy(real, trajs, n_t)
    gen_anat, gen_pers = anatomy(gen)
    chunk = lambda p: [sum(p[i:i + 32]) / 32 for i in range(0, len(p) - 31, 32)]
    # persistence is a per-sequence ratio averaged over sequences (evaluate_sequences)
    rc, gc = torch.tensor(chunk(real_pers)), torch.tensor(chunk(gen_pers))
    report = {
        "checkpoint": args.ckpt, "weights": "ema", "ddim_steps": args.steps, "n": args.n,
        "persistence_32seq_subsets": {
            "real": {"values": rc.tolist(), "mean": float(rc.mean()), "sd": float(rc.std())},
            "generated": {"values": gc.tolist(), "mean": float(gc.mean()), "sd": float(gc.std())}},
        "tracks_by_length": {"real": real_anat, "generated": gen_anat},
    }
    print(json.dumps({k: report[k] for k in ("persistence_32seq_subsets", "tracks_by_length")}, indent=1))
    Path(args.out).write_text(json.dumps(report, indent=2))
    torch.save(gen[:64], "experiments/dit_64/results/diag_persistence_generated.pt")


if __name__ == "__main__":
    main()
