"""E0 memorisation / diversity check for a trained checkpoint.

Each sequence is reduced to its peak path: per-frame argmax (range, Doppler)
bin, fitted with a straight line -> features (r0, dr, d0, dd).

Memorisation: nearest-neighbour distance from GENERATED sequences to the TRAIN
set, compared with the same distance for held-out VAL sequences, both on peak
path features and on raw normalised maps. A generator that copies training
sequences sits much closer to train than held-out data does.
Diversity: spread of the path features, generated vs val.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.sample import select_checkpoint_state
from src.train import build_model


def path_features(x_db):
    B, L, N, K = x_db.shape
    idx = x_db.reshape(B, L, -1).argmax(-1)
    r, d = (idx // K).float(), (idx % K).float()
    t = torch.arange(L, dtype=torch.float32)
    A = torch.stack([torch.ones(L), t], 1)
    pinv = torch.linalg.pinv(A)                      # (2, L)
    return torch.cat([r @ pinv.T, d @ pinv.T], 1)    # (B, 4): r0, dr, d0, dd


@torch.no_grad()
def nn_dist(queries, bank, device, chunk=2000):
    q = queries.to(device).flatten(1)
    best = torch.full((len(q),), float("inf"), device=device)
    for s in range(0, len(bank), chunk):
        b = bank[s:s + chunk].to(device).flatten(1)
        d2 = q.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1) - 2 * q @ b.T
        best = torch.minimum(best, d2.clamp_min(0).min(1).values)
    return best.sqrt().cpu()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e0_standard_bs32/last.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="samples/e0_diversity.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg, dc = ck["config"], ck["config"]["diffusion"]
    diff = GaussianDiffusion(dc["timesteps"], x0_clamp=dc.get("x0_clamp", DEFAULT_X0_CLAMP),
                             parameterization=dc.get("parameterization", "eps"),
                             terminal_x0=dc.get("terminal_x0", "model"),
                             schedule_shift=dc.get("schedule_shift", 1.0))
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ck, weights="ema"))
    model.eval()
    train = RadarSequenceDataset(cfg["data"]["cache_dir"], "train")
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    st = train.stats
    gen = []
    torch.manual_seed(7)
    for s in range(0, args.n, 64):
        b = min(64, args.n - s)
        gen.append(denormalize(diff.ddim_sample(model, (b, 16, 64, 64), device, steps=args.steps).float().cpu(), st))
    gen = torch.cat(gen)
    real_val = torch.stack([val.items[i]["x"] for i in range(args.n)]).float()
    z = lambda x: (x - st["mean"]) / st["std"]
    # The train set (20k x 16 x 64 x 64) held twice exceeds the job's RAM limit,
    # so path features and pixel distances are computed per chunk.
    f_tr, pix_gen, pix_val = [], None, None
    for s in range(0, len(train.items), 2000):
        chunk = torch.stack([it["x"] for it in train.items[s:s + 2000]]).float()
        f_tr.append(path_features(chunk))
        zc = z(chunk)
        g, v = nn_dist(z(gen), zc, device), nn_dist(z(real_val), zc, device)
        pix_gen = g if pix_gen is None else torch.minimum(pix_gen, g)
        pix_val = v if pix_val is None else torch.minimum(pix_val, v)
        del chunk, zc
    del train
    f_tr = torch.cat(f_tr)
    f_gen, f_val = path_features(gen), path_features(real_val)
    feat_gen, feat_val = nn_dist(f_gen, f_tr, device), nn_dist(f_val, f_tr, device)
    names = ["r0", "dr", "d0", "dd"]
    report = {
        "checkpoint": args.ckpt, "n": args.n, "ddim_steps": args.steps, "weights": "ema",
        "path_nn_to_train_median": {"generated": float(feat_gen.median()), "val": float(feat_val.median())},
        "pixel_nn_to_train_median": {"generated": float(pix_gen.median()), "val": float(pix_val.median())},
        "pixel_nn_ratio_generated_over_val": float(pix_gen.median() / pix_val.median()),
        "path_feature_std": {"generated": dict(zip(names, f_gen.std(0).tolist())),
                             "val": dict(zip(names, f_val.std(0).tolist()))},
        "distinct_generated_paths": int(torch.unique(f_gen.round(), dim=0).shape[0]),
    }
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
