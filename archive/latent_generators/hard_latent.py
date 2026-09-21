"""Structured learned generator for the full radar-sequence regime.

Learn the accepted target motion distribution from the training cache. Sample
target count and class from their empirical frequencies, then render fresh
target phases, amplitudes, clutter and receiver noise with the radar simulator.
"""
import argparse
from pathlib import Path

import torch
import yaml

from src.diffusion import GaussianDiffusion
from archive.latent_generators.easy_latent import TrajectoryDenoiser
from src.simulator import TemporalRadarSimulator


def load_scene_metadata(cache_dir, split="train"):
    motion, counts, classes = [], [], []
    for path in sorted(Path(cache_dir).glob(f"{split}_*.pt")):
        for item in torch.load(path, map_location="cpu"):
            n = int(item["n_targets"])
            if not 1 <= n <= 5:
                raise ValueError(f"invalid target count {n} in {path}")
            counts.append(n)
            motion.append(torch.stack((item["v0"][:n], item["acc"][:n]), -1))
            classes.append(item["cls"][:n])
    if not counts:
        raise ValueError(f"no {split} sequences in {cache_dir}")
    return torch.cat(motion), torch.tensor(counts), torch.cat(classes)


def train(cache_dir="data/cache", out="checkpoints/hard_latent.pt",
          steps=50000, batch_size=512, seed=2026, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    raw, counts, classes = load_scene_metadata(cache_dir)
    manifest = yaml.safe_load((Path(cache_dir) / "manifest.yaml").read_text())
    mean, std = raw.mean(0), raw.std(0).clamp_min(1e-6)
    data = ((raw - mean) / std).to(device)
    model = TrajectoryDenoiser().to(device)
    diff = GaussianDiffusion(parameterization="v", x0_clamp=None)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in range(1, steps + 1):
        x0 = data[torch.randint(len(data), (batch_size,), device=device)]
        t = torch.randint(diff.T, (batch_size,), device=device)
        eps = torch.randn_like(x0)
        pred = model(diff.q_sample(x0, t, eps), t)
        loss = (pred - diff.target(x0, t, eps)).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == steps:
            print(f"step={step} loss={loss.item():.5f}", flush=True)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "mean": mean, "std": std,
                "count_prob": torch.bincount(counts, minlength=6)[1:6].float() / len(counts),
                "class_prob": torch.bincount(classes, minlength=3).float() / len(classes),
                "steps": steps, "seed": seed, "seq_len": manifest["seq_len"],
                "frame_interval": manifest["frame_interval"],
                "latent_dim": 2, "regime": "full"}, path)
    return path


def _valid_motion(latent, sim, times):
    v0, acc = latent
    if not (sim.v_min + 0.5 <= v0 <= sim.v_max - 0.5
            and -sim.a_max <= acc <= sim.a_max):
        return None
    velocity = v0 + acc * times
    if not ((velocity >= sim.v_min) & (velocity <= sim.v_max)).all():
        return None
    offset = v0 * times + 0.5 * acc * times.square()
    low = max(sim.r_min + 10, float((sim.r_min - offset).max()))
    high = min(sim.r_max - 10, float((sim.r_max - offset).min()))
    if low >= high:
        return None
    r0 = torch.empty(1).uniform_(low, high)[0]
    return r0, v0, acc


@torch.no_grad()
def generate(checkpoint, n, seed=None, device=None, target_count=None):
    if n < 1:
        raise ValueError("n must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location="cpu")
    if state.get("regime") != "full" or state.get("latent_dim") != 2:
        raise ValueError("checkpoint is not a full-regime latent generator")
    model = TrajectoryDenoiser().to(device).eval()
    model.load_state_dict(state["model"])
    diff = GaussianDiffusion(parameterization="v", x0_clamp=None)
    sim = TemporalRadarSimulator(seq_len=state["seq_len"],
                                 frame_interval=state["frame_interval"])
    times = torch.arange(sim.L) * sim.Tf
    if seed is not None:
        torch.manual_seed(seed)
    if target_count is None:
        counts = torch.multinomial(state["count_prob"], n, replacement=True) + 1
    else:
        counts = torch.as_tensor(target_count, dtype=torch.long).reshape(-1)
        if counts.numel() == 1:
            counts = counts.expand(n).clone()
        if counts.numel() != n or ((counts < 1) | (counts > 5)).any():
            raise ValueError("target_count must be an integer in [1, 5] or one per sequence")
    total = int(counts.sum())
    motion = []
    attempts = 0
    while len(motion) < total and attempts < 30 * total:
        batch_size = min(max(64, 3 * (total - len(motion))), 2048)
        z = diff.p_sample_loop(model, (batch_size, 2), device)
        latent = (z * state["std"].to(device) + state["mean"].to(device)).cpu()
        for row in latent:
            attempts += 1
            valid = _valid_motion(row, sim, times)
            if valid is not None:
                motion.append(valid)
            if len(motion) == total:
                break
    if len(motion) != total:
        raise RuntimeError(f"only {len(motion)}/{total} valid motions")
    items, start = [], 0
    for count in counts.tolist():
        current = motion[start:start + count]
        start += count
        r0, v0, acc = (torch.stack([tr[j] for tr in current]) for j in range(3))
        cls = torch.multinomial(state["class_prob"], count, replacement=True)
        items.append(sim.gen_sequence(r0=r0, v0=v0, a=acc, cls=cls))
    from src.dataset import _pad
    items = [_pad(item) for item in items]
    return {key: torch.stack([torch.as_tensor(item[key]) for item in items])
            for key in items[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", default="checkpoints/hard_latent.pt")
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    train(args.cache, args.out, args.steps)
