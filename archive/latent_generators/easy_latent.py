"""Learn the E0 trajectory distribution, then render radar maps from it.

The pixel-space DiT can denoise an observed target but rarely creates one from
noise. E0 has only three scene variables (initial range, velocity and
acceleration); modelling those variables directly keeps target existence and
motion explicit while retaining a learned, unconditional generator.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from src.diffusion import GaussianDiffusion
from src.dit import timestep_embedding
from src.simulator import TemporalRadarSimulator


class TrajectoryDenoiser(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 64, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x, t, cond=None):
        return self.net(torch.cat((x, timestep_embedding(t, 64)), dim=-1))


def load_kinematics(cache_dir, split="train"):
    """Read the three E0 latent variables without retaining the large RD maps."""
    rows = []
    for path in sorted(Path(cache_dir).glob(f"{split}_*.pt")):
        for item in torch.load(path, map_location="cpu"):
            if int(item["n_targets"]) != 1 or int(item["cls"][0]) != 0:
                raise ValueError(f"{path} is not one-target steady E0 data")
            rows.append(torch.stack((item["traj"][0, 0, 0] * 3.0,
                                     item["v0"][0], item["acc"][0])))
    if not rows:
        raise ValueError(f"no {split} sequences in {cache_dir}")
    return torch.stack(rows)


def train(cache_dir="data/cache_easy", out="checkpoints/easy_latent.pt",
          steps=5000, batch_size=512, seed=2026, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    # Under the E0 simulator, initial range is uniform on the feasible
    # interval conditional on velocity and acceleration. Learn only the
    # correlated (velocity, acceleration) law and sample range exactly.
    raw = load_kinematics(cache_dir)[:, 1:]
    manifest = yaml.safe_load((Path(cache_dir) / "manifest.yaml").read_text())
    mean, std = raw.mean(0), raw.std(0).clamp_min(1e-6)
    data = ((raw - mean) / std).to(device)
    model = TrajectoryDenoiser().to(device)
    diff = GaussianDiffusion(parameterization="v", x0_clamp=None)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
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
                "steps": steps, "seed": seed, "seq_len": manifest["seq_len"],
                "frame_interval": manifest["frame_interval"],
                "latent_dim": 2}, path)
    return path


@torch.no_grad()
def generate(checkpoint, n, seed=None, device=None, ddim_steps=50,
             sampler="ddpm"):
    """Sample learned E0 kinematics and render exactly one target per scene."""
    if n < 1:
        raise ValueError("n must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location="cpu")
    if state.get("latent_dim") != 2:
        raise ValueError("checkpoint must use the two-variable E0 latent format")
    model = TrajectoryDenoiser().to(device).eval()
    model.load_state_dict(state["model"])
    diff = GaussianDiffusion(parameterization="v", x0_clamp=None)
    if seed is not None:
        torch.manual_seed(seed)
    sim = TemporalRadarSimulator(seq_len=state["seq_len"],
                                 frame_interval=state["frame_interval"],
                                 max_targets=1, force_class=0, scnr=20,
                                 clutter=False, noise=False)
    mean, std = state["mean"].to(device), state["std"].to(device)
    items = []
    attempts = 0
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler must be 'ddpm' or 'ddim'")
    while len(items) < n and attempts < max(100, 20 * n):
        count = min(max(32, n - len(items)), 512)
        z = (diff.p_sample_loop(model, (count, 2), device) if sampler == "ddpm"
             else diff.ddim_sample(model, (count, 2), device, steps=ddim_steps))
        latents = (z * std + mean).cpu()
        times = torch.arange(state["seq_len"]) * state["frame_interval"]
        for v0, acc in latents:
            attempts += 1
            if not (sim.v_min + 0.5 <= v0 <= sim.v_max - 0.5
                    and -sim.a_max <= acc <= sim.a_max):
                continue
            velocity = v0 + acc * times
            if not ((velocity >= sim.v_min) & (velocity <= sim.v_max)).all():
                continue
            offset = v0 * times + 0.5 * acc * times.square()
            low = max(sim.r_min + 10, float((sim.r_min - offset).max()))
            high = min(sim.r_max - 10, float((sim.r_max - offset).min()))
            if low >= high:
                continue
            r0 = torch.empty(1).uniform_(low, high)[0]
            try:
                item = sim.gen_sequence(r0=[r0], v0=[v0], a=[acc])
            except ValueError:
                continue  # reject trajectories outside the radar grid
            items.append(item)
            if len(items) == n:
                break
    if len(items) != n:
        raise RuntimeError(f"only {len(items)}/{n} valid trajectories after {attempts} draws")
    return {key: torch.stack([torch.as_tensor(item[key]) for item in items])
            for key in items[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache_easy")
    parser.add_argument("--out", default="checkpoints/easy_latent.pt")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    train(args.cache, args.out, args.steps)
