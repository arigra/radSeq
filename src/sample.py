"""Sample sequences from a checkpoint; write viz + metrics."""
import argparse
from pathlib import Path

import torch
import yaml

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion, diffusion_from_config
from src.train import build_model


def generate_easy(n_seq, seed=None, seq_len=16, frame_interval=0.5):
    """Sample the known one-target, noise-free E0 distribution exactly.

    This is a reference generator, independent of the learned denoiser.  It
    returns dB maps in the same format as ``generate``.
    """
    from src.simulator import generate_sequences

    return generate_sequences(
        n=n_seq, seq_len=seq_len, frame_interval=frame_interval,
        n_targets=1, target_class="steady", snr_db=20.0,
        clutter=False, noise=False, seed=seed)["x"]


def select_checkpoint_state(ckpt, component="model", weights=None):
    """Select raw or EMA weights, honoring the checkpoint's sample config."""
    if weights is None:
        weights = ckpt.get("config", {}).get("sample", {}).get("weights", "raw")
    if weights == "auto":
        weights = "ema" if ckpt.get(f"ema_{component}") is not None else "raw"
    if weights == "raw":
        key = component
    elif weights == "ema":
        key = f"ema_{component}"
    else:
        raise ValueError(f"unknown checkpoint weights {weights!r}")
    state = ckpt.get(key)
    if state is None:
        raise ValueError(f"checkpoint has no {weights} weights for {component}")
    return state


def _set_patch_reduction(model, reduction):
    if reduction is None:
        return
    if reduction not in ("mean", "tile", "hann"):
        raise ValueError(f"unknown patch reduction {reduction!r}")
    if not hasattr(model, "patch_reduction"):
        raise ValueError("patch reduction applies only to patch-based DiT models")
    model.patch_reduction = reduction


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_cache_dir(cache_dir):
    """Checkpoint configs store 'data/cache' relative to the repo; resolve it
    from the repo root so sampling also works from notebooks/."""
    path = Path(cache_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def generate(ckpt_path, n_seq, device, steps=50, cond=None, weights=None,
             patch_reduction=None, seed=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ckpt, weights=weights))
    _set_patch_reduction(model, patch_reduction)
    model.eval()
    diff = diffusion_from_config(cfg["diffusion"])
    L = cfg["data"]["seq_len"]
    if seed is not None:
        # Seed after model construction so different architectures receive the
        # same initial diffusion noise in paired comparisons.
        torch.manual_seed(seed)
    x = diff.ddim_sample(model, (n_seq, L, 64, 64), device, steps=steps, cond=cond)
    stats = torch.load(resolve_cache_dir(cfg["data"]["cache_dir"]) / "stats.pt",
                       map_location="cpu")
    return denormalize(x.float().cpu(), stats)


def load_trajectory_conditioned(ckpt_path, device, weights="ema"):
    """Load a trajectory-conditioned checkpoint once: (model, config)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ckpt, weights=weights))
    model.eval()
    if not getattr(model, "cond_channels", 0):
        raise ValueError(f"{ckpt_path} is not a trajectory-conditioned checkpoint")
    return model, cfg


@torch.no_grad()
def sample_trajectory_conditioned(model, cfg, labels, device, steps=30, guidance=2.0, seed=None):
    """Guided sampling with an already loaded model; returns dB maps (B, L, N, K).

    Classifier-free guidance on the network output:
    uncond + guidance * (cond - uncond), where uncond sees an all-zero
    condition map.
    """
    from src.trajectory_condition import render_condition, render_vehicle_condition

    diff = diffusion_from_config(cfg["diffusion"])
    N, K = cfg["data"].get("n_range", 64), cfg["data"].get("n_doppler", 64)
    if "present" in labels:          # RADIal-grid scenes: vehicle positions only
        cond_map = render_vehicle_condition(labels["traj"].to(device),
                                            labels["present"].to(device),
                                            n_range=N, n_doppler=K)
    else:
        cond_map = render_condition(labels["traj"].to(device),
                                    labels["n_targets"].to(device),
                                    labels["cls"].to(device))
    null = torch.zeros_like(cond_map)

    def guided(xt, t, _cond=None):
        out = model(torch.cat([xt, xt]), torch.cat([t, t]), None,
                    cond_map=torch.cat([cond_map, null]))
        with_cond, without = out.chunk(2)
        return without + guidance * (with_cond - without)

    B, L = labels["traj"].shape[0], cfg["data"]["seq_len"]
    if seed is not None:
        torch.manual_seed(seed)
    x = diff.ddim_sample(guided, (B, L, N, K), device, steps=steps)
    stats = torch.load(resolve_cache_dir(cfg["data"]["cache_dir"]) / "stats.pt",
                       map_location="cpu")
    return denormalize(x.float().cpu(), stats)


def generate_trajectory_conditioned(ckpt_path, labels, device, steps=30, guidance=2.0,
                                    weights="ema", seed=None):
    """Sample sequences that contain the requested targets (loads the checkpoint).

    labels: dict with traj (B, M, L, 2) bins, n_targets (B,), cls (B, M).
    Returns dB maps (B, L, 64, 64).
    """
    model, cfg = load_trajectory_conditioned(ckpt_path, device, weights)
    return sample_trajectory_conditioned(model, cfg, labels, device, steps=steps,
                                         guidance=guidance, seed=seed)


def generate_conditioned(ckpt_path, batch, device, steps=50, guidance=2.0,
                         weights=None, patch_reduction=None, seed=None):
    from src.conditioning import ConditionEncoder

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ckpt, weights=weights))
    _set_patch_reduction(model, patch_reduction)
    encoder = ConditionEncoder(dim=cfg["model"]["dim"]).to(device)
    encoder.load_state_dict(select_checkpoint_state(
        ckpt, component="encoder", weights=weights))
    model.eval(); encoder.eval()

    B = batch["v0"].shape[0]
    with torch.no_grad():
        cond = encoder(batch, device)
        null = encoder.null(B, device)

    def guided(xt, t, _cond=None):
        e_c = model(xt, t, cond)
        e_n = model(xt, t, null)
        return e_n + guidance * (e_c - e_n)

    diff = GaussianDiffusion(
        cfg["diffusion"]["timesteps"],
        x0_clamp=cfg["diffusion"].get("x0_clamp", DEFAULT_X0_CLAMP),
        parameterization=cfg["diffusion"].get("parameterization", "eps"))
    L = cfg["data"]["seq_len"]
    if seed is not None:
        torch.manual_seed(seed)
    x = diff.ddim_sample(guided, (B, L, 64, 64), device, steps=steps)
    stats = RadarSequenceDataset(cfg["data"]["cache_dir"], "val").stats
    return denormalize(x.cpu(), stats)


if __name__ == "__main__":
    from src.eval.metrics import evaluate_sequences
    from src.viz import sequence_gif, sequence_grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint for learned generation")
    ap.add_argument("--easy", action="store_true",
                    help="sample the exact one-target E0 simulator instead")
    ap.add_argument("--easy-latent", metavar="CHECKPOINT",
                    help="sample the learned E0 trajectory model and render RD maps")
    ap.add_argument("--hard-latent", metavar="CHECKPOINT",
                    help="sample the learned full-regime scene model")
    ap.add_argument("--easy-cache", default="data/cache_easy",
                    help="E0 validation cache used for reference metrics")
    ap.add_argument("--hard-cache", default="data/cache",
                    help="full-regime validation cache used for reference metrics")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-real", type=int, default=4,
                    help="real val sequences to render with GT markers")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--weights", choices=("raw", "ema", "auto"),
                    help="checkpoint weights (default: sample.weights, then raw)")
    ap.add_argument("--patch-reduction", choices=("mean", "tile", "hann"),
                    help="override DiT overlap reconstruction for diagnosis")
    ap.add_argument("--seed", type=int,
                    help="fix the initial sampling noise for paired comparisons")
    args = ap.parse_args()

    if sum((bool(args.ckpt), args.easy, bool(args.easy_latent),
            bool(args.hard_latent))) != 1:
        ap.error("specify exactly one of --ckpt, --easy, --easy-latent, or --hard-latent")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = (args.easy_cache if args.easy or args.easy_latent else
                 args.hard_cache if args.hard_latent else None)
    manifest = (yaml.safe_load((Path(cache_dir) / "manifest.yaml").read_text())
                if cache_dir is not None else None)
    generated_batch = None
    if args.hard_latent:
        from src.hard_latent import generate as generate_hard_latent
        generated_batch = generate_hard_latent(args.hard_latent, args.n,
                                                seed=args.seed, device=device)
        x = generated_batch["x"]
    elif args.easy_latent:
        from src.easy_latent import generate as generate_latent
        generated_batch = generate_latent(args.easy_latent, args.n, seed=args.seed,
                                          device=device)
        x = generated_batch["x"]
    elif args.easy:
        x = generate_easy(args.n, seed=args.seed,
                          seq_len=manifest["seq_len"],
                          frame_interval=manifest["frame_interval"])
    else:
        x = generate(args.ckpt, args.n, device, steps=args.steps,
                     weights=args.weights, patch_reduction=args.patch_reduction,
                     seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, seq in enumerate(x):
        marks = ({"traj": generated_batch["traj"][i],
                  "n_targets": generated_batch["n_targets"][i]}
                 if generated_batch is not None else {})
        sequence_grid(seq, out / f"seq_{i}.png", **marks)
        sequence_gif(seq, out / f"seq_{i}.gif", **marks)

    cfg = ({"data": {"seq_len": manifest["seq_len"],
                     "cache_dir": cache_dir}} if manifest is not None
           else torch.load(args.ckpt, map_location="cpu")["config"])
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    for i in range(min(args.n_real, len(val))):
        item = val[i]
        seq = denormalize(item["x"], val.stats)
        sequence_grid(seq, out / f"real_seq_{i}.png",
                      traj=item["traj"], n_targets=item["n_targets"])
        sequence_gif(seq, out / f"real_seq_{i}.gif",
                     traj=item["traj"], n_targets=item["n_targets"])
    x_real = torch.stack([denormalize(val[i]["x"], val.stats)
                          for i in range(min(len(val), args.n * 4))])
    metrics = evaluate_sequences(x, x_real, seq_len=cfg["data"]["seq_len"])
    with open(out / "metrics.yaml", "w") as fh:
        yaml.safe_dump(metrics, fh)
    print(yaml.safe_dump(metrics))
