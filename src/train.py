"""Training loop for the temporal radar DiT."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import RadarSequenceDataset
from src.diffusion import GaussianDiffusion
from src.dit import TemporalDiT
from src.losses import diffusion_loss, smooth_loss


def build_model(cfg, device):
    m = cfg["model"]
    return TemporalDiT(seq_len=cfg["data"]["seq_len"], patch=m["patch"],
                       stride=m["stride"], dim=m["dim"], depth=m["depth"],
                       heads=m["heads"]).to(device)


def train(cfg, device=None, max_steps=None, _record_losses=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    ds = RadarSequenceDataset(cfg["data"]["cache_dir"], "train")
    loader = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True,
                        num_workers=0, drop_last=True)
    model = build_model(cfg, device)
    encoder = None
    if tr.get("phase", 1) >= 3:
        from src.conditioning import ConditionEncoder
        encoder = ConditionEncoder(dim=cfg["model"]["dim"]).to(device)
        opt_params = list(model.parameters()) + list(encoder.parameters())
    else:
        opt_params = list(model.parameters())
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    opt = torch.optim.AdamW(opt_params, lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    use_wandb = tr.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="radSeq", config=cfg)

    ckpt_dir = Path(tr["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    losses, step = [], 0
    fixed_batch = next(iter(loader)) if _record_losses else None
    fixed_t = (torch.randint(0, diff.T, (tr["batch_size"],), device=device)
               if _record_losses else None)
    epoch = 0
    # When max_steps is set, keep cycling epochs until it is reached, even if
    # that exceeds tr["epochs"] (e.g. a tiny smoke-test dataset with few
    # batches per epoch). Without max_steps, honor tr["epochs"] as normal.
    while max_steps is not None or epoch < tr["epochs"]:
        for batch in loader:
            if _record_losses:
                batch = fixed_batch  # overfit a single batch deterministically
            x0 = batch["x"].to(device)
            if _record_losses:
                t = fixed_t  # overfit a single (batch, t) pair deterministically
            else:
                t = torch.randint(0, diff.T, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, eps)
            cond = (encoder(batch, device, dropout_p=tr.get("cond_dropout", 0.1))
                    if encoder is not None else None)
            eps_hat = model(xt, t, cond)
            loss = diffusion_loss(eps, eps_hat)
            # Clamp the predicted clean frame before the smoothness term: at
            # high t and with an untrained model, pred_x0 divides by a
            # near-zero sqrt(alpha_bar) and can blow up, spiking the loss and
            # destabilizing training (matches the clamp already used in the
            # DDIM/ancestral samplers in diffusion.py).
            x0_hat = diff.pred_x0(xt, t, eps_hat).clamp(-4, 4)
            loss = loss + tr["lambda_smooth"] * smooth_loss(x0_hat, diff.loss_weight(t))
            if tr.get("phase", 1) >= 2:
                from src.losses import traj_loss_from_batch  # added in Task 11
                loss = loss + traj_loss_from_batch(x0_hat, batch, tr, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            losses.append(loss.item())
            if use_wandb and step % 50 == 0:
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if max_steps is not None and step >= max_steps:
                torch.save({"model": model.state_dict(),
                           "encoder": encoder.state_dict() if encoder else None,
                           "config": cfg},
                           ckpt_dir / "last.pt")
                return losses if _record_losses else model
        torch.save({"model": model.state_dict(),
                   "encoder": encoder.state_dict() if encoder else None,
                   "config": cfg},
                   ckpt_dir / "last.pt")
        epoch += 1
    return losses if _record_losses else model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    train(config)
