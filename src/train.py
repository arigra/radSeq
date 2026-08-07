"""Training loop for the temporal radar DiT."""
import argparse
from pathlib import Path
import time

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


def _loss_components(model, encoder, diff, batch, cfg, device, dropout_p):
    tr = cfg["train"]
    x0 = batch["x"].to(device)
    t = torch.randint(0, diff.T, (x0.shape[0],), device=device)
    eps = torch.randn_like(x0)
    xt = diff.q_sample(x0, t, eps)
    cond = (encoder(batch, device, dropout_p=dropout_p)
            if encoder is not None else None)
    eps_hat = model(xt, t, cond)
    dit = diffusion_loss(eps, eps_hat)
    x0_hat = diff.pred_x0(xt, t, eps_hat).clamp(-4, 4)
    smooth = smooth_loss(x0_hat, diff.loss_weight(t))
    physics = torch.zeros((), device=device)
    if tr.get("phase", 1) >= 2:
        from src.losses import traj_loss_from_batch
        physics = traj_loss_from_batch(x0_hat, batch, tr, device)
    total = dit + tr["lambda_smooth"] * smooth + physics
    return total, {"dit": dit, "smooth": smooth, "physics": physics}


@torch.no_grad()
def validate(model, encoder, diff, loader, cfg, device):
    """Deterministic held-out objective for model selection and early stopping."""
    model.eval()
    if encoder is not None:
        encoder.eval()
    sums = {"total": 0.0, "dit": 0.0, "smooth": 0.0, "physics": 0.0}
    count = 0
    max_batches = cfg["train"].get("val_max_batches")
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg["train"].get("val_seed", 4321))
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            total, parts = _loss_components(
                model, encoder, diff, batch, cfg, device, dropout_p=0.0)
            batch_n = batch["x"].shape[0]
            sums["total"] += total.item() * batch_n
            for name, value in parts.items():
                sums[name] += value.item() * batch_n
            count += batch_n
    model.train()
    if encoder is not None:
        encoder.train()
    if count == 0:
        raise ValueError("validation loader produced no samples")
    return {name: value / count for name, value in sums.items()}


def early_stop_update(best, current, bad_epochs, min_delta):
    if current < best - min_delta:
        return current, 0, True
    return best, bad_epochs + 1, False


def _save_checkpoint(path, model, encoder, optimizer, cfg, epoch, step,
                     **extra):
    """Atomically save all state required to continue training."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 2, "model": model.state_dict(),
                "encoder": encoder.state_dict() if encoder else None,
                "optimizer": optimizer.state_dict(), "config": cfg,
                "epoch": epoch, "step": step, **extra}, tmp)
    tmp.replace(path)


def train(cfg, device=None, max_steps=None, _record_losses=False, resume=None,
          log_file=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    torch.manual_seed(tr.get("seed", cfg["data"].get("seed", 1234)))
    ds = RadarSequenceDataset(cfg["data"]["cache_dir"], "train")
    loader = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True,
                        num_workers=0, drop_last=True)
    val_loader = None
    if tr.get("val_every_epochs"):
        val_ds = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
        val_loader = DataLoader(
            val_ds, batch_size=tr.get("val_batch_size", tr["batch_size"]),
            shuffle=False, num_workers=0, drop_last=False)
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
    epoch, step = 0, 0
    best_val, bad_epochs = float("inf"), 0
    resume_state = None
    if resume:
        state = resume_state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        if encoder is not None:
            if state.get("encoder") is None:
                raise ValueError("phase 3 resume checkpoint has no encoder state")
            encoder.load_state_dict(state["encoder"])
        if "optimizer" not in state:
            raise ValueError("checkpoint predates resumable format (missing optimizer)")
        opt.load_state_dict(state["optimizer"])
        epoch, step = state.get("epoch", 0), state.get("step", 0)
        best_val = state.get("best_val", float("inf"))
        bad_epochs = state.get("bad_epochs", 0)
    use_wandb = tr.get("wandb", False)
    wandb_run = None
    if use_wandb:
        import wandb
        wb = tr.get("wandb_config", {})
        run_id = (resume_state or {}).get("wandb_run_id") or wb.get("run_id")
        wandb_run = wandb.init(
            project=wb.get("project", "radSeq"), entity=wb.get("entity"),
            name=wb.get("name"), group=wb.get("group"),
            job_type=wb.get("job_type", "train"), tags=wb.get("tags"),
            notes=wb.get("notes"), mode=wb.get("mode", "online"),
            id=run_id, resume="allow" if run_id else None, config=cfg)
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    ckpt_dir = Path(tr["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file or tr.get("log_file", "logs/train.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def report(message):
        line = "{} | {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    report("start device={} phase={} epoch={} step={} params={}".format(
        device, tr.get("phase", 1), epoch, step,
        sum(p.numel() for p in opt_params)))
    losses = []
    fixed_batch = next(iter(loader)) if _record_losses else None
    fixed_t = (torch.randint(0, diff.T, (tr["batch_size"],), device=device)
               if _record_losses else None)
    resumed_step, started = step, time.monotonic()
    log_every = max(1, tr.get("log_every_steps", 50))
    save_every = max(1, tr.get("save_every_steps", 250))

    def checkpoint_meta():
        return {
            "best_val": best_val, "bad_epochs": bad_epochs,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None}
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
            dit = diffusion_loss(eps, eps_hat)
            # Clamp x0 before regularization to avoid high-t numerical spikes.
            x0_hat = diff.pred_x0(xt, t, eps_hat).clamp(-4, 4)
            smooth = smooth_loss(x0_hat, diff.loss_weight(t))
            physics = torch.zeros((), device=device)
            if tr.get("phase", 1) >= 2:
                from src.losses import traj_loss_from_batch
                physics = traj_loss_from_batch(x0_hat, batch, tr, device)
            loss = dit + tr["lambda_smooth"] * smooth + physics
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if use_wandb and step % log_every == 0:
                wandb.log({
                    "global_step": step, "train/total_loss": loss.item(),
                    "train/dit_loss": dit.item(), "train/smooth_loss": smooth.item(),
                    "train/physics_loss": physics.item(),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": opt.param_groups[0]["lr"]})
            if step % log_every == 0:
                rate = (step - resumed_step) / max(time.monotonic() - started, 1e-9)
                report("epoch={}/{} step={} loss={:.6f} steps_per_sec={:.2f}".format(
                    epoch + 1, tr["epochs"], step, loss.item(), rate))
            if step % save_every == 0:
                _save_checkpoint(ckpt_dir / "last.pt", model, encoder, opt,
                                 cfg, epoch, step, **checkpoint_meta())
            if max_steps is not None and step >= max_steps:
                _save_checkpoint(ckpt_dir / "last.pt", model, encoder, opt,
                                 cfg, epoch, step, **checkpoint_meta())
                report("stopped at requested step={}".format(step))
                return losses if _record_losses else model
        epoch += 1
        should_stop = False
        val_every = tr.get("val_every_epochs")
        if val_loader is not None and epoch % val_every == 0:
            metrics = validate(model, encoder, diff, val_loader, cfg, device)
            best_val, bad_epochs, improved = early_stop_update(
                best_val, metrics["total"], bad_epochs,
                tr.get("early_stopping_min_delta", 0.0))
            report("validation epoch={} total={:.6f} dit={:.6f} smooth={:.6f} "
                   "physics={:.6f} best={:.6f} bad_epochs={}".format(
                       epoch, metrics["total"], metrics["dit"], metrics["smooth"],
                       metrics["physics"], best_val, bad_epochs))
            if use_wandb:
                wandb.log({
                    "epoch": epoch, "val/total_loss": metrics["total"],
                    "val/dit_loss": metrics["dit"],
                    "val/smooth_loss": metrics["smooth"],
                    "val/physics_loss": metrics["physics"],
                    "val/best_loss": best_val})
                wandb_run.summary["best_val_loss"] = best_val
                wandb_run.summary["best_epoch"] = epoch if improved else wandb_run.summary.get("best_epoch")
            if improved:
                _save_checkpoint(ckpt_dir / "best.pt", model, encoder, opt,
                                 cfg, epoch, step, **checkpoint_meta())
            patience = tr.get("early_stopping_patience")
            min_epochs = tr.get("early_stopping_min_epochs", 0)
            should_stop = bool(
                patience and epoch >= min_epochs and bad_epochs >= patience)
        _save_checkpoint(ckpt_dir / "last.pt", model, encoder, opt,
                         cfg, epoch, step, **checkpoint_meta())
        snapshot_every = tr.get("snapshot_every_epochs", 10)
        if snapshot_every and epoch % snapshot_every == 0:
            _save_checkpoint(ckpt_dir / "epoch_{:04d}.pt".format(epoch),
                             model, encoder, opt, cfg, epoch, step,
                             **checkpoint_meta())
        report("completed epoch={}/{} step={}".format(epoch, tr["epochs"], step))
        if should_stop:
            report("early stopping at epoch={} after {} unimproved validations".format(
                epoch, bad_epochs))
            break
    if use_wandb:
        wandb_run.summary["stopped_epoch"] = epoch
        wandb.finish()
    return losses if _record_losses else model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--resume", help="resume from a format-v2 checkpoint")
    ap.add_argument("--log-file", help="override train.log_file")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    train(config, resume=args.resume, log_file=args.log_file)
