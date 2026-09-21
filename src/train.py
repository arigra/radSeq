"""Training loop for the temporal radar DiT."""
import argparse
from pathlib import Path
import time

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import RadarSequenceDataset
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.dit import TemporalDiT, load_unconditional_weights
from src.ema import make_ema, update_ema
from src.losses import diffusion_loss, smooth_loss
from src.trajectory_condition import drop_condition, render_condition


def _research_loss(x0_hat, x0, t, batch, cfg):
    """Optional radar-specific research terms; absent from existing runs."""
    tr = cfg["train"]
    weights = (tr.get("lambda_range_doppler", 0.0),
               tr.get("lambda_target_support", 0.0))
    if not any(weights):
        return x0_hat.new_zeros(())
    keep = t <= tr.get("physics_max_t", 700)
    if not keep.any():
        return x0_hat.new_zeros(())
    from src.research_losses import range_doppler_residual, target_support_loss
    pred, true = x0_hat[keep], x0[keep]
    sub = {key: value[keep.to(value.device)] for key, value in batch.items()
           if key in ("traj", "n_targets")}
    total = pred.new_zeros(())
    if weights[0]:
        total = total + weights[0] * range_doppler_residual(
            pred, sub, frame_interval=cfg["data"]["frame_interval"])
    if weights[1]:
        total = total + weights[1] * target_support_loss(pred, true, sub)
    return total


def _predict(model, xt, t, cond, tr, device, cond_map=None):
    """Network forward, optionally under bf16 autocast (train.amp: bf16)."""
    amp = tr.get("amp", "off")
    if amp not in ("off", "bf16"):
        raise ValueError(f"unknown train.amp {amp!r}")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
        if cond_map is None:
            prediction = model(xt, t, cond)
        else:
            prediction = model(xt, t, cond, cond_map=cond_map)
    return prediction.float()


def _trajectory_condition(model, batch, device, dropout_p):
    """Condition map for trajectory-conditioned models (cond_channels > 0), else None."""
    if not getattr(model, "cond_channels", 0):
        return None
    cond_map = render_condition(batch["traj"].to(device), batch["n_targets"].to(device),
                                batch["cls"].to(device))
    return drop_condition(cond_map, dropout_p)


def build_model(cfg, device):
    m = cfg["model"]
    architecture = m.get("architecture", "temporal_dit")
    if architecture == "temporal_dit":
        model = TemporalDiT(
            seq_len=cfg["data"]["seq_len"], patch=m["patch"],
            stride=m["stride"], dim=m["dim"], depth=m["depth"],
            heads=m["heads"], attn_mode=m.get("attn_mode", "temporal"),
            patch_reduction=m.get("patch_reduction", "mean"),
            cond_channels=m.get("cond_channels", 0))
    elif architecture == "unet2d":
        from src.unet import SpatialUNet
        model = SpatialUNet(
            seq_len=cfg["data"]["seq_len"],
            base_channels=m.get("base_channels", 64),
            time_dim=m.get("dim", 256))
    else:
        raise ValueError(f"unknown model architecture {architecture!r}")
    return model.to(device)


def _loss_components(model, encoder, diff, batch, cfg, device, dropout_p):
    tr = cfg["train"]
    x0 = batch["x"].to(device)
    t = torch.randint(0, diff.T, (x0.shape[0],), device=device)
    eps = torch.randn_like(x0)
    xt = diff.q_sample(x0, t, eps)
    cond = (encoder(batch, device, dropout_p=dropout_p)
            if encoder is not None else None)
    cond_map = _trajectory_condition(model, batch, device, dropout_p)
    prediction = _predict(model, xt, t, cond, tr, device, cond_map)
    dit = diffusion_loss(
        diff.target(x0, t, eps), prediction,
        weight=diff.objective_weights(
            t, tr.get("loss_weighting", "none"), tr.get("min_snr_gamma", 5.0)))
    x0_hat = diff.clamp_x0(diff.to_eps_x0(prediction, xt, t)[1])
    smooth = smooth_loss(
        x0_hat, diff.loss_weight(t, tr.get("smooth_weight", "one_minus_alpha_bar")))
    physics = torch.zeros((), device=device)
    if tr.get("phase", 1) >= 2:
        from src.losses import traj_loss_from_batch
        physics = traj_loss_from_batch(x0_hat, batch, tr, device)
    research = _research_loss(x0_hat, x0, t, batch, cfg)
    total = dit + tr["lambda_smooth"] * smooth + physics + research
    return total, {"dit": dit, "smooth": smooth, "physics": physics,
                   "research": research}


@torch.no_grad()
def validate(model, encoder, diff, loader, cfg, device):
    """Deterministic held-out objective for model selection and early stopping."""
    model.eval()
    if encoder is not None:
        encoder.eval()
    sums = {"total": 0.0, "dit": 0.0, "smooth": 0.0,
            "physics": 0.0, "research": 0.0}
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


def ema_residual(decay, steps):
    """Share of the initial weights still present in an EMA after `steps`.

    EMA at 0.9999 has a ~10,000-step memory. Sampled after a short run it is
    largely the random initialisation, which silently yields a model that
    ignores its conditioning while the raw weights are fine.
    """
    if decay is None:
        return 0.0
    return float(decay) ** int(steps)


def train(cfg, device=None, max_steps=None, _record_losses=False, resume=None,
          log_file=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    torch.manual_seed(tr.get("seed", cfg["data"].get("seed", 1234)))
    ds = RadarSequenceDataset(cfg["data"]["cache_dir"], "train")
    subset = cfg["data"].get("train_subset")
    if subset:
        # scarce-data experiments: the first N training sequences and nothing else
        ds.items = ds.items[:subset]
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
    diff = GaussianDiffusion(
        cfg["diffusion"]["timesteps"],
        x0_clamp=cfg["diffusion"].get("x0_clamp", DEFAULT_X0_CLAMP),
        parameterization=cfg["diffusion"].get("parameterization", "eps"),
        terminal_x0=cfg["diffusion"].get("terminal_x0", "model"),
        schedule_shift=cfg["diffusion"].get("schedule_shift", 1.0))
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
    if tr.get("init_from") and resume_state is None:
        # Warm start. EMA weights by default, but only a long run warms its EMA
        # up: a short pretraining's EMA is still mostly random initialisation
        # (see ema_residual), so short runs must set init_weights: raw.
        source = torch.load(tr["init_from"], map_location="cpu")
        init_weights = tr.get("init_weights", "ema")
        if init_weights == "ema":
            state = source.get("ema_model") or source["model"]
        elif init_weights == "raw":
            state = source["model"]
        else:
            raise ValueError("train.init_weights must be 'ema' or 'raw'")
        load_unconditional_weights(model, state)
    ema_decay = tr.get("ema_decay")
    if ema_decay is not None and not 0.0 <= ema_decay < 1.0:
        raise ValueError("train.ema_decay must be in [0, 1)")
    ema_model = make_ema(model) if ema_decay is not None else None
    ema_encoder = (make_ema(encoder)
                   if ema_decay is not None and encoder is not None else None)
    ema_updates = 0
    if ema_model is not None and resume_state is not None:
        if resume_state.get("ema_model") is not None:
            ema_model.load_state_dict(resume_state["ema_model"])
        if ema_encoder is not None and resume_state.get("ema_encoder") is not None:
            ema_encoder.load_state_dict(resume_state["ema_encoder"])
        ema_updates = resume_state.get("ema_updates", 0)
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

    report("start device={} phase={} epoch={} step={} params={} ema_decay={} "
           "smooth_weight={} x0_clamp={}".format(
               device, tr.get("phase", 1), epoch, step,
               sum(p.numel() for p in opt_params), ema_decay,
               tr.get("smooth_weight", "one_minus_alpha_bar"), diff.x0_clamp)
           + " param={} loss_weighting={}".format(
               diff.parameterization, tr.get("loss_weighting", "none")))
    residual = ema_residual(ema_decay, tr["epochs"] * len(loader))
    if residual > 0.1:
        report("WARNING ema_decay={} over {} steps keeps {:.0%} of the initial "
               "weights in the EMA; sampling from it gives a partly untrained "
               "model. Lower ema_decay or sample raw weights.".format(
                   ema_decay, tr["epochs"] * len(loader), residual))
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
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
            "ema_model": (ema_model.state_dict()
                          if ema_model is not None else None),
            "ema_encoder": (ema_encoder.state_dict()
                            if ema_encoder is not None else None),
            "ema_updates": ema_updates}
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
            cond_map = _trajectory_condition(model, batch, device,
                                             tr.get("cond_dropout", 0.1))
            prediction = _predict(model, xt, t, cond, tr, device, cond_map)
            dit = diffusion_loss(
                diff.target(x0, t, eps), prediction,
                weight=diff.objective_weights(
                    t, tr.get("loss_weighting", "none"),
                    tr.get("min_snr_gamma", 5.0)))
            # Clamp x0 before regularization to avoid high-t numerical spikes.
            x0_hat = diff.clamp_x0(diff.to_eps_x0(prediction, xt, t)[1])
            smooth = smooth_loss(
                x0_hat,
                diff.loss_weight(t, tr.get("smooth_weight", "one_minus_alpha_bar")))
            physics = torch.zeros((), device=device)
            if tr.get("phase", 1) >= 2:
                from src.losses import traj_loss_from_batch
                physics = traj_loss_from_batch(x0_hat, batch, tr, device)
            research = _research_loss(x0_hat, x0, t, batch, cfg)
            loss = dit + tr["lambda_smooth"] * smooth + physics + research
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            if ema_model is not None:
                update_ema(ema_model, model, ema_decay)
                if ema_encoder is not None:
                    update_ema(ema_encoder, encoder, ema_decay)
                ema_updates += 1
            losses.append(loss.item())
            step += 1
            if use_wandb and step % log_every == 0:
                wandb.log({
                    "global_step": step, "train/total_loss": loss.item(),
                    "train/dit_loss": dit.item(), "train/smooth_loss": smooth.item(),
                    "train/physics_loss": physics.item(),
                    "train/research_loss": research.item(),
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
                   "physics={:.6f} research={:.6f} best={:.6f} bad_epochs={}".format(
                       epoch, metrics["total"], metrics["dit"], metrics["smooth"],
                       metrics["physics"], metrics["research"], best_val, bad_epochs))
            if use_wandb:
                wandb.log({
                    "epoch": epoch, "val/total_loss": metrics["total"],
                    "val/dit_loss": metrics["dit"],
                    "val/smooth_loss": metrics["smooth"],
                    "val/physics_loss": metrics["physics"],
                    "val/research_loss": metrics["research"],
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
