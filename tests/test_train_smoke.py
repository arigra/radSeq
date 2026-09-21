import torch
import yaml
from src.dataset import generate_cache
from src.train import early_stop_update, train


def _tiny_config(tmp_path):
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2,
                       shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4)
    # The tiny single-batch overfit needs a higher lr than the production
    # default (tuned for full-scale, many-epoch training) to visibly converge
    # within 150 steps; the config owns hyperparameters, not train().
    # log_file under tmp_path: the default (logs/phase1.log) would append test
    # runs to the real training log.
    cfg["train"].update(batch_size=2, epochs=1, lr=3.0e-4,
                        ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    # Pin to phase 1 so a future base.yaml phase flip cannot break this smoke test.
    cfg["train"]["phase"] = 1
    return cfg


def test_build_model_honors_attn_mode(tmp_path):
    """The control experiment selects its architecture from config alone, so
    nothing else about the training run can differ between the two arms."""
    from src.dit import FactorizedBlock, TemporalBlock
    from src.train import build_model
    cfg = _tiny_config(tmp_path)
    dev = torch.device("cpu")
    assert isinstance(build_model(cfg, dev).blocks[0], TemporalBlock)
    cfg["model"]["attn_mode"] = "factorized"
    assert isinstance(build_model(cfg, dev).blocks[0], FactorizedBlock)


def test_single_batch_overfit(tmp_path):
    """Loss on a fixed batch and fixed t must drop substantially."""
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    losses = train(cfg, device=torch.device("cpu"), max_steps=150,
                   _record_losses=True)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < 0.6 * early, f"no learning: {early:.4f} -> {late:.4f}"


def test_checkpoint_written(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=3)
    path = tmp_path / "ckpt" / "last.pt"
    assert path.exists()
    state = torch.load(path, map_location="cpu")
    assert state["format_version"] == 2
    assert state["step"] == 3
    assert "optimizer" in state


def test_resume_continues_step_count(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    path = tmp_path / "ckpt" / "last.pt"
    train(cfg, device=torch.device("cpu"), max_steps=2)
    train(cfg, device=torch.device("cpu"), max_steps=4, resume=path)
    state = torch.load(path, map_location="cpu")
    assert state["step"] == 4


def test_bf16_autocast_trains_and_rejects_unknown_modes(tmp_path):
    import pytest
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    cfg["train"].update(amp="bf16", log_file=str(tmp_path / "train.log"))
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    losses = train(cfg, device=torch.device("cpu"), max_steps=3,
                   _record_losses=True)
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    cfg["train"]["amp"] = "fp8"
    with pytest.raises(ValueError):
        train(cfg, device=torch.device("cpu"), max_steps=1)


def test_early_stopping_requires_meaningful_improvement():
    best, bad, improved = early_stop_update(0.5, 0.49, 3, min_delta=0.001)
    assert improved and best == 0.49 and bad == 0
    best, bad, improved = early_stop_update(best, 0.4895, bad, min_delta=0.001)
    assert not improved and best == 0.49 and bad == 1


def test_validation_writes_best_checkpoint(tmp_path):
    cfg = _tiny_config(tmp_path)
    cfg["train"].update(
        val_every_epochs=1, val_batch_size=2, val_seed=9,
        log_file=str(tmp_path / "train.log"))
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"))
    state = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu")
    assert state["epoch"] == 1
    assert state["best_val"] < float("inf")
    assert state["bad_epochs"] == 0


def test_viz_writes_files(tmp_path):
    from src.viz import sequence_grid, sequence_gif
    x = torch.randn(16, 64, 64)
    sequence_grid(x, str(tmp_path / "grid.png"))
    sequence_gif(x, str(tmp_path / "seq.gif"))
    assert (tmp_path / "grid.png").exists() and (tmp_path / "seq.gif").exists()


def test_trajectory_condition_map_is_built_only_for_conditional_models():
    from src.train import _trajectory_condition, build_model
    from src.simulator import generate_sequences
    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg["model"].update(dim=32, depth=1, heads=4, patch=8, stride=8, attn_mode="factorized")
    batch = generate_sequences(n=2, seed=0)
    plain = build_model(cfg, torch.device("cpu"))
    assert _trajectory_condition(plain, batch, torch.device("cpu"), 0.0) is None
    cfg["model"]["cond_channels"] = 4
    conditional = build_model(cfg, torch.device("cpu"))
    cond_map = _trajectory_condition(conditional, batch, torch.device("cpu"), 0.0)
    assert cond_map.shape == (2, 16, 4, 64, 64)


def test_init_from_warm_starts_a_conditional_model_exactly(tmp_path):
    torch.manual_seed(0)
    base_cfg = _tiny_config(tmp_path / "base")
    base_cfg["model"].update(patch=8, stride=8, attn_mode="factorized")
    generate_cache(base_cfg["data"]["cache_dir"], 4, 2, seq_len=16, seed=7, shard_size=4)
    train(base_cfg, device=torch.device("cpu"), max_steps=2)
    base = torch.load(tmp_path / "base" / "ckpt" / "last.pt", map_location="cpu")["model"]

    cfg = _tiny_config(tmp_path / "cond")
    cfg["data"]["cache_dir"] = base_cfg["data"]["cache_dir"]
    cfg["model"].update(patch=8, stride=8, attn_mode="factorized", cond_channels=4)
    # lr 0: weights cannot move, so the checkpoint shows exactly what init_from loaded
    cfg["train"].update(init_from=str(tmp_path / "base" / "ckpt" / "last.pt"),
                        cond_dropout=0.5, lr=0.0)
    losses = train(cfg, device=torch.device("cpu"), max_steps=2, _record_losses=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    state = torch.load(tmp_path / "cond" / "ckpt" / "last.pt", map_location="cpu")["model"]
    assert state["proj.weight"].shape == (64, 64 * 5)
    assert torch.equal(state["proj.weight"][:, :64], base["proj.weight"])
    assert torch.count_nonzero(state["proj.weight"][:, 64:]) == 0
    assert all(torch.equal(state[k], base[k]) for k in base if k != "proj.weight")


def test_train_subset_uses_only_the_first_n_training_sequences(tmp_path):
    """Scarce-data runs: 4 cached sequences, subset 2, batch 2 -> one step per epoch."""
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    cfg["data"]["train_subset"] = 2
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"))
    assert torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu")["step"] == 1


def test_ema_residual_measures_how_much_initialisation_survives():
    """EMA 0.9999 needs tens of thousands of steps. The fidelity pilot's
    7,480-step pretraining kept 47% of the random initialisation in its EMA
    weights, and the samples drawn from them landed only 8% of targets."""
    from src.train import ema_residual
    assert abs(ema_residual(0.9999, 7480) - 0.47) < 0.01
    assert ema_residual(0.9999, 39700) < 0.03
    assert ema_residual(None, 100) == 0.0


def test_init_weights_raw_warm_starts_from_the_trained_weights_not_the_ema(tmp_path):
    torch.manual_seed(0)
    base_cfg = _tiny_config(tmp_path / "base")
    base_cfg["model"].update(patch=8, stride=8, attn_mode="factorized", cond_channels=4)
    base_cfg["train"]["ema_decay"] = 0.9999
    generate_cache(base_cfg["data"]["cache_dir"], 4, 2, seq_len=16, seed=7, shard_size=4)
    train(base_cfg, device=torch.device("cpu"), max_steps=3)
    src_state = torch.load(tmp_path / "base" / "ckpt" / "last.pt", map_location="cpu")

    cfg = _tiny_config(tmp_path / "ft")
    cfg["data"]["cache_dir"] = base_cfg["data"]["cache_dir"]
    cfg["model"].update(patch=8, stride=8, attn_mode="factorized", cond_channels=4)
    cfg["train"].update(init_from=str(tmp_path / "base" / "ckpt" / "last.pt"),
                        init_weights="raw", lr=0.0)
    train(cfg, device=torch.device("cpu"), max_steps=1)
    got = torch.load(tmp_path / "ft" / "ckpt" / "last.pt", map_location="cpu")["model"]
    assert all(torch.equal(got[k], src_state["model"][k]) for k in src_state["model"])
    assert not all(torch.equal(got[k], src_state["ema_model"][k]) for k in src_state["model"])


def test_trains_on_a_scene_cache_with_vehicle_conditioning(tmp_path):
    """The RADIal-grid path: memory-mapped scene cache, non-square 512x256
    maps, position-only vehicle conditioning, and a train_subset."""
    import subprocess, sys
    cache = tmp_path / "scene"
    subprocess.run([sys.executable, "scripts/build_scene_cache.py", "--variant", "engineer",
                    "--n-train", "4", "--n-val", "2", "--seq-len", "2", "--out", str(cache)],
                   check=True, env={"PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "",
                                    "PATH": "/usr/bin:/bin"})
    cfg = _tiny_config(tmp_path)
    cfg["data"].update(kind="scene", cache_dir=str(cache), seq_len=2,
                       n_range=512, n_doppler=256, train_subset=2)
    cfg["model"].update(patch=32, stride=32, attn_mode="factorized", cond_channels=2)
    cfg["train"].update(batch_size=2, cond_dropout=0.5)
    losses = train(cfg, device=torch.device("cpu"), max_steps=2, _record_losses=True)
    assert len(losses) == 2 and all(torch.isfinite(torch.tensor(losses)))
    state = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu")
    assert state["step"] == 2
