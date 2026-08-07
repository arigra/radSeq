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
    cfg["train"].update(batch_size=2, epochs=1, lr=3.0e-4,
                        ckpt_dir=str(tmp_path / "ckpt"))
    # Pin to phase 1 so a future base.yaml phase flip cannot break this smoke test.
    cfg["train"]["phase"] = 1
    return cfg


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
