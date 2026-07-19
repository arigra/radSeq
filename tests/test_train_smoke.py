import torch
import yaml
from src.dataset import generate_cache
from src.train import train


def _tiny_config(tmp_path):
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2,
                       shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4)
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"))
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
    assert (tmp_path / "ckpt" / "last.pt").exists()


def test_viz_writes_files(tmp_path):
    from src.viz import sequence_grid, sequence_gif
    x = torch.randn(16, 64, 64)
    sequence_grid(x, str(tmp_path / "grid.png"))
    sequence_gif(x, str(tmp_path / "seq.gif"))
    assert (tmp_path / "grid.png").exists() and (tmp_path / "seq.gif").exists()
