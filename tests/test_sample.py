import torch
import yaml
from src.dataset import generate_cache
from src.train import train
from src.sample import generate


def test_generate_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2, shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4)
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=3)
    x = generate(str(tmp_path / "ckpt" / "last.pt"), n_seq=1,
                 device=torch.device("cpu"), steps=5)
    assert x.shape == (1, 16, 64, 64)
    assert torch.isfinite(x).all()
