import torch
import yaml

from src.dataset import generate_cache
from src.ema import make_ema, update_ema
from src.sample import select_checkpoint_state
from src.train import train


def test_ema_update_interpolates_parameters_and_copies_buffers():
    source = torch.nn.BatchNorm1d(2)
    with torch.no_grad():
        source.weight.fill_(1.0)
        source.running_mean.fill_(2.0)
    ema = make_ema(source)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.running_mean.fill_(4.0)
    update_ema(ema, source, decay=0.75)
    assert torch.equal(ema.weight, torch.full_like(ema.weight, 1.5))
    assert torch.equal(ema.running_mean, torch.full_like(ema.running_mean, 4.0))
    assert not any(p.requires_grad for p in ema.parameters())


def test_select_checkpoint_state_is_explicit_and_configurable():
    raw = {"x": torch.tensor(1)}
    ema = {"x": torch.tensor(2)}
    ckpt = {"model": raw, "ema_model": ema,
            "config": {"sample": {"weights": "ema"}}}
    assert select_checkpoint_state(ckpt) is ema
    assert select_checkpoint_state(ckpt, weights="raw") is raw
    assert select_checkpoint_state(ckpt, weights="auto") is ema


def test_training_checkpoint_contains_resumable_ema(tmp_path):
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2,
                       shard_size=4)
    cfg["model"].update(dim=32, depth=1, heads=4)
    cfg["train"].update(batch_size=2, epochs=1, ema_decay=0.9,
                        ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=2)
    state = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu")
    assert state["ema_updates"] == 2
    assert state["ema_model"] is not None
    assert set(state["ema_model"]) == set(state["model"])
    assert any(not torch.equal(state["ema_model"][name], value)
               for name, value in state["model"].items()
               if value.is_floating_point())
