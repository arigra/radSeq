import torch
import yaml

from src.dataset import generate_cache
from src.train import build_model
from src.train import train
from src.unet import SpatialUNet


def _small_unet():
    return SpatialUNet(seq_len=4, base_channels=16, time_dim=32)


def test_unet_forward_shape():
    model = _small_unet()
    out = model(torch.randn(2, 4, 16, 16), torch.tensor([10, 20]))
    assert out.shape == (2, 4, 16, 16)


def test_unet_zero_init_output():
    model = _small_unet()
    out = model(torch.randn(1, 4, 16, 16), torch.tensor([5]))
    assert out.abs().max() == 0.0


def test_build_model_selects_unet_from_config():
    with open("configs/abl_unet2d.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["model"].update(base_channels=16, dim=32)
    model = build_model(cfg, torch.device("cpu"))
    assert isinstance(model, SpatialUNet)
    assert model.dim == 32


def test_unet_uses_shared_training_loop(tmp_path):
    with open("configs/abl_unet2d.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2,
                       shard_size=4)
    cfg["model"].update(base_channels=16, dim=32)
    cfg["train"].update(batch_size=2, epochs=1,
                        ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=1)
    state = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu")
    assert state["step"] == 1
    assert state["config"]["model"]["architecture"] == "unet2d"
