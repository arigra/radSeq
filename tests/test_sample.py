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
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=3)
    x = generate(str(tmp_path / "ckpt" / "last.pt"), n_seq=1,
                 device=torch.device("cpu"), steps=5)
    assert x.shape == (1, 16, 64, 64)
    assert torch.isfinite(x).all()


def test_cache_dir_resolves_from_repo_root_not_cwd(tmp_path, monkeypatch):
    """Notebooks run from notebooks/, so 'data/cache' must not depend on cwd."""
    from pathlib import Path
    from src.sample import resolve_cache_dir
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert resolve_cache_dir("data/cache") == repo / "data" / "cache"
    assert resolve_cache_dir(str(tmp_path)) == tmp_path


def test_generate_uses_the_checkpoint_diffusion_config(tmp_path, monkeypatch):
    """A v-prediction, schedule-shifted checkpoint must sample with that schedule."""
    import src.sample as sample
    torch.manual_seed(0)
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2, shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4, patch=8, stride=8, attn_mode="factorized")
    cfg["diffusion"].update(parameterization="v", schedule_shift=4.0, x0_clamp="off")
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=2)
    seen = {}
    real_ddim = sample.GaussianDiffusion.ddim_sample

    def spy(self, *args, **kwargs):
        seen["diff"] = self
        return real_ddim(self, *args, **kwargs)

    monkeypatch.setattr(sample.GaussianDiffusion, "ddim_sample", spy)
    monkeypatch.chdir(tmp_path)                     # not the repo root, like a notebook
    x = generate(str(tmp_path / "ckpt" / "last.pt"), n_seq=1,
                 device=torch.device("cpu"), steps=3)
    assert x.shape == (1, 16, 64, 64) and torch.isfinite(x).all()
    assert (seen["diff"].parameterization, seen["diff"].schedule_shift,
            seen["diff"].x0_clamp) == ("v", 4.0, None)


def test_trajectory_conditioned_sampling_with_zero_guidance_is_unconditional(tmp_path):
    from src.sample import generate_trajectory_conditioned
    from src.simulator import generate_sequences
    torch.manual_seed(0)
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2, shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4, patch=8, stride=8,
                        attn_mode="factorized", cond_channels=4)
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=2)
    ckpt = str(tmp_path / "ckpt" / "last.pt")
    labels = generate_sequences(n=2, seed=3)
    guided = generate_trajectory_conditioned(ckpt, labels, torch.device("cpu"), steps=3,
                                             guidance=0.0, weights="raw", seed=5)
    plain = generate(ckpt, n_seq=2, device=torch.device("cpu"), steps=3, weights="raw", seed=5)
    assert guided.shape == (2, 16, 64, 64) and torch.isfinite(guided).all()
    assert torch.allclose(guided, plain, atol=1e-3)
