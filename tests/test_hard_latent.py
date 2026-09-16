import torch
import yaml

from src.hard_latent import generate, train


def test_hard_latent_training_and_sampling(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    items = []
    for i in range(24):
        count = i % 3 + 1
        items.append({"v0": torch.linspace(-1, 1, count),
                      "acc": torch.linspace(-0.1, 0.1, count),
                      "cls": torch.arange(count) % 3,
                      "n_targets": torch.tensor(count)})
    torch.save(items, cache / "train_0000.pt")
    (cache / "manifest.yaml").write_text(yaml.safe_dump({
        "seq_len": 4, "frame_interval": 0.5}))
    ckpt = train(str(cache), str(tmp_path / "model.pt"), steps=5,
                 batch_size=8, device="cpu")
    out = generate(str(ckpt), 2, seed=4, device="cpu")
    assert out["x"].shape == (2, 4, 64, 64)
    assert out["traj"].shape == (2, 5, 4, 2)
    assert (out["n_targets"] >= 1).all() and (out["n_targets"] <= 3).all()
    assert torch.isfinite(out["x"]).all()
    assert torch.equal(out["x"], generate(str(ckpt), 2, seed=4,
                                            device="cpu")["x"])
