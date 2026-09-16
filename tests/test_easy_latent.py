import torch
import yaml

from src.easy_latent import generate, train


def test_easy_latent_training_and_sampling(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    items = []
    for i in range(24):
        traj = torch.zeros(1, 4, 2)
        traj[0, :, 0] = 20 + i / 4
        items.append({"traj": traj, "v0": torch.tensor([float(i % 5 - 2)]),
                      "acc": torch.tensor([float(i % 3 - 1) * 0.1]),
                      "cls": torch.tensor([0]), "n_targets": torch.tensor(1)})
    torch.save(items, cache / "train_0000.pt")
    (cache / "manifest.yaml").write_text(yaml.safe_dump({
        "seq_len": 4, "frame_interval": 0.5}))
    ckpt = train(str(cache), str(tmp_path / "model.pt"), steps=5,
                 batch_size=8, device="cpu")
    out = generate(str(ckpt), 2, seed=4, device="cpu", ddim_steps=10,
                   sampler="ddim")
    assert out["x"].shape == (2, 4, 64, 64)
    assert out["traj"].shape == (2, 1, 4, 2)
    assert torch.equal(out["n_targets"], torch.ones(2, dtype=torch.long))
    assert torch.isfinite(out["x"]).all()
    assert torch.equal(out["x"], generate(str(ckpt), 2, seed=4,
                                            device="cpu", ddim_steps=10,
                                            sampler="ddim")["x"])
