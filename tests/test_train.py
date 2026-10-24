import torch


def test_partial_fine_tuning_updates_only_the_matching_parameters():
    import copy
    import yaml
    from src.train import train
    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["train"]["trainable"] = ["^out[.]"]
    cfg["train"]["wandb"] = False
    model = train(cfg, device=torch.device("cpu"), max_steps=2)
    ref = train({**cfg, "train": {**cfg["train"], "lr": 0.0}}, device=torch.device("cpu"), max_steps=2)
    changed = {n for (n, a), (_, b) in zip(model.named_parameters(), ref.named_parameters())
               if not torch.equal(a, b)}
    assert changed and all(n.startswith("out.") for n in changed)
