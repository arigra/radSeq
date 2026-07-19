import torch
from src.conditioning import ConditionEncoder


def _batch(B=4):
    return {
        "v0": torch.randn(B, 5), "acc": torch.randn(B, 5) * 0.3,
        "cls": torch.randint(0, 3, (B, 5)),
        "env": torch.randn(B, 3),
        "n_targets": torch.tensor([1, 2, 5, 3][:B]),
    }


def test_output_shape():
    enc = ConditionEncoder(dim=64)
    c = enc(_batch(), torch.device("cpu"))
    assert c.shape == (4, 64)


def test_padding_invariance():
    """Changing padded (unused) target slots must not change the output."""
    torch.manual_seed(0)
    enc = ConditionEncoder(dim=64)
    b1 = _batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()}
    b2["v0"][0, 1:] = 99.0     # sample 0 has n_targets=1; slots 1..4 are pad
    b2["acc"][0, 1:] = 99.0
    c1 = enc(b1, torch.device("cpu"))
    c2 = enc(b2, torch.device("cpu"))
    assert torch.allclose(c1[0], c2[0], atol=1e-6)


def test_cfg_dropout_yields_null():
    torch.manual_seed(0)
    enc = ConditionEncoder(dim=64)
    c = enc(_batch(), torch.device("cpu"), dropout_p=1.0)
    null = enc.null(4, torch.device("cpu"))
    assert torch.allclose(c, null)


def test_feeds_dit():
    from src.dit import TemporalDiT
    m = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=4,
                    dim=64, depth=2, heads=4)
    enc = ConditionEncoder(dim=64)
    x = torch.randn(4, 4, 16, 16)
    out = m(x, torch.tensor([1, 2, 3, 4]), cond=enc(_batch(), torch.device("cpu")))
    assert out.shape == (4, 4, 16, 16)
