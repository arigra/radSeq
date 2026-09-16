import pytest
import torch

from src.simulator import TemporalRadarSimulator, generate_sequences


def test_batch_shapes_and_keys():
    out = generate_sequences(n=3, seq_len=8, seed=0)
    assert out["x"].shape == (3, 8, 64, 64) and out["x"].dtype == torch.float32
    assert out["traj"].shape == (3, 5, 8, 2)
    assert out["v0"].shape == (3, 5) and out["acc"].shape == (3, 5)
    assert out["cls"].shape == (3, 5) and out["env"].shape == (3, 3)
    assert out["n_targets"].shape == (3,)


def test_defaults_match_the_simulator_exactly():
    """All-defaults generation is the same random stream as gen_sequence, so
    the recipe behind the locked cache is unchanged."""
    out = generate_sequences(n=2, seq_len=8, seed=11)
    torch.manual_seed(11)
    sim = TemporalRadarSimulator(seq_len=8)
    for i in range(2):
        ref = sim.gen_sequence()
        m = ref["n_targets"]
        assert int(out["n_targets"][i]) == m
        assert torch.equal(out["x"][i], ref["x"])
        assert torch.equal(out["traj"][i, :m], ref["traj"])


def test_exact_target_count():
    out = generate_sequences(n=4, seq_len=4, n_targets=3, seed=1)
    assert (out["n_targets"] == 3).all()


def test_single_static_target_on_an_empty_map_is_deterministic():
    """No clutter, no noise, one steady target at fixed kinematics: the only
    randomness left is a global phase, which |RD| ignores."""
    torch.manual_seed(0)
    kw = dict(n=1, seq_len=4, n_targets=1, target_class="steady", snr_db=20.0,
              clutter=False, noise=False, r0=90.0, v0=0.0, a=0.0)
    x1 = generate_sequences(seed=0, **kw)["x"]
    x2 = generate_sequences(seed=1, **kw)["x"]
    strong = x1 > 20.0                    # away from float-rounding-dominated nulls
    assert torch.allclose(x1[strong], x2[strong], atol=0.1)
    assert int(x1[0, 0].flatten().argmax()) == 30 * 64 + 32   # 90 m, 0 m/s


def test_noise_switch_keeps_the_random_stream():
    torch.manual_seed(0)
    kw = dict(n=2, seq_len=4, clutter=False, seed=3)
    on = generate_sequences(noise=True, **kw)
    off = generate_sequences(noise=False, **kw)
    assert torch.equal(on["traj"], off["traj"])        # second sequence too
    assert torch.equal(on["cls"], off["cls"])
    assert off["x"].amin() < on["x"].amin() - 20        # the noise floor is gone


def test_clutter_switch_removes_clutter():
    torch.manual_seed(0)
    kw = dict(n=1, seq_len=4, n_targets=1, target_class="steady", snr_db=20.0,
              noise=False, r0=90.0, v0=0.0, a=0.0, seed=5)
    with_clutter = generate_sequences(clutter=True, rho=0.9, nu=0.5, **kw)
    without = generate_sequences(clutter=False, **kw)
    assert without["x"].amin() < with_clutter["x"].amin() - 20


def test_per_target_lists_and_broadcasting():
    torch.manual_seed(0)
    out = generate_sequences(n=1, seq_len=4, r0=[40.0, 90.0, 150.0], v0=0.0, a=0.0,
                             target_class=["steady", "swerling1", "extended"], seed=2)
    assert int(out["n_targets"][0]) == 3
    assert torch.allclose(out["traj"][0, :3, 0, 0], torch.tensor([40.0, 90.0, 150.0]) / 3.0)
    assert out["cls"][0, :3].tolist() == [0, 1, 2]
    assert torch.allclose(out["v0"][0, :3], torch.zeros(3))


def test_unspecified_kinematics_stay_random_and_in_grid():
    torch.manual_seed(0)
    out = generate_sequences(n=6, seq_len=16, n_targets=2, r0=[60.0, 120.0], seed=4)
    assert torch.allclose(out["traj"][:, :2, 0, 0], torch.tensor([20.0, 40.0]).expand(6, 2))
    assert out["v0"][:, :2].std() > 0.1                  # drawn, not fixed
    assert (out["traj"][..., 0] >= 0).all() and (out["traj"][..., 0] <= 63).all()
    assert (out["traj"][..., 1] >= 0).all() and (out["traj"][..., 1] <= 63).all()


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        generate_sequences(r0=[40.0, 90.0], v0=[0.0, 1.0, 2.0])      # lengths disagree
    with pytest.raises(ValueError):
        generate_sequences(n_targets=2, r0=[40.0, 90.0, 150.0])
    with pytest.raises(ValueError):
        generate_sequences(n_targets=6)
    with pytest.raises(ValueError):
        generate_sequences(target_class="spaceship")
    with pytest.raises(ValueError):
        generate_sequences(rho=1.5)
    with pytest.raises(ValueError):
        generate_sequences(n_targets=1, r0=185.0, v0=7.0, a=0.0)      # walks off the grid
