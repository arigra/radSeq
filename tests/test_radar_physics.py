"""The simulator's parameters must be traceable to published specifications,
so these tests pin the derivations rather than any value fitted to recordings."""
import pytest
import torch

from src.radar_physics import (RADIAL_SPEC, clutter_cnr_db, sample_rcs,
                               target_snr_db, window)


def test_published_radial_specification():
    s = RADIAL_SPEC
    assert (s.n_range, s.n_doppler, s.n_rx, s.n_tx) == (512, 256, 16, 12)
    assert s.range_res_m == 0.2 and s.velocity_res_mps == 0.1
    assert s.frame_rate_hz == 5.0


def test_derived_waveform_quantities_match_the_grid():
    s = RADIAL_SPEC
    assert s.bandwidth_hz == pytest.approx(750e6)
    assert s.max_range_m == pytest.approx(102.2)
    assert s.frame_interval_s == pytest.approx(0.2)
    assert s.chirp_interval_s == pytest.approx(7.6e-5, rel=1e-2)


def test_looks_come_from_the_receive_array_not_a_fit():
    """radar_FFT carries one RD spectrum per receive antenna, so summing it in
    power integrates exactly n_rx looks."""
    assert RADIAL_SPEC.n_looks == RADIAL_SPEC.n_rx == 16


def test_target_snr_follows_the_radar_equation():
    near, far = target_snr_db(50.0, 10.0), target_snr_db(100.0, 10.0)
    expect = 40 * torch.log10(torch.tensor(2.0)).item()
    assert float(near - far) == pytest.approx(expect, rel=1e-4)
    big, small = target_snr_db(100.0, 20.0), target_snr_db(100.0, 10.0)
    assert float(big - small) == pytest.approx(10.0)


def test_clutter_falls_more_slowly_than_targets():
    """The illuminated ground patch grows with range, so clutter falls as
    1/R^3 while a point target falls as 1/R^4."""
    d_clutter = float(clutter_cnr_db(50.0) - clutter_cnr_db(100.0))
    d_target = float(target_snr_db(50.0, 10.0) - target_snr_db(100.0, 10.0))
    assert d_clutter < d_target
    expect = 30 * torch.log10(torch.tensor(2.0)).item()
    assert d_clutter == pytest.approx(expect, rel=1e-4)


def test_rcs_population_is_dominated_by_cars():
    torch.manual_seed(0)
    rcs, idx = sample_rcs(2000)
    assert (idx == 0).float().mean() > 0.7
    assert 8.0 < float(rcs[idx == 0].mean()) < 12.0


def test_hamming_window_matches_radials_processing_code():
    w = window(256, "hamming")
    assert float(w[0]) == pytest.approx(0.08, abs=1e-6)
    assert float(w.max()) == pytest.approx(1.0, abs=1e-3)
    assert torch.equal(window(64, "none"), torch.ones(64))
    with pytest.raises(ValueError):
        window(16, "blackman")


# ---- the simulator driven by the specification, with nothing fitted --------

def test_spec_driven_simulator_takes_its_settings_from_the_specification():
    from src.simulator import spec_simulator
    sim = spec_simulator(RADIAL_SPEC, seq_len=2)
    assert (sim.N, sim.K) == (512, 256)
    assert sim.n_looks == 16            # receive channels, not a fitted value
    assert sim.Tf == pytest.approx(0.2)  # 5 fps
    assert sim.window == "hamming"
    assert sim.range_gain_db is None     # nothing fitted to recordings


def test_windowing_suppresses_sidelobes():
    """RADIal windows both FFT axes (rpl.py lines 128-131). Without it an
    off-grid target leaks across the map and inflates the background.

    The target must be placed OFF bin centre: an on-grid tone has exact DFT
    nulls at every other bin, so an unwindowed map looks artificially clean
    and the comparison is degenerate.
    """
    import dataclasses
    from src.simulator import spec_simulator

    def sidelobe_db(spec):
        sim = spec_simulator(spec, seq_len=1, clutter=False, noise=False)
        torch.manual_seed(0)
        x = sim.gen_sequence(r0=torch.tensor([50.13]), v0=torch.tensor([5.07]),
                             a=torch.tensor([0.0]),
                             rcs_dbsm=torch.tensor([10.0]))["x"][0]
        row = x[int(round(50.13 / 0.2))]
        di, peak = int(row.argmax()), float(row.max())
        return max(float(row[di - 4]), float(row[di + 4])) - peak

    plain = sidelobe_db(dataclasses.replace(RADIAL_SPEC, window="none"))
    windowed = sidelobe_db(RADIAL_SPEC)
    assert windowed < plain - 15.0, (plain, windowed)


def test_target_brightness_follows_range_not_a_fitted_constant():
    """A receding target must dim along its own track, by the radar equation.
    The shipped simulator held one gain constant for the whole sequence."""
    from src.simulator import spec_simulator
    sim = spec_simulator(RADIAL_SPEC, seq_len=8, clutter=False)
    torch.manual_seed(0)
    out = sim.gen_sequence(r0=torch.tensor([25.0]), v0=torch.tensor([10.0]),
                           a=torch.tensor([0.0]), rcs_dbsm=torch.tensor([10.0]),
                           cls=torch.tensor([0]))
    x = out["x"]
    # 25 m -> 25 + 10*7*0.2 = 39 m, so 40*log10(39/25) ~ 7.7 dB of dimming
    assert float(x[0].max()) > float(x[-1].max()) + 4.0


def test_rcs_is_reported_so_labels_can_carry_target_strength():
    from src.simulator import spec_simulator
    sim = spec_simulator(RADIAL_SPEC, seq_len=2)
    out = sim.gen_sequence(n_targets=2)
    assert out["rcs_dbsm"].shape == (2,)
    assert torch.isfinite(out["rcs_dbsm"]).all()
