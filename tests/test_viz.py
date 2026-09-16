import matplotlib
import matplotlib.pyplot as plt
import pytest

from src.simulator import generate_sequences
from src.viz import show_sequence

matplotlib.use("Agg")   # tests only; the module itself forces no backend


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.mark.parametrize("frames", [(0,), (0, 1, 2, 3)])
def test_show_sequence_single_and_multiple_frames(frames):
    """One frame used to break: plt.subplots returns a bare Axes, not an array."""
    data = generate_sequences(n=1, seq_len=4, n_targets=2, clutter=False,
                              noise=False, seed=0)
    show_sequence(data, frames=frames, title="t")
    fig = plt.gcf()
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == len(frames)
    assert all(len(ax.collections) == 1 for ax in image_axes)   # the target markers


def test_show_rows_with_histograms():
    from src.viz import show_rows
    clean = generate_sequences(n=1, seq_len=4, n_targets=1, clutter=False, noise=False, seed=0)
    noisy = generate_sequences(n=1, seq_len=4, n_targets=1, clutter=False, noise=True, seed=0)
    show_rows([clean, noisy], ["clean", "noisy"], frames=(0, 1), hist=True)
    fig = plt.gcf()
    assert len([ax for ax in fig.axes if ax.images]) == 4          # 2 scenes x 2 frames
    assert len([ax for ax in fig.axes if ax.patches]) == 2         # one histogram per scene


def test_show_kinematics_and_its_peak_finder():
    import numpy as np
    from src.viz import _target_peaks, show_kinematics
    const = generate_sequences(n_targets=1, target_class="steady", snr_db=20,
                               clutter=False, noise=False, r0=60, v0=2.5, a=0.0, seed=0)
    show_kinematics([const], ["constant velocity"])
    assert len(plt.gcf().axes) == 4
    traj = const["traj"][0, 0].numpy()
    pos, _ = _target_peaks(const["x"][0].numpy(), traj)
    assert np.abs(pos - traj).max() <= 1.5          # label and map agree within the straddle


def test_second_difference_axis_starts_at_zero():
    from src.viz import show_kinematics
    d = generate_sequences(n_targets=1, target_class="steady", snr_db=20, seq_len=6,
                           clutter=False, noise=False, r0=90, v0=1.5, a=0.2, seed=0)
    show_kinematics([d], ["x"])
    assert plt.gcf().axes[2].get_ylim()[0] == 0


def test_show_cuts_uses_physical_axes():
    from src.viz import show_cuts
    d = generate_sequences(n_targets=1, target_class="steady", snr_db=20, seq_len=2,
                           clutter=False, noise=False, r0=60, v0=2.5, a=0, seed=0)
    show_cuts(d)
    image_ax = [ax for ax in plt.gcf().axes if ax.images][0]
    assert image_ax.get_xlabel() == "radial velocity (m/s)"
    assert image_ax.get_ylabel() == "range (m)"


def test_show_clutter_correlation_follows_rho():
    from src.viz import show_clutter_correlation
    show_clutter_correlation(rhos=(0.0, 0.9), n_seq=2, seq_len=6)
    ax = plt.gcf().axes[0]
    signal = ax.lines[0].get_ydata()
    assert signal[1] - signal[0] > 0.5            # clutter signal tracks rho


def test_show_range_profile_extended_spans_three_bins():
    import numpy as np
    from src.viz import show_range_profile
    kw = dict(n_targets=1, snr_db=20, seq_len=1, clutter=False, noise=False,
              r0=90, v0=0, a=0, seed=0)
    point = generate_sequences(target_class="steady", **kw)
    extended = generate_sequences(target_class="extended", **kw)
    show_range_profile([point, extended], ["steady", "extended"])
    p_line, e_line = plt.gcf().axes[0].lines[:2]
    p, e = np.asarray(p_line.get_ydata()), np.asarray(e_line.get_ydata())
    centre = len(p) // 2                          # the target bin
    assert p[centre] - max(p[centre - 1], p[centre + 1]) > 20     # point: neighbours are nulls
    assert e[centre] - max(e[centre - 1], e[centre + 1]) < 8      # extended: flanks ~3 dB down


def test_show_brightness_one_line_per_scene():
    from src.viz import show_brightness
    kw = dict(n_targets=1, snr_db=15, seq_len=6, clutter=False, noise=False,
              r0=90, v0=1.5, a=0.2, seed=3)
    scenes = [generate_sequences(target_class=c, **kw) for c in ("steady", "swerling1")]
    show_brightness(scenes, ["steady", "swerling1"])
    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 2 and len(ax.lines[0].get_ydata()) == 6


def test_show_normalization_draws_both_histograms():
    from src.viz import show_normalization
    d = generate_sequences(n=2, seq_len=4, seed=0)
    show_normalization(d["x"], {"mean": 41.48, "std": 11.07})
    axes = plt.gcf().axes
    assert len(axes) == 2 and all(ax.patches for ax in axes)
    assert axes[0].get_xlabel() == "dB" and axes[1].get_xlabel() == "normalised value"


def test_show_rows_accepts_unlabelled_generated_scenes():
    """Unconditional DiT samples have maps but no target labels."""
    import torch
    from src.viz import show_rows
    real = generate_sequences(n=1, seq_len=4, n_targets=1, clutter=False, noise=False, seed=0)
    generated = {"x": torch.randn(1, 4, 64, 64)}
    show_rows([real, generated], ["simulator", "DiT"], frames=(0, 1))
    image_axes = [ax for ax in plt.gcf().axes if ax.images]
    assert len(image_axes) == 4
    assert [len(ax.collections) for ax in image_axes] == [1, 1, 0, 0]   # circles only on labelled row
