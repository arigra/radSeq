import math
from pathlib import Path

import pytest
import torch

from scripts.diag_denoising_by_timestep import (
    PairMoments,
    build_region_masks,
    evaluate_arm,
    parse_arms,
    parse_timesteps,
    summarize_regions,
    target_frame_metrics,
)
from src.diffusion import GaussianDiffusion


def test_build_region_masks_ignores_padding_and_clips_boundaries():
    # The second padded target sits at (6, 6) but n_targets=1, so it must not
    # affect either mask.  The valid target moves from the center to a corner.
    traj = torch.tensor([[[[3.0, 3.0], [0.0, 0.0]],
                          [[6.0, 6.0], [6.0, 6.0]]]])
    target, clutter = build_region_masks(
        traj, torch.tensor([1]), 7, 7,
        target_half_width=1, clutter_guard_half_width=2)

    assert target.shape == (1, 2, 7, 7)
    assert target[0, 0].sum() == 9
    assert target[0, 1].sum() == 4
    assert clutter[0, 0].sum() == 24  # outside a centered 5x5 guard
    assert clutter[0, 1].sum() == 40  # outside a corner-clipped 3x3 guard
    assert not (target & clutter).any()
    assert not target[0, 0, 6, 6]


def test_build_region_masks_rejects_a_guard_smaller_than_target_window():
    traj = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="guard"):
        build_region_masks(
            traj, torch.tensor([1]), 4, 4,
            target_half_width=2, clutter_guard_half_width=1)


def test_pair_moments_streams_exact_error_and_correlation():
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0])
    prediction = torch.tensor([2.0, 4.0, 6.0, 8.0])
    moments = PairMoments()
    moments.update(reference[:2], prediction[:2])
    moments.update(reference[2:], prediction[2:])
    summary = moments.summary()

    assert summary["count"] == 4
    assert summary["reference_mean"] == pytest.approx(2.5)
    assert summary["prediction_std"] == pytest.approx(2 * math.sqrt(1.25))
    assert summary["mse"] == pytest.approx(7.5)
    assert summary["mae"] == pytest.approx(2.5)
    assert summary["bias"] == pytest.approx(2.5)
    assert summary["pearson_r"] == pytest.approx(1.0)


def test_summarize_regions_reports_error_ratio_and_db_contrast():
    acc = {name: PairMoments() for name in ("all", "target_window", "clutter_only")}
    reference = torch.tensor([[[[2.0, 0.0]]]])
    prediction = torch.tensor([[[[1.0, 0.5]]]])
    target = torch.tensor([[[[True, False]]]])
    clutter = ~target
    acc["all"].update(reference, prediction)
    acc["target_window"].update(reference, prediction, target)
    acc["clutter_only"].update(reference, prediction, clutter)

    summary = summarize_regions(acc, data_std_db=3.0)
    assert summary["target_to_clutter_mse_ratio"] == pytest.approx(4.0)
    contrast = summary["mean_target_minus_clutter_db"]
    assert contrast["reference"] == pytest.approx(6.0)
    assert contrast["prediction"] == pytest.approx(1.5)
    assert contrast["retained_fraction"] == pytest.approx(0.25)


def test_target_frame_metrics_uses_only_real_targets():
    sequences = torch.zeros(1, 2, 7, 7)
    sequences[0, 0, 3, 3] = 20.0  # detected
    sequences[0, 1, 2, 2] = 11.0  # below the 12 dB detector threshold
    traj = torch.tensor([[[[3.0, 3.0], [2.0, 2.0]],
                          [[6.0, 6.0], [6.0, 6.0]]]])

    metrics = target_frame_metrics(
        sequences, traj, torch.tensor([1]), contrast_half_width=0)
    assert metrics["target_frames"] == 2
    assert metrics["found_target_frames"] == 1
    assert metrics["target_frame_recall"] == pytest.approx(0.5)
    assert metrics["detected_peaks"] == 1
    assert metrics["peak_precision"] == pytest.approx(1.0)
    assert metrics["target_peak_minus_frame_median_db"]["mean"] == pytest.approx(15.5)


def test_cli_parsers_preserve_order_deduplicate_and_validate():
    assert parse_timesteps("10,0,10,999", 1000) == [10, 0, 999]
    with pytest.raises(ValueError, match="outside"):
        parse_timesteps("1000", 1000)
    assert parse_arms(["base=a.pt", "control=b.pt"]) == [
        ("base", Path("a.pt")),
        ("control", Path("b.pt")),
    ]


def test_evaluate_arm_with_exact_noise_prediction_has_zero_error():
    x0 = torch.zeros(1, 2, 7, 7)
    x0[0, :, 3, 3] = 3.0
    noise = torch.randn_like(x0)
    traj = torch.tensor([[[[3.0, 3.0], [3.0, 3.0]]]])
    n_targets = torch.tensor([1])
    target, clutter = build_region_masks(
        traj, n_targets, 7, 7,
        target_half_width=1, clutter_guard_half_width=2)

    class ExactNoise(torch.nn.Module):
        def forward(self, xt, timestep, cond=None):
            return noise.to(xt.device)

    result = evaluate_arm(
        ExactNoise(),
        GaussianDiffusion(10),
        x0,
        noise,
        traj,
        n_targets,
        target,
        clutter,
        timesteps=[5],
        batch_size=1,
        device=torch.device("cpu"),
        data_mean_db=0.0,
        data_std_db=5.0,
        tolerance_bins=2.0,
        threshold_db=12.0,
        max_peaks=5,
        contrast_half_width=1,
    )["5"]

    assert result["epsilon_prediction"]["regions"]["all"]["mse"] == 0.0
    assert result["x0_reconstruction_raw"]["regions"]["all"]["mse"] < 1e-12
    assert result["fraction_changed_by_sampler_clamp"] == 0.0
    assert result["sampler_clamped_target_frame_metrics"][
        "target_frame_recall"] == 1.0
