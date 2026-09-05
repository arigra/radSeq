import pytest
import torch

from scripts.evaluate_ablations import (
    Arm,
    _json_safe,
    distribution_stats,
    grid_seam_stats,
    parse_arm,
)


def test_parse_arm_supports_optional_weight_and_reduction_overrides():
    assert parse_arm("base=checkpoints/base.pt") == Arm(
        "base", __import__("pathlib").Path("checkpoints/base.pt"), None, None)
    assert parse_arm(
        "ema=checkpoints/run.pt,weights=ema,reduction=tile") == Arm(
            "ema", __import__("pathlib").Path("checkpoints/run.pt"),
            "ema", "tile")


def test_parse_arm_rejects_unknown_or_duplicate_options():
    with pytest.raises(ValueError, match="unknown arm option"):
        parse_arm("x=a.pt,seed=1")
    with pytest.raises(ValueError, match="duplicate arm option"):
        parse_arm("x=a.pt,weights=raw,weights=ema")


def test_distribution_stats_reports_population_location_and_sample_std():
    result = distribution_stats(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert result["mean"] == pytest.approx(1.5)
    assert result["std"] == pytest.approx(torch.tensor([0., 1., 2., 3.]).std().item())
    assert result["p50"] == pytest.approx(1.5)


def test_grid_seam_stats_detects_periodic_tile_discontinuities():
    image = torch.zeros(1, 1, 8, 8)
    image[..., 4:, :] += 4.0
    image[..., :, 4:] += 4.0
    result = grid_seam_stats(image, period=4)
    assert result["boundary_mean_abs_neighbor_difference"] == pytest.approx(4.0)
    assert result["interior_mean_abs_neighbor_difference"] == 0.0
    assert result["boundary_to_interior_ratio"] is None


def test_json_safe_replaces_nonfinite_metrics():
    assert _json_safe({"a": float("nan"), "b": [float("inf"), 2.0]}) == {
        "a": None, "b": [None, 2.0]}
