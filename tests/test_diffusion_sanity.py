import json

import pytest
import torch

from scripts.diag_diffusion_sanity import (
    GaussianOracle,
    TwoPointOracle,
    run_sanity,
    write_report,
)
from src.diffusion import GaussianDiffusion


def test_oracles_return_finite_epsilon_with_expected_shape():
    diffusion = GaussianDiffusion(timesteps=100)
    xt = torch.tensor([[-1.0], [0.0], [1.0]])
    t = torch.tensor([0, 50, 99])

    gaussian = GaussianOracle(diffusion, mean=0.25, std=0.5)(xt, t)
    two_point = TwoPointOracle(diffusion, magnitude=2.0)(xt, t)

    assert gaussian.shape == xt.shape
    assert two_point.shape == xt.shape
    assert torch.isfinite(gaussian).all()
    assert torch.isfinite(two_point).all()


@pytest.fixture(scope="module")
def sanity_report():
    # Smaller than the CLI default while retaining ample statistical power.
    return run_sanity(n=4_096, timesteps=200, ddim_steps=50, seed=314159)


def test_forward_q_and_pred_x0_analytic_checks_pass(sanity_report):
    forward = sanity_report["checks"]["forward_process"]
    assert forward["q_marginals"]["passed"]
    assert forward["q_pred_x0_roundtrip"]["passed"]


def test_ddim_analytic_oracle_checks_pass(sanity_report):
    checks = sanity_report["checks"]
    assert checks["ddim_gaussian_oracle"]["passed"]
    assert checks["ddim_two_point_oracle"]["passed"]
    assert sanity_report["passed"]


def test_report_is_json_serializable(sanity_report, tmp_path):
    output = write_report(sanity_report, tmp_path / "sanity.json")
    loaded = json.loads(output.read_text())
    assert loaded["passed"] is True
    assert loaded["config"]["device"] == "cpu"
    assert loaded["scope_limit"]
