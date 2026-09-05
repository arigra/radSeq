"""CPU-only analytic sanity checks for the diffusion and DDIM machinery.

This diagnostic deliberately does not train or load a neural network.  It uses
closed-form Bayes-optimal epsilon predictors for distributions whose target
moments/support are known:

* a Gaussian checks that DDIM preserves a continuous location and scale;
* an equal mixture of point masses at ``-2`` and ``+2`` checks nonlinear,
  bimodal generation without allowing a Gaussian-looking answer to pass.

Together with direct checks of ``q_sample`` and ``pred_x0``, this separates the
repository's diffusion algebra/sampler from model optimization and radar-model
architecture.  A passing report says the plumbing works on these controls; it
does not say that a learned radar denoiser is accurate at high noise.

Run from the repository root:

    python scripts/diag_diffusion_sanity.py

The default run is deterministic, CPU-only, and writes
``samples/diag_diffusion_sanity.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

# Keep the documented ``python scripts/...`` invocation working even when the
# repository has not been installed as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusion import GaussianDiffusion


GAUSSIAN_MEAN = 0.25
GAUSSIAN_STD = 0.5
TWO_POINT_MAGNITUDE = 2.0


def _schedule_value(diffusion: GaussianDiffusion, t: torch.Tensor,
                    x: torch.Tensor) -> torch.Tensor:
    """Broadcast alpha_bar[t] without relying on private diffusion methods."""
    values = diffusion.alphas_bar.to(device=x.device, dtype=x.dtype)[t]
    return values.view(-1, *([1] * (x.ndim - 1)))


class GaussianOracle:
    """Bayes-optimal epsilon predictor for x0 ~ Normal(mean, std**2)."""

    def __init__(self, diffusion: GaussianDiffusion, mean: float, std: float):
        self.diffusion = diffusion
        self.mean = mean
        self.variance = std ** 2

    def __call__(self, xt: torch.Tensor, t: torch.Tensor,
                 cond: Any = None) -> torch.Tensor:
        del cond
        ab = _schedule_value(self.diffusion, t, xt)
        noise_var = 1.0 - ab
        xt_var = ab * self.variance + noise_var
        posterior_x0 = (
            self.mean
            + ab.sqrt() * self.variance / xt_var
            * (xt - ab.sqrt() * self.mean)
        )
        return (xt - ab.sqrt() * posterior_x0) / noise_var.sqrt()


class TwoPointOracle:
    """Bayes-optimal epsilon predictor for P(x0=-m)=P(x0=+m)=1/2."""

    def __init__(self, diffusion: GaussianDiffusion, magnitude: float):
        self.diffusion = diffusion
        self.magnitude = magnitude

    def __call__(self, xt: torch.Tensor, t: torch.Tensor,
                 cond: Any = None) -> torch.Tensor:
        del cond
        ab = _schedule_value(self.diffusion, t, xt)
        noise_var = 1.0 - ab
        # The posterior log-odds are 2*m*sqrt(ab)*xt/(1-ab), hence
        # E[x0|xt] = m*tanh(m*sqrt(ab)*xt/(1-ab)).
        posterior_x0 = self.magnitude * torch.tanh(
            self.magnitude * ab.sqrt() * xt / noise_var
        )
        return (xt - ab.sqrt() * posterior_x0) / noise_var.sqrt()


def _selected_timesteps(timesteps: int) -> list[int]:
    return sorted(set([0, timesteps // 4, timesteps // 2,
                       3 * timesteps // 4, timesteps - 1]))


def _forward_checks(diffusion: GaussianDiffusion, n: int,
                    seed: int) -> dict[str, Any]:
    """Check analytic q marginals and the q/pred_x0 algebraic inverse."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Float64 makes the inverse check about the formula instead of expected
    # float32 cancellation at alpha_bar ~= 0.
    x0 = (GAUSSIAN_MEAN + GAUSSIAN_STD
          * torch.randn((n, 1), generator=generator, dtype=torch.float64))
    eps = torch.randn((n, 1), generator=generator, dtype=torch.float64)

    marginal_rows: dict[str, dict[str, float]] = {}
    max_mean_z = 0.0
    max_std_z = 0.0
    max_roundtrip_error = 0.0
    for ti in _selected_timesteps(diffusion.T):
        t = torch.full((n,), ti, dtype=torch.long)
        xt = diffusion.q_sample(x0, t, eps)
        x0_hat = diffusion.pred_x0(xt, t, eps)
        ab = float(diffusion.alphas_bar[ti])
        expected_mean = math.sqrt(ab) * GAUSSIAN_MEAN
        expected_std = math.sqrt(ab * GAUSSIAN_STD ** 2 + 1.0 - ab)
        observed_mean = float(xt.mean())
        observed_std = float(xt.std(unbiased=True))
        mean_se = expected_std / math.sqrt(n)
        std_se = expected_std / math.sqrt(2.0 * (n - 1))
        mean_z = abs(observed_mean - expected_mean) / mean_se
        std_z = abs(observed_std - expected_std) / std_se
        roundtrip_error = float((x0_hat - x0).abs().max())
        max_mean_z = max(max_mean_z, mean_z)
        max_std_z = max(max_std_z, std_z)
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
        marginal_rows[str(ti)] = {
            "alpha_bar": ab,
            "expected_mean": expected_mean,
            "observed_mean": observed_mean,
            "mean_error_standard_scores": mean_z,
            "expected_std": expected_std,
            "observed_std": observed_std,
            "std_error_standard_scores": std_z,
            "roundtrip_max_abs_error": roundtrip_error,
        }

    marginal_threshold_z = 5.0
    roundtrip_threshold = 1e-10
    return {
        "distribution": {
            "name": "normal",
            "x0_mean": GAUSSIAN_MEAN,
            "x0_std": GAUSSIAN_STD,
        },
        "timesteps": marginal_rows,
        "q_marginals": {
            "max_mean_error_standard_scores": max_mean_z,
            "max_std_error_standard_scores": max_std_z,
            "threshold_standard_scores": marginal_threshold_z,
            "passed": (max_mean_z <= marginal_threshold_z
                       and max_std_z <= marginal_threshold_z),
        },
        "q_pred_x0_roundtrip": {
            "dtype": "float64",
            "max_abs_error": max_roundtrip_error,
            "threshold": roundtrip_threshold,
            "passed": max_roundtrip_error <= roundtrip_threshold,
        },
    }


def _ddim_gaussian_check(diffusion: GaussianDiffusion, n: int, steps: int,
                         seed: int) -> dict[str, Any]:
    oracle = GaussianOracle(diffusion, GAUSSIAN_MEAN, GAUSSIAN_STD)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sample = diffusion.ddim_sample(
            oracle, (n, 1), torch.device("cpu"), steps=steps
        ).flatten()

    observed_mean = float(sample.mean())
    observed_std = float(sample.std(unbiased=True))
    mean_abs_error = abs(observed_mean - GAUSSIAN_MEAN)
    std_relative_error = abs(observed_std - GAUSSIAN_STD) / GAUSSIAN_STD
    # The deterministic finite-step DDIM discretization is expected to be
    # mildly under-dispersed (about 3% here at 50 steps).  These tolerances
    # include that known discretization error and Monte Carlo error.
    mean_threshold = 0.04
    std_relative_threshold = 0.07
    return {
        "distribution": {
            "name": "normal",
            "target_mean": GAUSSIAN_MEAN,
            "target_std": GAUSSIAN_STD,
        },
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "mean_abs_error": mean_abs_error,
        "std_relative_error": std_relative_error,
        "thresholds": {
            "mean_abs_error": mean_threshold,
            "std_relative_error": std_relative_threshold,
        },
        "passed": (mean_abs_error <= mean_threshold
                   and std_relative_error <= std_relative_threshold),
    }


def _ddim_two_point_check(diffusion: GaussianDiffusion, n: int, steps: int,
                          seed: int) -> dict[str, Any]:
    oracle = TwoPointOracle(diffusion, TWO_POINT_MAGNITUDE)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sample = diffusion.ddim_sample(
            oracle, (n, 1), torch.device("cpu"), steps=steps
        ).flatten()

    positive_fraction = float((sample > 0).double().mean())
    support_mae = float((sample.abs() - TWO_POINT_MAGNITUDE).abs().mean())
    observed_mean = float(sample.mean())
    observed_std = float(sample.std(unbiased=False))
    balance_error = abs(positive_fraction - 0.5)
    support_mae_threshold = 1e-4
    balance_threshold = 0.03
    std_relative_error = abs(observed_std - TWO_POINT_MAGNITUDE) / TWO_POINT_MAGNITUDE
    std_relative_threshold = 0.01
    return {
        "distribution": {
            "name": "equal_two_point_mixture",
            "support": [-TWO_POINT_MAGNITUDE, TWO_POINT_MAGNITUDE],
            "target_positive_fraction": 0.5,
            "target_mean": 0.0,
            "target_std": TWO_POINT_MAGNITUDE,
        },
        "observed_positive_fraction": positive_fraction,
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "support_mean_abs_error": support_mae,
        "balance_abs_error": balance_error,
        "std_relative_error": std_relative_error,
        "thresholds": {
            "support_mean_abs_error": support_mae_threshold,
            "balance_abs_error": balance_threshold,
            "std_relative_error": std_relative_threshold,
        },
        "passed": (support_mae <= support_mae_threshold
                   and balance_error <= balance_threshold
                   and std_relative_error <= std_relative_threshold),
    }


def run_sanity(n: int = 16_384, timesteps: int = 1_000,
               ddim_steps: int = 50, seed: int = 20260815) -> dict[str, Any]:
    """Run all checks and return a JSON-serializable report."""
    if n < 1_024:
        raise ValueError("n must be at least 1024 for stable moment checks")
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if not 2 <= ddim_steps <= timesteps:
        raise ValueError("ddim_steps must be between 2 and timesteps")

    diffusion = GaussianDiffusion(timesteps=timesteps)
    schedule_monotone = bool(torch.all(
        diffusion.alphas_bar[1:] <= diffusion.alphas_bar[:-1]
    ))
    schedule = {
        "alpha_bar_first": float(diffusion.alphas_bar[0]),
        "alpha_bar_last": float(diffusion.alphas_bar[-1]),
        "monotone_nonincreasing": schedule_monotone,
        "passed": (schedule_monotone
                   and float(diffusion.alphas_bar[0]) > 0.99
                   and float(diffusion.alphas_bar[-1]) < 0.01),
    }
    forward = _forward_checks(diffusion, n=n, seed=seed)
    gaussian = _ddim_gaussian_check(
        diffusion, n=n, steps=ddim_steps, seed=seed + 1
    )
    two_point = _ddim_two_point_check(
        diffusion, n=n, steps=ddim_steps, seed=seed + 2
    )
    checks = {
        "schedule": schedule,
        "forward_process": forward,
        "ddim_gaussian_oracle": gaussian,
        "ddim_two_point_oracle": two_point,
    }
    passed = (
        schedule["passed"]
        and forward["q_marginals"]["passed"]
        and forward["q_pred_x0_roundtrip"]["passed"]
        and gaussian["passed"]
        and two_point["passed"]
    )
    interpretation = (
        "PASS: q_sample, pred_x0, and DDIM reproduce the analytic CPU "
        "controls. A radar sampling failure is therefore not explained by "
        "a basic diffusion-plumbing error covered by these tests."
        if passed else
        "FAIL: at least one analytic diffusion control failed; inspect the "
        "failed check before attributing sampling problems to the model."
    )
    return {
        "schema_version": 1,
        "purpose": (
            "Trained-model-independent checks of q_sample, pred_x0, and DDIM "
            "against analytically known one-dimensional distributions."
        ),
        "config": {
            "device": "cpu",
            "dtype_sampling": "float32",
            "n": n,
            "timesteps": timesteps,
            "ddim_steps": ddim_steps,
            "seed": seed,
        },
        "checks": checks,
        "passed": passed,
        "scope_limit": (
            "This does not test a learned denoiser, optimization, radar data, "
            "conditioning, patch aggregation, or model capacity."
        ),
        "interpretation": interpretation,
    }


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16_384)
    parser.add_argument("--timesteps", type=int, default=1_000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out", default="samples/diag_diffusion_sanity.json")
    args = parser.parse_args()

    report = run_sanity(n=args.n, timesteps=args.timesteps,
                        ddim_steps=args.ddim_steps, seed=args.seed)
    output = write_report(report, args.out)
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
