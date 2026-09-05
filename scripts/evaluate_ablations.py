"""Evaluate trained ablation checkpoints under one paired sampling protocol.

Each ``--arm`` has the form::

    NAME=CHECKPOINT[,weights=raw|ema|auto][,reduction=mean|tile]

The same seed is reset before sampling every arm, so models with the same output
shape receive identical initial Gaussian noise. Results are written after each
arm so a later failure does not discard completed evaluations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from src.dataset import denormalize
from src.diffusion import GaussianDiffusion
from src.eval.metrics import evaluate_sequences
from src.sample import _set_patch_reduction, select_checkpoint_state
from src.train import build_model


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: Path
    weights: str | None = None
    reduction: str | None = None


def parse_arm(value: str) -> Arm:
    """Parse one arm specification and reject ambiguous/unknown options."""
    if "=" not in value:
        raise ValueError(f"arm must be NAME=CHECKPOINT, got {value!r}")
    name, remainder = value.split("=", 1)
    name = name.strip()
    pieces = [piece.strip() for piece in remainder.split(",")]
    path = pieces[0]
    if not name or not path:
        raise ValueError(f"arm must have a non-empty name and path, got {value!r}")
    options: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            raise ValueError(f"arm option must be KEY=VALUE, got {piece!r}")
        key, option_value = (part.strip() for part in piece.split("=", 1))
        if key in options:
            raise ValueError(f"duplicate arm option {key!r}")
        options[key] = option_value
    unknown = set(options) - {"weights", "reduction"}
    if unknown:
        raise ValueError(f"unknown arm option(s): {', '.join(sorted(unknown))}")
    weights = options.get("weights")
    reduction = options.get("reduction")
    if weights not in (None, "raw", "ema", "auto"):
        raise ValueError(f"unknown weights {weights!r}")
    if reduction not in (None, "mean", "tile"):
        raise ValueError(f"unknown reduction {reduction!r}")
    return Arm(name, Path(path), weights, reduction)


def distribution_stats(x: torch.Tensor) -> dict[str, float]:
    """Stable normalized-distribution summary without retaining a histogram."""
    values = x.detach().float().reshape(-1).cpu()
    if values.numel() > 16_000_000:
        values = values[:: values.numel() // 16_000_000 + 1]
    quantiles = torch.quantile(
        values, torch.tensor([0.5, 0.75, 0.99, 0.999]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "p999": float(quantiles[3]),
    }


def grid_seam_stats(x: torch.Tensor, period: int = 8) -> dict[str, float | None]:
    """Compare neighbor jumps on a periodic tile grid with all other jumps."""
    if x.ndim < 2 or period <= 1:
        raise ValueError("seam stats require image tensors and period > 1")
    height, width = x.shape[-2:]
    if height < period or width < period:
        raise ValueError("seam period exceeds image dimensions")
    x = x.detach().float().cpu()
    row_diffs = (x[..., 1:, :] - x[..., :-1, :]).abs()
    col_diffs = (x[..., :, 1:] - x[..., :, :-1]).abs()
    row_is_seam = torch.arange(1, height) % period == 0
    col_is_seam = torch.arange(1, width) % period == 0
    boundary = torch.cat([
        row_diffs[..., row_is_seam, :].reshape(-1),
        col_diffs[..., :, col_is_seam].reshape(-1),
    ])
    interior = torch.cat([
        row_diffs[..., ~row_is_seam, :].reshape(-1),
        col_diffs[..., :, ~col_is_seam].reshape(-1),
    ])
    boundary_mean = float(boundary.mean())
    interior_mean = float(interior.mean())
    return {
        "period_pixels": period,
        "boundary_mean_abs_neighbor_difference": boundary_mean,
        "interior_mean_abs_neighbor_difference": interior_mean,
        "boundary_to_interior_ratio": (
            boundary_mean / interior_mean if interior_mean else None),
    }


def load_raw_validation(cache_dir: Path, limit: int) -> torch.Tensor:
    sequences: list[torch.Tensor] = []
    for path in sorted(cache_dir.glob("val_*.pt")):
        for item in torch.load(path, map_location="cpu"):
            sequences.append(item["x"].float())
            if len(sequences) == limit:
                return torch.stack(sequences)
    raise ValueError(f"requested {limit} validation sequences, found {len(sequences)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
    temporary.replace(path)


def _json_safe(value):
    """Replace non-finite metric values with JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@torch.inference_mode()
def evaluate_arm(
    arm: Arm,
    device: torch.device,
    n: int,
    steps: int,
    seed: int,
    real_raw: torch.Tensor,
    mean: float,
    std: float,
) -> dict:
    checkpoint = torch.load(arm.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(config, device)
    selected = select_checkpoint_state(
        checkpoint, weights=arm.weights)
    model.load_state_dict(selected)
    _set_patch_reduction(model, arm.reduction)
    model.eval()
    effective_weights = arm.weights
    if effective_weights is None:
        effective_weights = config.get("sample", {}).get("weights", "raw")
    if effective_weights == "auto":
        effective_weights = "ema" if checkpoint.get("ema_model") else "raw"

    diffusion = GaussianDiffusion(config["diffusion"]["timesteps"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    generated = diffusion.ddim_sample(
        model, (n, config["data"]["seq_len"], 64, 64), device, steps=steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    generated_cpu = generated.cpu()
    generated_raw = denormalize(generated_cpu, {"mean": mean, "std": std})
    model_config = config["model"]
    result = {
        "checkpoint": str(arm.checkpoint),
        "checkpoint_sha256": _sha256(arm.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "architecture": model_config.get("architecture", "temporal_dit"),
        "attention_mode": model_config.get("attn_mode"),
        "patch": model_config.get("patch"),
        "stride": model_config.get("stride"),
        "patch_reduction": arm.reduction or model_config.get("patch_reduction", "mean"),
        "weights": effective_weights,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "ema_updates": checkpoint.get("ema_updates"),
        "sample_seconds": elapsed,
        "generated_normalized": distribution_stats(generated_cpu),
        "grid_seams": grid_seam_stats(generated_cpu),
        "metrics": evaluate_sequences(generated_raw, real_raw),
    }
    del generated, generated_cpu, generated_raw, model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="samples/ablation_comparison.json")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    try:
        arms = [parse_arm(value) for value in args.arm]
    except ValueError as exc:
        parser.error(str(exc))
    if len({arm.name for arm in arms}) != len(arms):
        parser.error("arm names must be unique")
    missing = [str(arm.checkpoint) for arm in arms if not arm.checkpoint.is_file()]
    if missing:
        parser.error("missing checkpoint(s): " + ", ".join(missing))

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    cache_dir = Path(args.cache)
    stats = torch.load(cache_dir / "stats.pt", map_location="cpu")
    mean, std = float(stats["mean"]), float(stats["std"])
    real_all = load_raw_validation(cache_dir, 2 * args.n)
    real_a, real_b = real_all[:args.n], real_all[args.n:]
    report = {
        "status": "running",
        "protocol": {
            "n_generated_per_arm": args.n,
            "ddim_steps": args.steps,
            "paired_initial_noise_seed": args.seed,
            "device": str(device),
            "normalization_mean_db": mean,
            "normalization_std_db": std,
        },
        "real_normalized": distribution_stats((real_b - mean) / std),
        "real_grid_seams": grid_seam_stats((real_b - mean) / std),
        "real_vs_real_metrics": evaluate_sequences(real_a, real_b),
        "arms": {},
    }
    output = Path(args.out)
    _write_json(report, output)
    for arm in arms:
        print(f"sampling {arm.name}: {arm.checkpoint}", flush=True)
        report["arms"][arm.name] = evaluate_arm(
            arm, device, args.n, args.steps, args.seed, real_b, mean, std)
        _write_json(report, output)
    report["status"] = "complete"
    _write_json(report, output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
