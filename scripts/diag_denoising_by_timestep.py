"""Measure denoising fidelity by diffusion timestep and spatial region.

This is a one-step diagnostic, not a generation benchmark.  A held-out real
sequence ``x0`` is noised to each requested timestep with one fixed epsilon
draw, and every checkpoint is asked to predict that same epsilon.  The report
separates pixels close to ground-truth target trajectories from clutter-only
pixels, so a good aggregate loss cannot hide a target-specific failure.

Default arms are the matched-12,500-step temporal baseline and the two
factorized-attention controls.  Run from the repository root, for example:

    python -m scripts.diag_denoising_by_timestep --device cuda

The default workload is intentionally an evaluation job (32 sequences x 11
timesteps x 3 checkpoints), so it should be scheduled on a GPU.  Unit tests
cover the mask, aggregation, and target-detection helpers without loading a
checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.diffusion import GaussianDiffusion
from src.eval.metrics import detect_peaks
from src.train import build_model


DEFAULT_ARMS = (
    ("baseline_temporal_d8", "checkpoints/phase1_wandb/epoch_0010.pt"),
    ("control_factorized_d5", "checkpoints/ctrl_factorized_d5/last.pt"),
    ("control_factorized_d6", "checkpoints/ctrl_factorized_d6/last.pt"),
)
DEFAULT_TIMESTEPS = "0,5,10,25,50,100,200,400,600,800,950"


def parse_arms(values: Iterable[str] | None) -> list[tuple[str, Path]]:
    """Parse repeatable ``NAME=CHECKPOINT`` arguments with useful errors."""
    raw = list(values) if values else [f"{name}={path}" for name, path in DEFAULT_ARMS]
    arms: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in raw:
        if "=" not in value:
            raise ValueError(f"arm must be NAME=CHECKPOINT, got {value!r}")
        name, path = value.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"arm must have a non-empty name and path, got {value!r}")
        if name in seen:
            raise ValueError(f"duplicate arm name: {name}")
        seen.add(name)
        arms.append((name, Path(path)))
    return arms


def parse_timesteps(value: str, total_timesteps: int) -> list[int]:
    """Return unique timesteps in input order, validating schedule bounds."""
    try:
        requested = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated timestep list: {value!r}") from exc
    if not requested:
        raise ValueError("at least one timestep is required")
    out: list[int] = []
    for timestep in requested:
        if not 0 <= timestep < total_timesteps:
            raise ValueError(
                f"timestep {timestep} is outside [0, {total_timesteps - 1}]")
        if timestep not in out:
            out.append(timestep)
    return out


def build_region_masks(
    traj: torch.Tensor,
    n_targets: torch.Tensor,
    height: int,
    width: int,
    target_half_width: int = 2,
    clutter_guard_half_width: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target-window and clutter-only masks from padded trajectories.

    ``traj`` is ``(B, M, L, 2)`` in ``(range_bin, doppler_bin)`` order and
    ``n_targets`` says how many entries of the padded M dimension are real.
    Target windows are clipped at image boundaries.  Clutter pixels are those
    outside a larger guard window around every target; pixels in the gap are
    intentionally ignored to reduce target-sidelobe contamination.
    """
    if traj.ndim != 4 or traj.shape[-1] != 2:
        raise ValueError("traj must have shape (B, M, L, 2)")
    if n_targets.ndim != 1 or n_targets.shape[0] != traj.shape[0]:
        raise ValueError("n_targets must have shape (B,)")
    if target_half_width < 0:
        raise ValueError("target_half_width must be non-negative")
    if clutter_guard_half_width < target_half_width:
        raise ValueError("clutter guard must be at least as wide as the target window")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    batch, max_targets, seq_len, _ = traj.shape
    target = torch.zeros(
        batch, seq_len, height, width, dtype=torch.bool, device=traj.device)
    guarded = torch.zeros_like(target)

    for bi in range(batch):
        count = int(n_targets[bi].item())
        if not 0 <= count <= max_targets:
            raise ValueError(
                f"n_targets[{bi}]={count} is outside padded target dimension {max_targets}")
        for mi in range(count):
            for li in range(seq_len):
                row = int(torch.round(traj[bi, mi, li, 0]).item())
                col = int(torch.round(traj[bi, mi, li, 1]).item())
                row = min(max(row, 0), height - 1)
                col = min(max(col, 0), width - 1)

                tr0 = max(0, row - target_half_width)
                tr1 = min(height, row + target_half_width + 1)
                tc0 = max(0, col - target_half_width)
                tc1 = min(width, col + target_half_width + 1)
                target[bi, li, tr0:tr1, tc0:tc1] = True

                gr0 = max(0, row - clutter_guard_half_width)
                gr1 = min(height, row + clutter_guard_half_width + 1)
                gc0 = max(0, col - clutter_guard_half_width)
                gc1 = min(width, col + clutter_guard_half_width + 1)
                guarded[bi, li, gr0:gr1, gc0:gc1] = True

    return target, ~guarded


@dataclass
class PairMoments:
    """Streaming paired statistics, avoiding storage of full model outputs."""

    count: int = 0
    reference_sum: float = 0.0
    prediction_sum: float = 0.0
    reference_sq_sum: float = 0.0
    prediction_sq_sum: float = 0.0
    cross_sum: float = 0.0
    error_sq_sum: float = 0.0
    error_abs_sum: float = 0.0
    error_sum: float = 0.0

    def update(
        self,
        reference: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if reference.shape != prediction.shape:
            raise ValueError("reference and prediction shapes differ")
        if mask is not None:
            if mask.shape != reference.shape:
                raise ValueError("mask must have the same shape as the paired tensors")
            reference = reference[mask]
            prediction = prediction[mask]
        else:
            reference = reference.reshape(-1)
            prediction = prediction.reshape(-1)
        if reference.numel() == 0:
            return

        # Float64 reductions keep correlations stable over millions of pixels.
        reference = reference.detach().double()
        prediction = prediction.detach().double()
        error = prediction - reference
        self.count += reference.numel()
        self.reference_sum += float(reference.sum())
        self.prediction_sum += float(prediction.sum())
        self.reference_sq_sum += float(reference.square().sum())
        self.prediction_sq_sum += float(prediction.square().sum())
        self.cross_sum += float((reference * prediction).sum())
        self.error_sq_sum += float(error.square().sum())
        self.error_abs_sum += float(error.abs().sum())
        self.error_sum += float(error.sum())

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0}
        n = float(self.count)
        ref_mean = self.reference_sum / n
        pred_mean = self.prediction_sum / n
        ref_var = max(self.reference_sq_sum / n - ref_mean**2, 0.0)
        pred_var = max(self.prediction_sq_sum / n - pred_mean**2, 0.0)
        covariance = self.cross_sum / n - ref_mean * pred_mean
        denom = math.sqrt(ref_var * pred_var)
        correlation = covariance / denom if denom > 0 else None
        mse = self.error_sq_sum / n
        return {
            "count": self.count,
            "reference_mean": ref_mean,
            "reference_std": math.sqrt(ref_var),
            "prediction_mean": pred_mean,
            "prediction_std": math.sqrt(pred_var),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.error_abs_sum / n,
            "bias": self.error_sum / n,
            "pearson_r": correlation,
        }


def _region_accumulators() -> dict[str, PairMoments]:
    return {name: PairMoments() for name in ("all", "target_window", "clutter_only")}


def _update_regions(
    accumulators: dict[str, PairMoments],
    reference: torch.Tensor,
    prediction: torch.Tensor,
    target_mask: torch.Tensor,
    clutter_mask: torch.Tensor,
) -> None:
    accumulators["all"].update(reference, prediction)
    accumulators["target_window"].update(reference, prediction, target_mask)
    accumulators["clutter_only"].update(reference, prediction, clutter_mask)


def summarize_regions(
    accumulators: dict[str, PairMoments], data_std_db: float | None = None
) -> dict:
    """Summarize all regions and expose target/background gaps explicitly."""
    regions = {name: accumulator.summary() for name, accumulator in accumulators.items()}
    target = regions["target_window"]
    clutter = regions["clutter_only"]
    target_mse = target.get("mse")
    clutter_mse = clutter.get("mse")
    ratio = (
        target_mse / clutter_mse
        if target_mse is not None and clutter_mse not in (None, 0.0)
        else None
    )
    out = {"regions": regions, "target_to_clutter_mse_ratio": ratio}
    if data_std_db is not None and target.get("count") and clutter.get("count"):
        reference_contrast = (
            target["reference_mean"] - clutter["reference_mean"]
        ) * data_std_db
        prediction_contrast = (
            target["prediction_mean"] - clutter["prediction_mean"]
        ) * data_std_db
        out["mean_target_minus_clutter_db"] = {
            "reference": reference_contrast,
            "prediction": prediction_contrast,
            "retained_fraction": (
                prediction_contrast / reference_contrast
                if reference_contrast != 0.0
                else None
            ),
        }
    return out


def target_frame_metrics(
    sequences_db: torch.Tensor,
    traj: torch.Tensor,
    n_targets: torch.Tensor,
    tolerance_bins: float = 2.0,
    threshold_db: float = 12.0,
    max_peaks: int = 5,
    contrast_half_width: int = 2,
) -> dict:
    """Target-aware per-frame peak precision, recall, and brightness contrast.

    Recall counts a target-frame as found when ``detect_peaks`` returns a peak
    within ``tolerance_bins`` of its ground-truth trajectory coordinate.  This
    is deliberately a per-frame diagnostic, not the filtered-track recall used
    for final generated-sequence scoring.  Contrast is the brightest value in
    the target window minus that frame's median, in dB.
    """
    if sequences_db.ndim != 4:
        raise ValueError("sequences_db must have shape (B, L, H, W)")
    if sequences_db.shape[:2] != (traj.shape[0], traj.shape[2]):
        raise ValueError("sequence and trajectory batch/time dimensions differ")
    sequences_db = sequences_db.detach().cpu()
    traj = traj.detach().cpu()
    n_targets = n_targets.detach().cpu()
    _, _, height, width = sequences_db.shape
    target_frames = found_target_frames = 0
    peaks_total = matched_peaks = 0
    contrasts: list[float] = []

    for bi, sequence in enumerate(sequences_db):
        count = int(n_targets[bi])
        for li, frame in enumerate(sequence):
            peaks = detect_peaks(
                frame, threshold_db=threshold_db, max_peaks=max_peaks)
            truth = traj[bi, :count, li]
            peaks_total += len(peaks)
            if count and len(peaks):
                distances = torch.cdist(peaks, truth)
                matched_peaks += int((distances.min(dim=1).values <= tolerance_bins).sum())
                found_target_frames += int(
                    (distances.min(dim=0).values <= tolerance_bins).sum())
            target_frames += count

            frame_median = float(frame.median())
            for target in truth:
                row = min(max(int(torch.round(target[0])), 0), height - 1)
                col = min(max(int(torch.round(target[1])), 0), width - 1)
                r0 = max(0, row - contrast_half_width)
                r1 = min(height, row + contrast_half_width + 1)
                c0 = max(0, col - contrast_half_width)
                c1 = min(width, col + contrast_half_width + 1)
                contrasts.append(float(frame[r0:r1, c0:c1].max()) - frame_median)

    contrast_tensor = torch.tensor(contrasts, dtype=torch.float64)
    if len(contrast_tensor):
        quantiles = torch.quantile(contrast_tensor, torch.tensor(
            [0.1, 0.5, 0.9], dtype=torch.float64))
        contrast_summary = {
            "mean": float(contrast_tensor.mean()),
            "std": float(contrast_tensor.std(unbiased=False)),
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
        }
    else:
        contrast_summary = None
    return {
        "target_frames": target_frames,
        "found_target_frames": found_target_frames,
        "target_frame_recall": (
            found_target_frames / target_frames if target_frames else None),
        "detected_peaks": peaks_total,
        "peaks_matching_target": matched_peaks,
        "peak_precision": matched_peaks / peaks_total if peaks_total else None,
        "target_peak_minus_frame_median_db": contrast_summary,
    }


def load_validation_items(cache_dir: Path, limit: int, offset: int = 0) -> dict:
    """Load a deterministic contiguous slice without retaining whole shards."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    chosen: list[dict] = []
    seen = 0
    for shard_path in sorted(cache_dir.glob("val_*.pt")):
        shard = torch.load(shard_path, map_location="cpu")
        for item in shard:
            if seen >= offset and len(chosen) < limit:
                chosen.append(item)
            seen += 1
            if len(chosen) == limit:
                break
        if len(chosen) == limit:
            break
    if len(chosen) != limit:
        raise ValueError(
            f"requested {limit} validation items at offset {offset}, found {len(chosen)}")
    return {
        "x": torch.stack([item["x"] for item in chosen]).float(),
        "traj": torch.stack([item["traj"] for item in chosen]).float(),
        "n_targets": torch.tensor(
            [int(item["n_targets"]) for item in chosen], dtype=torch.long),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_by_region(
    values: torch.Tensor,
    target_mask: torch.Tensor,
    clutter_mask: torch.Tensor,
    data_std_db: float,
) -> dict:
    acc = _region_accumulators()
    _update_regions(acc, values, values, target_mask, clutter_mask)
    summary = summarize_regions(acc, data_std_db=data_std_db)
    # Self-comparison error fields are uninformative in this reference block.
    for region in summary["regions"].values():
        for key in ("prediction_mean", "prediction_std", "mse", "rmse", "mae", "bias",
                    "pearson_r"):
            region.pop(key, None)
    summary.pop("target_to_clutter_mse_ratio", None)
    contrast = summary.pop("mean_target_minus_clutter_db")
    summary["mean_target_minus_clutter_db"] = contrast["reference"]
    return summary


@torch.inference_mode()
def evaluate_arm(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    x0_cpu: torch.Tensor,
    noise_cpu: torch.Tensor,
    traj_cpu: torch.Tensor,
    n_targets_cpu: torch.Tensor,
    target_mask_cpu: torch.Tensor,
    clutter_mask_cpu: torch.Tensor,
    timesteps: list[int],
    batch_size: int,
    device: torch.device,
    data_mean_db: float,
    data_std_db: float,
    tolerance_bins: float,
    threshold_db: float,
    max_peaks: int,
    contrast_half_width: int,
) -> dict[str, dict]:
    """Evaluate one checkpoint on paired held-out inputs and noise draws."""
    results: dict[str, dict] = {}
    for timestep in timesteps:
        eps_acc = _region_accumulators()
        raw_x0_acc = _region_accumulators()
        clamped_x0_acc = _region_accumulators()
        raw_outputs: list[torch.Tensor] = []
        clamped_outputs: list[torch.Tensor] = []
        clipped_counts = {"all": 0, "target_window": 0, "clutter_only": 0}
        value_counts = {"all": 0, "target_window": 0, "clutter_only": 0}

        for start in range(0, len(x0_cpu), batch_size):
            stop = min(start + batch_size, len(x0_cpu))
            x0 = x0_cpu[start:stop].to(device)
            noise = noise_cpu[start:stop].to(device)
            target_mask = target_mask_cpu[start:stop].to(device)
            clutter_mask = clutter_mask_cpu[start:stop].to(device)
            t = torch.full(
                (len(x0),), timestep, dtype=torch.long, device=device)
            xt = diffusion.q_sample(x0, t, noise)
            eps_hat = model(xt, t, None)
            x0_hat = diffusion.pred_x0(xt, t, eps_hat)
            x0_hat_clamped = x0_hat.clamp(-4.0, 4.0)

            _update_regions(eps_acc, noise, eps_hat, target_mask, clutter_mask)
            _update_regions(raw_x0_acc, x0, x0_hat, target_mask, clutter_mask)
            _update_regions(
                clamped_x0_acc, x0, x0_hat_clamped, target_mask, clutter_mask)
            clipped = x0_hat.abs() > 4.0
            clipped_counts["all"] += int(clipped.sum())
            value_counts["all"] += clipped.numel()
            clipped_counts["target_window"] += int((clipped & target_mask).sum())
            value_counts["target_window"] += int(target_mask.sum())
            clipped_counts["clutter_only"] += int((clipped & clutter_mask).sum())
            value_counts["clutter_only"] += int(clutter_mask.sum())
            raw_outputs.append(x0_hat.cpu())
            clamped_outputs.append(x0_hat_clamped.cpu())

        reconstructed_raw_norm = torch.cat(raw_outputs)
        reconstructed_norm = torch.cat(clamped_outputs)
        reconstructed_raw_db = reconstructed_raw_norm * data_std_db + data_mean_db
        reconstructed_db = reconstructed_norm * data_std_db + data_mean_db
        raw_frame_metrics = target_frame_metrics(
            reconstructed_raw_db,
            traj_cpu,
            n_targets_cpu,
            tolerance_bins=tolerance_bins,
            threshold_db=threshold_db,
            max_peaks=max_peaks,
            contrast_half_width=contrast_half_width,
        )
        frame_metrics = target_frame_metrics(
            reconstructed_db,
            traj_cpu,
            n_targets_cpu,
            tolerance_bins=tolerance_bins,
            threshold_db=threshold_db,
            max_peaks=max_peaks,
            contrast_half_width=contrast_half_width,
        )
        alpha_bar = float(diffusion.alphas_bar[timestep])
        results[str(timestep)] = {
            "alpha_bar": alpha_bar,
            "signal_to_noise_ratio": alpha_bar / max(1.0 - alpha_bar, 1e-30),
            "epsilon_prediction": summarize_regions(eps_acc),
            "x0_reconstruction_raw": summarize_regions(
                raw_x0_acc, data_std_db=data_std_db),
            "x0_reconstruction_sampler_clamped": summarize_regions(
                clamped_x0_acc, data_std_db=data_std_db),
            "fraction_changed_by_sampler_clamp": (
                clipped_counts["all"] / value_counts["all"]),
            "fraction_changed_by_sampler_clamp_by_region": {
                name: clipped_counts[name] / value_counts[name]
                for name in clipped_counts
            },
            "raw_target_frame_metrics": raw_frame_metrics,
            "sampler_clamped_target_frame_metrics": frame_metrics,
        }
    return results


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    temporary.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument(
        "--arm", action="append", metavar="NAME=CHECKPOINT",
        help="checkpoint arm; repeat to override the three defaults")
    parser.add_argument("--timesteps", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-half-width", type=int, default=2)
    parser.add_argument("--clutter-guard-half-width", type=int, default=4)
    parser.add_argument("--detection-tolerance-bins", type=float, default=2.0)
    parser.add_argument("--detection-threshold-db", type=float, default=12.0)
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="samples/diag_denoising_by_timestep.json")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        arms = parse_arms(args.arm)
    except ValueError as exc:
        parser.error(str(exc))
    missing = [str(path) for _, path in arms if not path.is_file()]
    if missing:
        parser.error("missing checkpoint(s): " + ", ".join(missing))

    cache_dir = Path(args.cache)
    stats_path = cache_dir / "stats.pt"
    if not stats_path.is_file():
        parser.error(f"missing normalization statistics: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")
    data_mean_db = float(stats["mean"])
    data_std_db = float(stats["std"])
    selected = load_validation_items(
        cache_dir, args.n_sequences, args.validation_offset)
    raw_x0 = selected["x"]
    x0 = (raw_x0 - data_mean_db) / data_std_db
    traj = selected["traj"]
    n_targets = selected["n_targets"]
    target_mask, clutter_mask = build_region_masks(
        traj,
        n_targets,
        raw_x0.shape[-2],
        raw_x0.shape[-1],
        target_half_width=args.target_half_width,
        clutter_guard_half_width=args.clutter_guard_half_width,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    fixed_noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    device = _resolve_device(args.device)

    report = {
        "status": "running",
        "protocol": {
            "diagnostic": "one_step_denoising_of_held_out_real_sequences",
            "validation_selection": {
                "split": "val",
                "offset": args.validation_offset,
                "count": args.n_sequences,
            },
            "paired_noise": True,
            "noise_seed": args.seed,
            "timesteps_requested": args.timesteps,
            "batch_size": args.batch_size,
            "device": str(device),
            "target_region": {
                "shape": "square trajectory-centered window",
                "half_width_bins": args.target_half_width,
            },
            "clutter_region": {
                "definition": "outside all trajectory-centered guard windows",
                "guard_half_width_bins": args.clutter_guard_half_width,
                "note": "pixels between target and guard windows are excluded",
            },
            "target_frame_detection": {
                "threshold_db_above_frame_median": args.detection_threshold_db,
                "max_peaks_per_frame": args.max_peaks,
                "match_tolerance_bins": args.detection_tolerance_bins,
                "note": "per-frame diagnostic; not filtered-track recall",
            },
            "sampler_x0_clamp": [-4.0, 4.0],
        },
        "dataset": {
            "cache": str(cache_dir),
            "normalization_mean_db": data_mean_db,
            "normalization_std_db": data_std_db,
            "target_pixels": int(target_mask.sum()),
            "clutter_only_pixels": int(clutter_mask.sum()),
            "ignored_guard_pixels": int((~target_mask & ~clutter_mask).sum()),
            "reference_distribution_normalized": _distribution_by_region(
                x0, target_mask, clutter_mask, data_std_db),
            "reference_target_frame_metrics": target_frame_metrics(
                raw_x0,
                traj,
                n_targets,
                tolerance_bins=args.detection_tolerance_bins,
                threshold_db=args.detection_threshold_db,
                max_peaks=args.max_peaks,
                contrast_half_width=args.target_half_width,
            ),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        },
        "arms": {},
    }
    out_path = Path(args.out)
    _write_report(report, out_path)

    for arm_name, checkpoint_path in arms:
        print(f"loading {arm_name}: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        total_timesteps = int(config["diffusion"]["timesteps"])
        try:
            timesteps = parse_timesteps(args.timesteps, total_timesteps)
        except ValueError as exc:
            parser.error(f"{arm_name}: {exc}")
        if int(config["data"]["seq_len"]) != x0.shape[1]:
            raise ValueError(
                f"{arm_name} expects sequence length {config['data']['seq_len']}, "
                f"but selected validation data has {x0.shape[1]}")

        model = build_model(config, device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        checkpoint_meta = {
            "checkpoint": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
            "attention_mode": config["model"].get("attn_mode", "temporal"),
            "depth": config["model"]["depth"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "diffusion_timesteps": total_timesteps,
            "evaluated_timesteps": timesteps,
        }
        del checkpoint
        diffusion = GaussianDiffusion(total_timesteps)
        print(
            f"evaluating {arm_name} at {len(timesteps)} timesteps on {device}",
            flush=True,
        )
        checkpoint_meta["by_timestep"] = evaluate_arm(
            model,
            diffusion,
            x0,
            fixed_noise,
            traj,
            n_targets,
            target_mask,
            clutter_mask,
            timesteps,
            args.batch_size,
            device,
            data_mean_db,
            data_std_db,
            args.detection_tolerance_bins,
            args.detection_threshold_db,
            args.max_peaks,
            args.target_half_width,
        )
        report["arms"][arm_name] = checkpoint_meta
        _write_report(report, out_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["status"] = "complete"
    _write_report(report, out_path)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
