"""Synthesize a wide cirrus field from fine and coarse scattering means."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as torch_f

from .scattering_mean import FullMeanFeature, MeanFeature, ScatteringMean2D


@dataclass(frozen=True)
class ScatteringTargetGroup:
    """A named, equally weighted group of log scattering coefficients."""

    name: str
    features: tuple[MeanFeature | FullMeanFeature, ...]
    target_log: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class PatchCrossScaleConstraint:
    """Robust ensemble target evaluated on equal-footprint output patches."""

    name: str
    calculator: ScatteringMean2D
    features: tuple[MeanFeature | FullMeanFeature, ...]
    target_log: np.ndarray
    uncertainty_log: np.ndarray
    patch_shape: tuple[int, int] = (150, 150)
    batch_size: int = 4
    huber_delta: float = 1.5
    uncertainty_floor_log: float = math.log1p(0.10)
    strategy: str = "patch"
    global_calculator: ScatteringMean2D | None = None


@dataclass(frozen=True)
class ConditionalPatchScatteringConstraint:
    """Patch scattering target predicted from local Planck-radiance mean."""

    name: str
    calculator: ScatteringMean2D
    features: tuple[MeanFeature | FullMeanFeature, ...]
    intercept_log: np.ndarray
    slope_log_radiance: np.ndarray
    uncertainty_log: np.ndarray
    radiance_mean_map: np.ndarray
    reference_log_radiance: float
    patch_shape: tuple[int, int]
    batch_size: int = 4
    weight: float = 1.0
    huber_delta: float = 1.5
    uncertainty_floor_log: float = math.log1p(0.05)


@dataclass(frozen=True)
class ExtraConstraint:
    """Additional differentiable feature-vector constraint."""

    name: str
    measure: Callable[[torch.Tensor], torch.Tensor]
    target: np.ndarray
    weight: float = 1.0
    convergence_rms: float = 0.05
    warmup_steps: int = 0
    loss: str = "mse"
    huber_delta: float = 1.5


@dataclass(frozen=True)
class SynthesisConfig:
    """Adam settings for the first Planck--Herschel synthesis test."""

    steps: int = 300
    learning_rate: float = 0.03
    seed: int = 4676
    seed_fraction: float = 0.01
    seed_highpass_pixels: float = 50.0
    convergence_rms: float = 0.05
    convergence_patience: int = 20
    gradient_clip: float = 10.0
    resolution_fwhm_pixels: float | None = 6.0
    device: str = "gpu"


class MeanPreservingImage(torch.nn.Module):
    """Learn a resolution-limited field while preserving its mean exactly."""

    def __init__(
        self,
        initial: torch.Tensor,
        target_mean: float,
        resolution_fwhm_pixels: float | None = None,
    ):
        super().__init__()
        self.param = torch.nn.Parameter(initial.clone())
        self.target_mean = float(target_mean)
        if resolution_fwhm_pixels is None or resolution_fwhm_pixels <= 0:
            self.register_buffer("kernel", torch.empty(0))
        else:
            sigma = float(resolution_fwhm_pixels) / 2.354820045
            radius = max(int(math.ceil(4.0 * sigma)), 1)
            coordinate = torch.arange(-radius, radius + 1, device=initial.device, dtype=initial.dtype)
            kernel_1d = torch.exp(-0.5 * (coordinate / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            self.register_buffer("kernel", kernel_2d[None, None])

    @property
    def image(self) -> torch.Tensor:
        image = self.param
        if self.kernel.numel():
            radius = self.kernel.shape[-1] // 2
            image = torch_f.conv2d(
                torch_f.pad(image[:, None], (radius, radius, radius, radius), mode="reflect"),
                self.kernel,
            )[:, 0]
        return image - image.mean(dim=(-2, -1), keepdim=True) + self.target_mean


def highpass_random_seed(
    shape: tuple[int, int],
    *,
    cutoff_pixels: float,
    target_std: float,
    seed: int,
) -> np.ndarray:
    """Return reproducible zero-mean Gaussian fluctuations above a soft cutoff."""

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    radius = np.hypot(fy, fx)
    lowpass = np.exp(-0.5 * (radius * float(cutoff_pixels)) ** 2)
    highpass = np.fft.ifft2(np.fft.fft2(noise) * (1.0 - lowpass)).real
    highpass -= highpass.mean()
    highpass *= float(target_std) / max(float(highpass.std()), np.finfo(float).eps)
    return highpass


def synthesize_scattering_means(
    image_init: np.ndarray,
    groups: tuple[ScatteringTargetGroup, ...],
    *,
    calculator: ScatteringMean2D,
    target_mean: float,
    config: SynthesisConfig | None = None,
    feature_mode: str = "isotropic",
    patch_constraint: PatchCrossScaleConstraint | None = None,
    conditional_constraint: ConditionalPatchScatteringConstraint | None = None,
    extra_constraints: tuple[ExtraConstraint, ...] = (),
    callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Optimize selected scattering means with equal weight per target group."""

    cfg = config or SynthesisConfig()
    image_init = np.asarray(image_init, dtype=np.float32)
    seed_image = highpass_random_seed(
        image_init.shape,
        cutoff_pixels=cfg.seed_highpass_pixels,
        target_std=cfg.seed_fraction * float(image_init.std()),
        seed=cfg.seed,
    )
    initial = torch.as_tensor(image_init + seed_image, dtype=torch.float32, device=calculator.device)[None]
    model = MeanPreservingImage(
        initial,
        target_mean,
        resolution_fwhm_pixels=cfg.resolution_fwhm_pixels,
    ).to(calculator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=35, min_lr=cfg.learning_rate / 64.0
    )
    all_features = tuple(feature for group in groups for feature in group.features)
    group_slices: dict[str, slice] = {}
    start = 0
    for group in groups:
        stop = start + len(group.features)
        group_slices[group.name] = slice(start, stop)
        start = stop
    target = torch.as_tensor(
        np.concatenate([np.asarray(group.target_log) for group in groups]),
        dtype=torch.float32,
        device=calculator.device,
    )[None]
    extra_targets = {
        constraint.name: torch.as_tensor(
            constraint.target, dtype=torch.float32, device=calculator.device
        )[None]
        for constraint in extra_constraints
    }

    def measure_mean(
        selected_calculator: ScatteringMean2D,
        image: torch.Tensor,
        features: tuple[MeanFeature | FullMeanFeature, ...],
    ) -> torch.Tensor:
        if feature_mode == "isotropic":
            return selected_calculator.log_features(image, features)  # type: ignore[arg-type]
        if feature_mode == "full":
            return selected_calculator.log_full_features(image, features)  # type: ignore[arg-type]
        raise ValueError(f"Unknown feature mode {feature_mode!r}")

    rng = np.random.default_rng(cfg.seed + 1)

    history: list[dict[str, float]] = []
    converged_steps = 0
    for step in range(cfg.steps):
        optimizer.zero_grad()
        losses = []
        row: dict[str, float] = {"step": float(step)}
        all_converged = True
        measured = measure_mean(calculator, model.image, all_features)
        for group in groups:
            selection = group_slices[group.name]
            delta_log = measured[:, selection] - target[:, selection]
            losses.append(float(group.weight) * torch.mean(delta_log**2))
            relative_rms = torch.sqrt(torch.mean(torch.expm1(delta_log) ** 2))
            value = float(relative_rms.detach().cpu())
            row[f"rms_{group.name}"] = value
            all_converged &= value <= cfg.convergence_rms
        if conditional_constraint is not None:
            ny, nx = model.image.shape[-2:]
            ph, pw = conditional_constraint.patch_shape
            if ph > ny or pw > nx:
                raise ValueError("Conditional patch is larger than the synthesized image.")
            patches, radiance_values = [], []
            for _ in range(conditional_constraint.batch_size):
                y = int(rng.integers(0, ny - ph + 1))
                x = int(rng.integers(0, nx - pw + 1))
                patches.append(model.image[:, y : y + ph, x : x + pw])
                radiance_values.append(
                    conditional_constraint.radiance_mean_map[y + ph // 2, x + pw // 2]
                )
            patch_images = torch.cat(patches, dim=0)
            values = measure_mean(
                conditional_constraint.calculator, patch_images, conditional_constraint.features
            )
            radiance_tensor = torch.as_tensor(
                radiance_values, dtype=values.dtype, device=values.device
            )
            intercept = torch.as_tensor(
                conditional_constraint.intercept_log, dtype=values.dtype, device=values.device
            )
            slope = torch.as_tensor(
                conditional_constraint.slope_log_radiance, dtype=values.dtype, device=values.device
            )
            sigma = torch.as_tensor(
                np.maximum(
                    conditional_constraint.uncertainty_log,
                    conditional_constraint.uncertainty_floor_log,
                ), dtype=values.dtype, device=values.device
            )
            target_conditional = intercept[None] + (
                torch.log(radiance_tensor) - conditional_constraint.reference_log_radiance
            )[:, None] * slope[None]
            standardized = (values - target_conditional) / sigma[None]
            conditional_loss = conditional_constraint.weight * torch_f.huber_loss(
                standardized, torch.zeros_like(standardized), reduction="mean",
                delta=conditional_constraint.huber_delta,
            ) * conditional_constraint.uncertainty_floor_log**2
            losses.append(conditional_loss)
            conditional_center = torch.median(values.detach(), dim=0).values
            target_center = torch.median(target_conditional.detach(), dim=0).values
            conditional_rms = torch.sqrt(
                torch.mean(torch.expm1(conditional_center - target_center) ** 2)
            )
            row[f"rms_{conditional_constraint.name}"] = float(conditional_rms.cpu())
            all_converged &= float(conditional_rms.cpu()) <= cfg.convergence_rms
        if patch_constraint is not None:
            if patch_constraint.strategy == "patch":
                ny, nx = model.image.shape[-2:]
                ph, pw = patch_constraint.patch_shape
                if ph > ny or pw > nx:
                    raise ValueError("Cross-scale patch is larger than the synthesized image.")
                patches = []
                for _ in range(patch_constraint.batch_size):
                    y = int(rng.integers(0, ny - ph + 1))
                    x = int(rng.integers(0, nx - pw + 1))
                    patches.append(model.image[:, y : y + ph, x : x + pw])
                patch_images = torch.cat(patches, dim=0)
                patch_values = measure_mean(
                    patch_constraint.calculator, patch_images, patch_constraint.features
                )
            elif patch_constraint.strategy == "global":
                if patch_constraint.global_calculator is None:
                    raise ValueError("The global fallback requires global_calculator.")
                patch_values = measure_mean(
                    patch_constraint.global_calculator,
                    model.image,
                    patch_constraint.features,
                )
            else:
                raise ValueError(f"Unknown cross-scale strategy {patch_constraint.strategy!r}")
            patch_target = torch.as_tensor(
                patch_constraint.target_log,
                dtype=torch.float32,
                device=calculator.device,
            )[None]
            patch_sigma = torch.as_tensor(
                np.maximum(
                    patch_constraint.uncertainty_log,
                    patch_constraint.uncertainty_floor_log,
                ),
                dtype=torch.float32,
                device=calculator.device,
            )[None]
            standardized = (patch_values - patch_target) / patch_sigma
            cross_loss = torch_f.huber_loss(
                standardized,
                torch.zeros_like(standardized),
                reduction="mean",
                delta=patch_constraint.huber_delta,
            ) * patch_constraint.uncertainty_floor_log**2
            losses.append(cross_loss)
            center = torch.median(patch_values.detach(), dim=0).values
            cross_rms = torch.sqrt(torch.mean(torch.expm1(center - patch_target[0]) ** 2))
            row[f"rms_{patch_constraint.name}"] = float(cross_rms.cpu())
            all_converged &= float(cross_rms.cpu()) <= cfg.convergence_rms
        for constraint in extra_constraints:
            extra_value = constraint.measure(model.image)
            extra_delta = extra_value - extra_targets[constraint.name]
            ramp = min(1.0, (step + 1) / max(constraint.warmup_steps, 1))
            if constraint.loss == "mse":
                raw_extra_loss = torch.mean(extra_delta**2)
            elif constraint.loss == "huber":
                raw_extra_loss = torch_f.huber_loss(
                    extra_delta,
                    torch.zeros_like(extra_delta),
                    reduction="mean",
                    delta=constraint.huber_delta,
                )
            else:
                raise ValueError(f"Unknown extra-constraint loss {constraint.loss!r}")
            extra_loss = constraint.weight * ramp * raw_extra_loss
            losses.append(extra_loss)
            row[f"rms_{constraint.name}"] = float(
                torch.sqrt(torch.mean(extra_delta.detach() ** 2)).cpu()
            )
            all_converged &= row[f"rms_{constraint.name}"] <= constraint.convergence_rms
        loss = torch.stack(losses).mean()
        loss.backward()
        if cfg.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        scheduler.step(float(loss.detach().cpu()))
        row["loss"] = float(loss.detach().cpu())
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        row["mean"] = float(model.image.mean().detach().cpu())
        history.append(row)
        if callback is not None:
            callback(row)

        converged_steps = converged_steps + 1 if all_converged else 0
        if converged_steps >= cfg.convergence_patience:
            break

    # Keep the saved result in double precision so the requested Planck mean is
    # preserved to numerical precision rather than only float32 precision.
    output = model.image[0].detach().cpu().numpy().astype(np.float64)
    output += target_mean - float(output.mean(dtype=np.float64))
    return output, history
