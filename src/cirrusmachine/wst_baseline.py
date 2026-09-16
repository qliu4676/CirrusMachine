"""CPU-first baseline WST synthesis for one latent cirrus realization.

This module implements baseline A from the UWtile_04703 design review:
ensemble-spectrum initialization, a robust smoothed CDF, reduced-angle
scattering means, uncertainty-band losses, and a guarded central crop.  Phase
harmonics and self-modulation intentionally live outside this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import time
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_f

from .power_spectrum import power_spectrum_2d
from .scattering_mean import (
    FullMeanFeature,
    MeanFeature,
    ScatteringMean2D,
    isotropic_feature_set,
    within_band_pairs,
)


def _measure_scattering(calculator: ScatteringMean2D, image: torch.Tensor, features):
    """Dispatch between rotation-averaged and orientation-resolved means."""

    if features and isinstance(features[0], FullMeanFeature):
        return calculator.log_full_features(image, features)
    return calculator.log_features(image, features)
from .wst_preflight import selected_features_from_reliability


@dataclass(frozen=True)
class BaselineSynthesisConfig:
    """Resolved optimizer choices for the first independent realization."""

    output_pixels: int = 480
    guard_pixels: int = 103
    pixel_scale_arcsec: float = 7.5
    J: int = 8
    L: int = 8
    steps: int = 300
    s2_start_step: int = 60
    phase_start_step: int = 0
    include_validation_j7: bool = False
    include_unreliable_j1: bool = False
    learning_rate: float = 0.03
    gradient_clip: float = 10.0
    scheduler_patience: int = 30
    early_stop_patience: int = 20
    seed: int = 4703
    torch_threads: int = 4
    cdf_sigma_pixels: float = 4.0
    power_bins: int = 28
    reliable_frequency_max_arcmin_inv: float = 1.0
    downweighted_frequency_max_arcmin_inv: float = 2.0
    downweight_sigma_factor: float = 2.0
    uncertainty_floor_log: float = math.log1p(0.05)

    @property
    def canvas_pixels(self) -> int:
        return self.output_pixels + 2 * self.guard_pixels

    @property
    def pixel_scale_arcmin(self) -> float:
        return self.pixel_scale_arcsec / 60.0


class CentralMeanPreservingImage(torch.nn.Module):
    """Learn a guarded field while fixing the delivered crop's mean exactly."""

    def __init__(
        self,
        initial: torch.Tensor,
        target_mean: float,
        guard_pixels: int,
    ) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(initial.clone())
        self.target_mean = float(target_mean)
        self.guard_pixels = int(guard_pixels)

    @property
    def image(self) -> torch.Tensor:
        guard = self.guard_pixels
        central = self.param[
            ..., guard : self.param.shape[-2] - guard,
            guard : self.param.shape[-1] - guard,
        ]
        return self.param - central.mean(dim=(-2, -1), keepdim=True) + self.target_mean


def uncertainty_band_loss(
    measured: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    *,
    band_sigma: float = 1.0,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth loss only for residuals outside an uncertainty band."""

    standardized = (measured - target) / sigma
    violation = torch.relu(torch.abs(standardized) - float(band_sigma))
    loss = torch_f.huber_loss(
        violation,
        torch.zeros_like(violation),
        reduction="mean",
        delta=float(huber_delta),
    )
    return loss, standardized


class PeriodicGaussianSmoother:
    """Differentiable Gaussian smoothing on a stationary periodic canvas."""

    def __init__(
        self,
        shape: tuple[int, int],
        sigma_pixels: float,
        *,
        device: torch.device,
    ) -> None:
        fy = torch.fft.fftfreq(shape[0], device=device)[:, None]
        fx = torch.fft.rfftfreq(shape[1], device=device)[None, :]
        transfer = torch.exp(
            -2.0 * math.pi**2 * float(sigma_pixels) ** 2 * (fy**2 + fx**2)
        )
        self.transfer = transfer.to(torch.float32)
        self.shape = shape

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft2(image, dim=(-2, -1))
        return torch.fft.irfft2(
            transformed * self.transfer[None],
            s=self.shape,
            dim=(-2, -1),
        )


class RadialPowerDescriptor:
    """Differentiable log radial power in fixed physical-frequency bins."""

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_scale_arcmin: float,
        edges: np.ndarray,
        *,
        device: torch.device,
    ) -> None:
        fy = np.fft.fftfreq(shape[0], d=pixel_scale_arcmin)[:, None]
        fx = np.fft.rfftfreq(shape[1], d=pixel_scale_arcmin)[None, :]
        radius = np.hypot(fy, fx)
        flat_radius = radius.ravel()
        bin_index = np.digitize(flat_radius, edges) - 1
        valid = (
            (bin_index >= 0)
            & (bin_index < len(edges) - 1)
            & (flat_radius > 0)
        )
        self.valid_index = torch.as_tensor(
            np.flatnonzero(valid), dtype=torch.long, device=device
        )
        self.bin_index = torch.as_tensor(
            bin_index[valid], dtype=torch.long, device=device
        )
        self.bin_count = torch.bincount(
            self.bin_index, minlength=len(edges) - 1
        ).to(torch.float32)
        if torch.any(self.bin_count == 0):
            raise ValueError("Power-spectrum bins must all contain Fourier modes")
        self.shape = shape
        self.centers = np.sqrt(edges[:-1] * edges[1:])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        centered = image - image.mean(dim=(-2, -1), keepdim=True)
        transformed = torch.fft.rfft2(centered, dim=(-2, -1))
        power = transformed.abs().square() / float(np.prod(self.shape))
        selected = power.reshape(power.shape[0], -1)[:, self.valid_index]
        rows = []
        for sample in selected:
            total = torch.zeros(
                len(self.centers), dtype=sample.dtype, device=sample.device
            )
            total.scatter_add_(0, self.bin_index, sample)
            rows.append(total / self.bin_count)
        mean_power = torch.stack(rows)
        return torch.log(mean_power.clamp_min(torch.finfo(mean_power.dtype).tiny))


def latent_power_curve(
    frequency: np.ndarray,
    fitted_power: np.ndarray,
    evaluation_frequency: np.ndarray,
    *,
    large_scale_slope: float,
    reliable_frequency_max: float,
) -> np.ndarray:
    """Evaluate the hybrid model with a physical high-k slope extrapolation."""

    frequency = np.asarray(frequency, dtype=float)
    fitted_power = np.asarray(fitted_power, dtype=float)
    evaluation = np.asarray(evaluation_frequency, dtype=float)
    zero = evaluation <= 0.0
    safe = np.maximum(evaluation, np.finfo(float).tiny)
    log_model = np.interp(
        np.log(safe),
        np.log(frequency),
        np.log(fitted_power),
    )
    below = safe < frequency.min()
    if np.any(below):
        log_model[below] = (
            np.log(fitted_power[0])
            + large_scale_slope * np.log(safe[below] / frequency[0])
        )
    pivot_power = float(
        np.exp(
            np.interp(
                np.log(reliable_frequency_max),
                np.log(frequency),
                np.log(fitted_power),
            )
        )
    )
    above = safe > reliable_frequency_max
    if np.any(above):
        log_model[above] = (
            np.log(pivot_power)
            + large_scale_slope
            * np.log(safe[above] / reliable_frequency_max)
        )
    log_model[zero] = -np.inf
    return np.exp(log_model)


def sample_ensemble_power_field(
    shape: tuple[int, int],
    *,
    pixel_scale_arcmin: float,
    frequency: np.ndarray,
    fitted_power: np.ndarray,
    large_scale_slope: float,
    reliable_frequency_max: float,
    target_mean: float,
    seed: int,
) -> np.ndarray:
    """Draw independent Fourier amplitudes/phases from the latent power model."""

    rng = np.random.default_rng(seed)
    white = rng.normal(size=shape)
    transformed = np.fft.rfft2(white)
    fy = np.fft.fftfreq(shape[0], d=pixel_scale_arcmin)[:, None]
    fx = np.fft.rfftfreq(shape[1], d=pixel_scale_arcmin)[None, :]
    radius = np.hypot(fy, fx)
    model = latent_power_curve(
        frequency,
        fitted_power,
        radius,
        large_scale_slope=large_scale_slope,
        reliable_frequency_max=reliable_frequency_max,
    )
    model[0, 0] = 0.0
    sampled = np.fft.irfft2(transformed * np.sqrt(model), s=shape)
    sampled -= sampled.mean()
    sampled += float(target_mean)
    return sampled.astype(np.float32)


def build_power_targets(
    shape: tuple[int, int],
    config: BaselineSynthesisConfig,
    power: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log-bin target and uncertainty for the guarded canvas."""

    minimum_frequency = 1.0 / (
        config.output_pixels * config.pixel_scale_arcmin
    )
    descriptor = None
    edges = None
    for bin_count in range(config.power_bins, 7, -1):
        candidate = np.geomspace(
            minimum_frequency,
            config.downweighted_frequency_max_arcmin_inv,
            bin_count + 1,
        )
        try:
            descriptor = RadialPowerDescriptor(
                shape,
                config.pixel_scale_arcmin,
                candidate,
                device=torch.device("cpu"),
            )
            edges = candidate
            break
        except ValueError:
            continue
    if descriptor is None or edges is None:
        raise ValueError("Could not construct populated radial power bins")
    target = latent_power_curve(
        power["frequency"],
        power["fitted_power"],
        descriptor.centers,
        large_scale_slope=float(power["large_scale_slope"]),
        reliable_frequency_max=config.reliable_frequency_max_arcmin_inv,
    )
    patch_scatter = np.interp(
        descriptor.centers,
        power["patch_frequency"],
        power["patch_log_scatter"],
        left=power["patch_log_scatter"][0],
        right=power["patch_log_scatter"][-1],
    )
    sample_uncertainty = 1.0 / np.sqrt(
        descriptor.bin_count.detach().cpu().numpy()
    )
    uncertainty = np.maximum(patch_scatter, sample_uncertainty)
    uncertainty = np.maximum(uncertainty, config.uncertainty_floor_log)
    uncertainty = np.where(
        descriptor.centers > config.reliable_frequency_max_arcmin_inv,
        uncertainty * config.downweight_sigma_factor,
        uncertainty,
    )
    return edges, np.log(target), uncertainty


def build_wst_targets(
    reliability: Mapping[str, np.ndarray],
    wst: Mapping[str, np.ndarray],
    config: BaselineSynthesisConfig,
    *,
    validation_image: np.ndarray | None = None,
    validation_mask_fraction: np.ndarray | None = None,
) -> tuple[tuple[MeanFeature, ...], np.ndarray, np.ndarray]:
    """Select canonical features and propagate measurement/systematic errors."""

    reliability = {
        key: np.asarray(value).copy() for key, value in reliability.items()
    }
    if config.include_validation_j7:
        if validation_image is None or validation_mask_fraction is None:
            raise ValueError(
                "The image and mask-fraction map are required when including j=7"
            )
        status = reliability["status"].astype(object)
        status[np.asarray(reliability["j"]) == 7] = "downweight"
        reliability["status"] = status
    if config.include_unreliable_j1:
        status = reliability["status"].astype(object)
        status[np.asarray(reliability["j"]) == 1] = "downweight"
        reliability["status"] = status
    features = selected_features_from_reliability(
        dict(reliability), config.L, dict(wst)
    )
    lookup = {
        (
            int(order),
            int(j1),
            int(j2),
            int(dl),
        ): index
        for index, (order, j1, j2, dl) in enumerate(
            zip(
                wst["feature_order"],
                wst["feature_j1"],
                wst["feature_j2"],
                wst["feature_delta_l"],
                strict=True,
            )
        )
    }
    scale_status = {
        int(j): str(status)
        for j, status in zip(
            reliability["j"], reliability["status"], strict=True
        )
    }
    patch_J = int(
        max(
            np.where(np.asarray(wst["patch_supported"], dtype=bool), wst["feature_j1"], -1)
        )
        + 1
    )
    patch_features = isotropic_feature_set(
        range(patch_J), within_band_pairs(range(patch_J)), config.L
    )
    patch_lookup = {feature: index for index, feature in enumerate(patch_features)}
    patch_values = np.asarray(wst["patch_values"], dtype=float)
    validation_lookup: dict[MeanFeature, int] = {}
    validation_values = None
    validation_weights = None
    if config.include_validation_j7:
        validation_features = tuple(
            feature
            for feature in features
            if 7 in (
                (feature.j1,)
                if feature.order == 1
                else (feature.j1, feature.j2)
            )
        )
        patch_pixels = 360
        stride = validation_image.shape[0] - patch_pixels
        coords = ((0, 0), (0, stride), (stride, 0), (stride, stride))
        patches = np.stack(
            [
                validation_image[
                    y : y + patch_pixels, x : x + patch_pixels
                ]
                for y, x in coords
            ]
        )
        calculator = ScatteringMean2D(
            (patch_pixels, patch_pixels),
            config.J,
            config.L,
            mask_boundary=True,
            boundary_sigma=0.5,
            device="cpu",
        )
        with torch.no_grad():
            validation_values = (
                calculator.log_features(patches, validation_features)
                .cpu()
                .numpy()
            )
        reliability_map = np.clip(
            1.0 - np.asarray(validation_mask_fraction), 0.0, 1.0
        ) ** 2
        validation_weights = np.asarray(
            [
                reliability_map[
                    y : y + patch_pixels, x : x + patch_pixels
                ].mean()
                for y, x in coords
            ]
        )
        validation_weights /= validation_weights.sum()
        validation_lookup = {
            feature: index
            for index, feature in enumerate(validation_features)
        }

    target = []
    uncertainty = []
    for feature in features:
        index = lookup[
            (feature.order, feature.j1, feature.j2, feature.delta_l)
        ]
        if feature in validation_lookup:
            samples = validation_values[:, validation_lookup[feature]]
            patch_mad = 1.4826022185 * float(
                np.median(np.abs(samples - np.median(samples)))
            )
            # The four 360-pixel windows overlap and contain few independent
            # j=7 structures, so retain the full patch dispersion rather than
            # dividing it by sqrt(N).
            center_uncertainty = patch_mad
            mask_systematic = abs(
                float(np.sum(samples * validation_weights))
                - float(np.mean(samples))
            )
        else:
            samples = patch_values[:, patch_lookup[feature]]
            patch_mad = 1.4826022185 * float(
                np.median(np.abs(samples - np.median(samples)))
            )
            center_uncertainty = patch_mad / math.sqrt(4.0)
            mask_systematic = (
                float(wst["mask_shift_sigma"][index]) * patch_mad
            )
        boundary_systematic = abs(
            float(wst["full_plain"][index]) - float(wst["full_interior"][index])
        )
        sigma = max(
            center_uncertainty,
            boundary_systematic,
            mask_systematic,
            config.uncertainty_floor_log,
        )
        involved = (
            (feature.j1,)
            if feature.order == 1
            else (feature.j1, feature.j2)
        )
        if any(scale_status[scale] == "downweight" for scale in involved):
            sigma *= config.downweight_sigma_factor
        if config.include_unreliable_j1 and 1 in involved:
            # j=1 is below the empirical four-pixel reliability floor. The
            # ordinary down-weighting above supplies factor two; apply another
            # factor two so this sensitivity ablation uses fourfold inflation.
            sigma *= config.downweight_sigma_factor
        target.append(float(wst["target"][index]))
        uncertainty.append(sigma)
    return features, np.asarray(target), np.asarray(uncertainty)


def block_bootstrap_quantile_uncertainty(
    image: np.ndarray,
    levels: np.ndarray,
    *,
    block_pixels: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Estimate global-CDF uncertainty by resampling correlation-scale blocks."""

    image = np.asarray(image, dtype=float)
    levels = np.asarray(levels, dtype=float)
    ny = image.shape[0] // block_pixels
    nx = image.shape[1] // block_pixels
    if min(ny, nx) < 2:
        raise ValueError("At least two blocks per axis are required")
    trimmed = image[: ny * block_pixels, : nx * block_pixels]
    blocks = np.stack(
        [
            trimmed[
                y * block_pixels : (y + 1) * block_pixels,
                x * block_pixels : (x + 1) * block_pixels,
            ].ravel()
            for y in range(ny)
            for x in range(nx)
        ]
    )
    rng = np.random.default_rng(seed)
    quantiles = []
    for _ in range(draws):
        selection = rng.integers(0, len(blocks), len(blocks))
        quantiles.append(np.quantile(blocks[selection].ravel(), levels))
    return np.std(np.asarray(quantiles), axis=0, ddof=1)


def _family_metrics(standardized: torch.Tensor) -> dict[str, float]:
    detached = standardized.detach()
    violation = torch.relu(torch.abs(detached) - 1.0)
    return {
        "median_abs_z": float(torch.median(torch.abs(detached)).cpu()),
        "rms_z": float(torch.sqrt(torch.mean(detached**2)).cpu()),
        "max_abs_z": float(torch.max(torch.abs(detached)).cpu()),
        "violation_rms": float(torch.sqrt(torch.mean(violation**2)).cpu()),
        "within_fraction": float(torch.mean((torch.abs(detached) <= 1.0).float()).cpu()),
        "within_fraction_gate": float(
            torch.mean((torch.abs(detached) <= 1.001).float()).cpu()
        ),
    }


def synthesize_baseline(
    initial: np.ndarray,
    *,
    target_mean: float,
    features: tuple[MeanFeature, ...],
    wst_target: np.ndarray,
    wst_uncertainty: np.ndarray,
    power_edges: np.ndarray,
    power_target_log: np.ndarray,
    power_uncertainty_log: np.ndarray,
    cdf_levels: np.ndarray,
    cdf_target: np.ndarray,
    cdf_uncertainty: np.ndarray,
    config: BaselineSynthesisConfig,
    phase_measure: Callable[[torch.Tensor], torch.Tensor] | None = None,
    phase_target: np.ndarray | None = None,
    phase_uncertainty: np.ndarray | None = None,
    auxiliary_constraints: Mapping[
        str, tuple[Callable[[torch.Tensor], torch.Tensor], np.ndarray, np.ndarray]
    ]
    | None = None,
    auxiliary_weights: Mapping[str, float] | None = None,
    early_scattering: tuple[
        int,
        int,
        tuple[MeanFeature, ...],
        np.ndarray,
        np.ndarray,
        Callable[[torch.Tensor], torch.Tensor] | None,
        np.ndarray | None,
        np.ndarray | None,
    ]
    | None = None,
    cheap_steps: frozenset[int] = frozenset(),
    best_selection_start_step: int = 0,
    corrected_early_stop_start_step: int | None = None,
    corrected_early_stop_interval: int = 5,
    corrected_early_stop_patience: int = 2,
    callback: Callable[[int, np.ndarray, list[dict[str, float]]], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    """Run baseline A and return the best guarded canvas."""

    torch.set_num_threads(config.torch_threads)
    device = torch.device("cpu")
    shape = tuple(map(int, initial.shape))
    if shape != (config.canvas_pixels, config.canvas_pixels):
        raise ValueError(
            f"Expected guarded canvas {(config.canvas_pixels,) * 2}, got {shape}"
        )
    calculator = ScatteringMean2D(
        shape, config.J, config.L, device="cpu"
    )
    power_descriptor = RadialPowerDescriptor(
        shape,
        config.pixel_scale_arcmin,
        power_edges,
        device=device,
    )
    smoother = PeriodicGaussianSmoother(
        shape, config.cdf_sigma_pixels, device=device
    )
    model = CentralMeanPreservingImage(
        torch.as_tensor(initial, dtype=torch.float32, device=device)[None],
        target_mean,
        config.guard_pixels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=config.learning_rate / 64.0,
    )

    target_wst = torch.as_tensor(
        wst_target, dtype=torch.float32, device=device
    )[None]
    sigma_wst = torch.as_tensor(
        wst_uncertainty, dtype=torch.float32, device=device
    )[None]
    target_power = torch.as_tensor(
        power_target_log, dtype=torch.float32, device=device
    )[None]
    sigma_power = torch.as_tensor(
        power_uncertainty_log, dtype=torch.float32, device=device
    )[None]
    levels = torch.as_tensor(cdf_levels, dtype=torch.float32, device=device)
    target_cdf_tensor = torch.as_tensor(
        cdf_target, dtype=torch.float32, device=device
    )[None]
    sigma_cdf = torch.as_tensor(
        cdf_uncertainty, dtype=torch.float32, device=device
    )[None]
    if phase_measure is not None:
        if phase_target is None or phase_uncertainty is None:
            raise ValueError("Phase target and uncertainty are required together")
        target_phase = torch.as_tensor(
            phase_target, dtype=torch.float32, device=device
        )[None]
        sigma_phase = torch.as_tensor(
            phase_uncertainty, dtype=torch.float32, device=device
        )[None]
    else:
        target_phase = None
        sigma_phase = None
    prepared_auxiliary: dict[
        str, tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]
    ] = {}
    for name, (measure, target, uncertainty) in (auxiliary_constraints or {}).items():
        if not name or any(character.isspace() for character in name):
            raise ValueError("Auxiliary constraint names must be nonempty identifiers")
        target_tensor = torch.as_tensor(
            target, dtype=torch.float32, device=device
        ).reshape(1, -1)
        sigma_tensor = torch.as_tensor(
            uncertainty, dtype=torch.float32, device=device
        ).reshape(1, -1)
        if target_tensor.shape != sigma_tensor.shape:
            raise ValueError(f"Auxiliary target/sigma mismatch for {name}")
        if torch.any(~torch.isfinite(target_tensor)) or torch.any(
            ~torch.isfinite(sigma_tensor) | (sigma_tensor <= 0.0)
        ):
            raise ValueError(f"Auxiliary target/sigma is invalid for {name}")
        prepared_auxiliary[name] = (measure, target_tensor, sigma_tensor)
    unknown_weights = set(auxiliary_weights or {}) - set(prepared_auxiliary)
    if unknown_weights:
        raise ValueError(
            "Auxiliary weights have no matching constraint: "
            + ", ".join(sorted(unknown_weights))
        )
    prepared_auxiliary_weights = {
        name: float((auxiliary_weights or {}).get(name, 1.0))
        for name in prepared_auxiliary
    }
    if any(
        not math.isfinite(weight) or weight <= 0.0
        for weight in prepared_auxiliary_weights.values()
    ):
        raise ValueError("Auxiliary weights must be finite and positive")
    first_order = torch.as_tensor(
        [feature.order == 1 for feature in features],
        dtype=torch.bool,
        device=device,
    )
    second_order = ~first_order
    prepared_early = None
    if early_scattering is not None:
        (
            early_end_step,
            early_L,
            early_features,
            early_target_array,
            early_sigma_array,
            early_phase_measure,
            early_phase_target_array,
            early_phase_sigma_array,
        ) = early_scattering
        early_calculator = ScatteringMean2D(shape, config.J, early_L, device="cpu")
        early_target_tensor = torch.as_tensor(
            early_target_array, dtype=torch.float32, device=device
        )[None]
        early_sigma_tensor = torch.as_tensor(
            early_sigma_array, dtype=torch.float32, device=device
        )[None]
        early_first_order = torch.as_tensor(
            [feature.order == 1 for feature in early_features],
            dtype=torch.bool,
            device=device,
        )
        if early_phase_measure is not None:
            if early_phase_target_array is None or early_phase_sigma_array is None:
                raise ValueError("Early phase target and uncertainty are required")
            early_phase_target_tensor = torch.as_tensor(
                early_phase_target_array, dtype=torch.float32, device=device
            )[None]
            early_phase_sigma_tensor = torch.as_tensor(
                early_phase_sigma_array, dtype=torch.float32, device=device
            )[None]
        else:
            early_phase_target_tensor = None
            early_phase_sigma_tensor = None
        prepared_early = (
            int(early_end_step),
            early_calculator,
            early_features,
            early_target_tensor,
            early_sigma_tensor,
            early_first_order,
            early_phase_measure,
            early_phase_target_tensor,
            early_phase_sigma_tensor,
        )

    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_step = -1
    best_param = model.param.detach().clone()
    converged = 0
    s1_z = s2_z = phase_z = None
    started = time.perf_counter()
    for step in range(config.steps):
        if prepared_early is not None and step >= prepared_early[0]:
            # The L=4 filter banks are no longer needed once the L=8 stage
            # begins.  They are large at 1024^2 and retaining both angular
            # resolutions can exceed a CPU-only memory budget.  Clear the
            # shared filter storage in-place because the caller still holds
            # the early descriptor for provenance during this synchronous run.
            early_calculator = prepared_early[1]
            early_calculator.filters = ()
            early_phase_measure = prepared_early[6]
            early_descriptor = getattr(early_phase_measure, "__self__", None)
            if early_descriptor is not None:
                early_descriptor.calculator.filters = ()
            prepared_early = None
            gc.collect()
            print(
                "RELEASED L=4 scattering filters before the L=8 refinement stage.",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        image = model.image
        cheap_step = step in cheap_steps
        early_step = prepared_early is not None and step < prepared_early[0]
        if not cheap_step:
            if early_step:
                (
                    _,
                    selected_calculator,
                    selected_features,
                    selected_target_wst,
                    selected_sigma_wst,
                    selected_first_order,
                    selected_phase_measure,
                    selected_target_phase,
                    selected_sigma_phase,
                ) = prepared_early
            else:
                selected_calculator = calculator
                selected_features = features
                selected_target_wst = target_wst
                selected_sigma_wst = sigma_wst
                selected_first_order = first_order
                selected_phase_measure = phase_measure
                selected_target_phase = target_phase
                selected_sigma_phase = sigma_phase
            measured_wst = _measure_scattering(
                selected_calculator, image, selected_features
            )
            s1_loss, s1_z = uncertainty_band_loss(
                measured_wst[:, selected_first_order],
                selected_target_wst[:, selected_first_order],
                selected_sigma_wst[:, selected_first_order],
            )
            s2_loss, s2_z = uncertainty_band_loss(
                measured_wst[:, ~selected_first_order],
                selected_target_wst[:, ~selected_first_order],
                selected_sigma_wst[:, ~selected_first_order],
            )
        elif s1_z is None or s2_z is None:
            raise ValueError("The schedule cannot begin with a cheap step")
        measured_power = power_descriptor(image)
        power_loss, power_z = uncertainty_band_loss(
            measured_power, target_power, sigma_power
        )
        smoothed = smoother(image)
        guard = config.guard_pixels
        smoothed = smoothed[
            ...,
            guard : smoothed.shape[-2] - guard,
            guard : smoothed.shape[-1] - guard,
        ]
        measured_cdf = torch.quantile(
            smoothed.reshape(smoothed.shape[0], -1), levels, dim=1
        ).T
        cdf_loss, cdf_z = uncertainty_band_loss(
            measured_cdf, target_cdf_tensor, sigma_cdf
        )
        phase_loss = None
        if not cheap_step and selected_phase_measure is not None:
            measured_phase = selected_phase_measure(image)
            phase_loss, phase_z = uncertainty_band_loss(
                measured_phase, selected_target_phase, selected_sigma_phase
            )
        auxiliary_results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, (measure, target_tensor, sigma_tensor) in prepared_auxiliary.items():
            measured = measure(image).reshape(1, -1)
            if measured.shape != target_tensor.shape:
                raise ValueError(
                    f"Auxiliary measure {name} returned {tuple(measured.shape)}, "
                    f"expected {tuple(target_tensor.shape)}"
                )
            auxiliary_results[name] = uncertainty_band_loss(
                measured, target_tensor, sigma_tensor
            )

        active = [(power_loss, 1.0), (cdf_loss, 1.0)]
        stage = "energy"
        if not cheap_step:
            active.append((s1_loss, 1.0))
            active.append((s2_loss, 1.0))
            stage = "morphology"
        if phase_loss is not None:
            active.append((phase_loss, 1.0))
            stage = "phase"
        if cheap_step:
            stage = "cheap"
        active.extend(
            (
                result[0],
                prepared_auxiliary_weights[name],
            )
            for name, result in auxiliary_results.items()
        )
        if auxiliary_results:
            stage += "+conditioned"
        loss = sum(term * weight for term, weight in active) / sum(
            weight for _, weight in active
        )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            ).detach().cpu()
        )
        optimizer.step()
        scheduler.step(float(loss.detach().cpu()))

        guard = config.guard_pixels
        delivered = model.image[
            ...,
            guard : model.image.shape[-2] - guard,
            guard : model.image.shape[-1] - guard,
        ]
        row: dict[str, float] = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": gradient_norm,
            "mean": float(delivered.mean().detach().cpu()),
            "canvas_mean": float(model.image.mean().detach().cpu()),
            "minimum": float(delivered.min().detach().cpu()),
            "negative_pixels": float(
                torch.count_nonzero(delivered.detach() < 0).cpu()
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "stage_s2_active": float(step >= config.s2_start_step),
            "stage_cheap": float(cheap_step),
            "stage_early_L": float(early_step),
        }
        for family, metrics in (
            ("power", _family_metrics(power_z)),
            ("cdf", _family_metrics(cdf_z)),
            ("s1", _family_metrics(s1_z)),
            ("s2", _family_metrics(s2_z)),
        ):
            row.update({f"{family}_{key}": value for key, value in metrics.items()})
        if phase_z is not None:
            row.update(
                {
                    f"phase_{key}": value
                    for key, value in _family_metrics(phase_z).items()
                }
            )
        for name, (_, standardized) in auxiliary_results.items():
            row.update(
                {
                    f"{name}_{key}": value
                    for key, value in _family_metrics(standardized).items()
                }
            )
        history.append(row)

        comparable_loss = (
            row["loss"]
            if step >= best_selection_start_step and not cheap_step
            else math.inf
        )
        if comparable_loss < best_loss:
            best_loss = comparable_loss
            best_step = step
            best_param = model.param.detach().clone()
        active_metrics = ["power", "cdf", "s1", "s2"]
        if phase_z is not None:
            active_metrics.append("phase")
        active_metrics.extend(auxiliary_results)
        if corrected_early_stop_start_step is None:
            all_inside = (
                step >= config.s2_start_step
                and all(
                    row[f"{name}_within_fraction_gate"] >= 0.90
                    for name in active_metrics
                )
                and all(
                    row[f"{name}_violation_rms"] <= 0.15
                    for name in active_metrics
                )
            )
            converged = converged + 1 if all_inside else 0
            stop_now = converged >= config.early_stop_patience
        else:
            evaluate_stop = (
                not cheap_step
                and step >= corrected_early_stop_start_step
                and (step + 1) % corrected_early_stop_interval == 0
            )
            if evaluate_stop:
                all_inside = all(
                    row[f"{name}_violation_rms"] <= 0.05
                    and row[f"{name}_max_abs_z"] <= 1.25
                    for name in active_metrics
                    if name != "planck"
                ) and row["planck_rms_z"] <= 1.0
                converged = converged + 1 if all_inside else 0
            stop_now = converged >= corrected_early_stop_patience

        if callback is not None and (
            step == 0
            or (step + 1) % 25 == 0
            or step + 1 == config.steps
        ):
            callback(
                step,
                model.image[0].detach().cpu().numpy().astype(np.float32),
                history,
            )
        if stop_now:
            break

    with torch.no_grad():
        model.param.copy_(best_param)
        output = model.image[0].cpu().numpy().astype(np.float64)
    guard = config.guard_pixels
    central_mean = float(
        output[
            guard : output.shape[0] - guard,
            guard : output.shape[1] - guard,
        ].mean()
    )
    output += target_mean - central_mean
    summary = {
        "best_step": float(best_step),
        "best_loss": float(best_loss),
        "elapsed_seconds": time.perf_counter() - started,
        "steps_completed": float(len(history)),
    }
    return output, history, summary


def central_crop(image: np.ndarray, output_pixels: int) -> np.ndarray:
    """Return the central square deliverable from a guarded canvas."""

    image = np.asarray(image)
    y = (image.shape[0] - output_pixels) // 2
    x = (image.shape[1] - output_pixels) // 2
    return image[y : y + output_pixels, x : x + output_pixels]


def artifact_metrics(image: np.ndarray, pixel_scale_arcmin: float) -> dict[str, float]:
    """Return artifact gates that are diagnostics, never optimization losses."""

    image = np.asarray(image, dtype=float)
    median = float(np.median(image))
    robust_sigma = 1.4826022185 * float(np.median(np.abs(image - median)))
    robust_sigma = max(robust_sigma, np.finfo(float).eps)
    gradient_y = np.diff(image, axis=0)
    gradient_x = np.diff(image, axis=1)
    band = max(4, min(image.shape) // 40)
    border_values = np.concatenate(
        (
            gradient_y[:band].ravel(),
            gradient_y[-band:].ravel(),
            gradient_x[:, :band].ravel(),
            gradient_x[:, -band:].ravel(),
        )
    )
    interior_values = np.concatenate(
        (
            gradient_y[band:-band].ravel(),
            gradient_x[:, band:-band].ravel(),
        )
    )
    border_ratio = np.sqrt(np.mean(border_values**2)) / np.sqrt(
        np.mean(interior_values**2)
    )

    power = power_spectrum_2d(image)
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0], d=pixel_scale_arcmin))
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1], d=pixel_scale_arcmin))
    radius = np.hypot(fy[:, None], fx[None, :])
    theta = np.arctan2(fy[:, None], fx[None, :])
    use = (radius >= 1.0 / 30.0) & (radius <= 2.0)
    axial = use & (
        np.abs(np.sin(2.0 * theta)) < np.sin(np.deg2rad(10.0))
    )
    diagonal = use & (
        np.abs(np.cos(2.0 * theta)) < np.sin(np.deg2rad(10.0))
    )
    axial_ratio = float(np.median(power[axial]) / np.median(power[diagonal]))
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "mad_std": robust_sigma,
        "minimum": float(np.min(image)),
        "maximum": float(np.max(image)),
        "negative_pixels": int(np.count_nonzero(image < 0)),
        "robust_extreme_max_abs_z": float(
            np.max(np.abs(image - median)) / robust_sigma
        ),
        "fraction_beyond_6_mad": float(
            np.mean(np.abs(image - median) > 6.0 * robust_sigma)
        ),
        "border_gradient_ratio": float(border_ratio),
        "axis_to_diagonal_power": axial_ratio,
    }
