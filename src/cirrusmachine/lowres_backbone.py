"""Reusable absolute calibration and smooth low-resolution backbone controls.

This module deliberately stops at the backbone.  It does not compose a neural
residual, generate texture, or call the WST synthesis machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as torch_f

from .neural_field.model import MultiResolutionLatentField, coordinate_grid


@dataclass(frozen=True)
class AbsoluteCalibration:
    """Robust affine conversion from a low-resolution tracer to optical units."""

    slope: float
    intercept: float
    slope_uncertainty: float
    intercept_uncertainty: float
    correlation: float
    residual_mad: float
    blocks_used: int
    block_pixels: int
    mask_sensitivity_slope_sigma: float
    mask_sensitivity_intercept_sigma: float
    uncertainty_inflation: float
    statements: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BackboneFit:
    """Metrics for one restricted smooth-backbone parameterization."""

    method: str
    parameter_count: int
    runtime_seconds: float
    steps: int
    train_rmse: float
    holdout_rmse: float
    beam_rmse: float
    beam_correlation: float
    edge_gradient_ratio: float
    axial_gradient_moment: float
    representation_fwhm_pixels: float
    high_frequency_fraction: float
    statements: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def normalized_gaussian(
    image: np.ndarray, sigma_pixels: float, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smooth finite pixels without leaking NaNs into their neighbors."""

    values = np.asarray(image, dtype=float)
    finite = np.isfinite(values)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    numerator = gaussian_filter(
        np.where(finite, values, 0.0), float(sigma_pixels), mode="reflect"
    )
    support = gaussian_filter(finite.astype(float), float(sigma_pixels), mode="reflect")
    result = np.divide(
        numerator,
        support,
        out=np.full(values.shape, np.nan, dtype=float),
        where=support > 1.0e-6,
    )
    return result, support


def block_means(
    x: np.ndarray,
    y: np.ndarray,
    block_pixels: int,
    mask_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independent-ish block means and their source-mask fractions."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be matching 2-D arrays")
    block = int(block_pixels)
    if block < 1:
        raise ValueError("block_pixels must be positive")
    mask = (
        np.zeros_like(x, dtype=float)
        if mask_fraction is None
        else np.asarray(mask_fraction, dtype=float)
    )
    if mask.shape != x.shape:
        raise ValueError("mask_fraction must match x")
    xs: list[float] = []
    ys: list[float] = []
    ms: list[float] = []
    for y0 in range(0, x.shape[0], block):
        for x0 in range(0, x.shape[1], block):
            region = np.s_[y0 : y0 + block, x0 : x0 + block]
            valid = np.isfinite(x[region]) & np.isfinite(y[region])
            if valid.mean() < 0.8:
                continue
            xs.append(float(np.mean(x[region][valid])))
            ys.append(float(np.mean(y[region][valid])))
            ms.append(float(np.mean(mask[region][valid])))
    return np.asarray(xs), np.asarray(ys), np.asarray(ms)


def _huber_affine(
    x: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray | None = None,
    iterations: int = 30,
) -> tuple[float, float, np.ndarray]:
    """Fit ``y = slope*x + intercept`` with deterministic Huber IRLS."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 6 or x.shape != y.shape:
        raise ValueError("at least six paired samples are required")
    center = float(np.median(x))
    scale = float(np.median(np.abs(x - center)) * 1.4826)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("low-resolution tracer has no usable dynamic range")
    z = (x - center) / scale
    design = np.column_stack((z, np.ones_like(z)))
    weights0 = (
        np.ones_like(x)
        if base_weight is None
        else np.clip(np.asarray(base_weight, dtype=float), 0.0, np.inf)
    )
    weights = weights0.copy()
    coefficients = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None
    )[0]
    for _ in range(int(iterations)):
        residual = y - design @ coefficients
        robust_scale = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
        robust_scale = max(robust_scale, np.finfo(float).eps)
        u = np.abs(residual) / (1.345 * robust_scale)
        huber = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1.0))
        weights = weights0 * huber
        updated = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None
        )[0]
        if np.max(np.abs(updated - coefficients)) < 1.0e-10:
            coefficients = updated
            break
        coefficients = updated
    slope = float(coefficients[0] / scale)
    intercept = float(coefficients[1] - coefficients[0] * center / scale)
    return slope, intercept, weights


def calibrate_lowres_to_optical(
    optical: np.ndarray,
    lowres: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    lowres_beam_fwhm_arcmin: float,
    optical_fwhm_arcsec: float = 0.0,
    mask_fraction: np.ndarray | None = None,
    bootstrap_samples: int = 400,
    seed: int = 4703,
) -> tuple[np.ndarray, np.ndarray, AbsoluteCalibration]:
    """Calibrate a low-resolution image to an absolute optical map.

    The primary fit includes every valid beam-sized block and does not use the
    source mask.  A mask-aware refit is used only to inflate uncertainty.
    """

    optical = np.asarray(optical, dtype=float)
    lowres = np.asarray(lowres, dtype=float)
    if optical.shape != lowres.shape or optical.ndim != 2:
        raise ValueError("optical and lowres must be matching 2-D arrays")
    if pixel_scale_arcsec <= 0.0 or lowres_beam_fwhm_arcmin <= 0.0:
        raise ValueError("pixel scale and beam must be positive")
    target_fwhm_arcsec = 60.0 * float(lowres_beam_fwhm_arcmin)
    if optical_fwhm_arcsec < 0.0 or optical_fwhm_arcsec >= target_fwhm_arcsec:
        raise ValueError("optical FWHM must lie below the low-resolution beam")
    match_fwhm = math.sqrt(target_fwhm_arcsec**2 - optical_fwhm_arcsec**2)
    matched, support = normalized_gaussian(
        optical, match_fwhm / (2.354820045 * pixel_scale_arcsec)
    )
    valid = np.isfinite(lowres) & np.isfinite(matched) & (support >= 0.8)
    block = max(1, int(round(target_fwhm_arcsec / pixel_scale_arcsec)))
    bx, by, bm = block_means(
        np.where(valid, lowres, np.nan),
        np.where(valid, matched, np.nan),
        block,
        mask_fraction,
    )
    slope, intercept, _ = _huber_affine(bx, by)
    prediction = slope * lowres + intercept
    residual = by - (slope * bx + intercept)
    residual_mad = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    correlation = float(np.corrcoef(bx, by)[0, 1])

    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(bootstrap_samples), 2), dtype=float)
    for index in range(boot.shape[0]):
        take = rng.integers(0, bx.size, size=bx.size)
        try:
            boot[index, :2] = _huber_affine(bx[take], by[take])[:2]
        except ValueError:
            boot[index] = np.nan
    base_uncertainty = np.nanstd(boot, axis=0, ddof=1)

    # The mask is deliberately absent from the primary estimator.  It only
    # defines this sensitivity refit, whose coefficient shifts inflate errors.
    mask_weight = np.clip(1.0 - bm, 0.05, 1.0)
    masked_slope, masked_intercept, _ = _huber_affine(bx, by, mask_weight)
    shifts = np.abs(
        np.asarray([masked_slope - slope, masked_intercept - intercept], dtype=float)
    )
    inflated = np.sqrt(base_uncertainty**2 + shifts**2)
    inflation = float(
        np.nanmax(
            np.divide(
                inflated,
                base_uncertainty,
                out=np.ones_like(inflated),
                where=base_uncertainty > 0.0,
            )
        )
    )
    slope_sigma = float(
        shifts[0] / base_uncertainty[0] if base_uncertainty[0] > 0.0 else np.inf
    )
    intercept_sigma = float(
        shifts[1] / base_uncertainty[1] if base_uncertainty[1] > 0.0 else np.inf
    )
    statements = [
        "All valid beam-sized blocks were retained in the primary affine calibration.",
        "The source mask was used only for a sensitivity refit, not as a model input.",
    ]
    if inflation > 1.05:
        statements.append(
            f"Calibration uncertainty was inflated by {inflation:.2f}x because of mask sensitivity."
        )
    else:
        statements.append("Mask sensitivity did not materially inflate calibration uncertainty.")
    result = AbsoluteCalibration(
        slope=slope,
        intercept=intercept,
        slope_uncertainty=float(inflated[0]),
        intercept_uncertainty=float(inflated[1]),
        correlation=correlation,
        residual_mad=residual_mad,
        blocks_used=int(bx.size),
        block_pixels=block,
        mask_sensitivity_slope_sigma=slope_sigma,
        mask_sensitivity_intercept_sigma=intercept_sigma,
        uncertainty_inflation=inflation,
        statements=tuple(statements),
    )
    return matched.astype(np.float32), prediction.astype(np.float32), result


class _RestrictedNeuralBackbone(torch.nn.Module):
    """Low-capacity coordinate field with no Fourier or fine-grid features."""

    def __init__(self, mean: float, scale: float) -> None:
        super().__init__()
        self.field = MultiResolutionLatentField(
            # The finest level is comparable to the number of independent
            # low-resolution beams across the 60 arcmin field. Bicubic
            # interpolation removes the derivative discontinuities produced
            # by the original low-resolution bilinear grids.
            grid_resolutions=(4, 8, 12),
            channels_per_level=2,
            hidden_features=16,
            hidden_layers=1,
            output_scale=max(float(scale), 1.0e-6),
            include_coordinates=True,
            rotate_and_offset_grids=True,
            grid_frame_seed=4703,
            fourier_feature_count=0,
            interpolation_mode="bicubic",
        )
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))

    def forward(self, shape: tuple[int, int]) -> torch.Tensor:
        coordinates = coordinate_grid(shape, device=self.mean.device)
        return self.mean + self.field(coordinates)


class _CoarseGridBackbone(torch.nn.Module):
    """Single bilinearly interpolated grid used as the transparent control."""

    def __init__(self, mean: float, scale: float, resolution: int = 12) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(4703)
        initial = torch.randn((1, 1, resolution, resolution), generator=generator)
        self.grid = torch.nn.Parameter(float(mean) + 0.01 * float(scale) * initial)

    def forward(self, shape: tuple[int, int]) -> torch.Tensor:
        return torch_f.interpolate(
            self.grid,
            size=shape,
            mode="bilinear",
            align_corners=True,
        )[0, 0]


def _torch_gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(4.0 * float(sigma))))
    coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel /= kernel.sum()
    value = image[None, None]
    value = torch_f.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = torch_f.conv2d(value, kernel[None, None, None, :])
    value = torch_f.pad(value, (0, 0, radius, radius), mode="reflect")
    return torch_f.conv2d(value, kernel[None, None, :, None])[0, 0]


def _curvature(image: torch.Tensor) -> torch.Tensor:
    dxx = image[:, 2:] - 2.0 * image[:, 1:-1] + image[:, :-2]
    dyy = image[2:, :] - 2.0 * image[1:-1, :] + image[:-2, :]
    return dxx.square().mean() + dyy.square().mean()


def checkerboard_holdout(
    shape: tuple[int, int], block_pixels: int, fraction: int = 5
) -> np.ndarray:
    """Select deterministic beam-sized blocks for spatial validation."""

    y, x = np.indices(shape)
    by = y // max(1, int(block_pixels))
    bx = x // max(1, int(block_pixels))
    return ((3 * by + 2 * bx) % int(fraction)) == 0


def _fit_model(
    model: torch.nn.Module,
    target: np.ndarray,
    *,
    beam_sigma_pixels: float,
    train_mask: np.ndarray,
    steps: int,
    learning_rate: float,
    torch_threads: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    torch.set_num_threads(int(torch_threads))
    target_t = torch.as_tensor(target, dtype=torch.float32)
    valid_t = torch.as_tensor(np.isfinite(target) & train_mask, dtype=torch.bool)
    fill = float(np.nanmedian(target))
    target_t = torch.nan_to_num(target_t, nan=fill)
    scale = float(np.nanstd(target))
    scale = max(scale, 1.0e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    started = time.perf_counter()
    completed = 0
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        unfiltered = model(tuple(target.shape))
        # The low-resolution observation cannot identify arbitrarily sharp
        # structure in its pre-beam field.  A common representation band-limit
        # suppresses interpolation facets in both candidate parameterizations.
        representation_sigma = max(float(beam_sigma_pixels) / 4.0, 0.75)
        raw = _torch_gaussian(unfiltered, representation_sigma)
        observed = _torch_gaussian(raw, float(beam_sigma_pixels))
        data_loss = ((observed[valid_t] - target_t[valid_t]) / scale).square().mean()
        curvature_loss = _curvature(raw) / scale**2
        subbeam = raw - _torch_gaussian(raw, max(float(beam_sigma_pixels) / 2.0, 1.0))
        subbeam_loss = subbeam.square().mean() / scale**2
        loss = data_loss + 0.02 * curvature_loss + 0.05 * subbeam_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        completed = step + 1
        value = float(loss.detach())
        if value < best - 1.0e-7:
            best = value
            best_state = {key: item.detach().clone() for key, item in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 50:
            break
    runtime = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        unfiltered = model(tuple(target.shape))
        raw = _torch_gaussian(
            unfiltered, max(float(beam_sigma_pixels) / 4.0, 0.75)
        )
        observed = _torch_gaussian(raw, float(beam_sigma_pixels))
    return raw.numpy(), observed.numpy(), completed, runtime


def _fit_metrics(
    method: str,
    model: torch.nn.Module,
    target: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    train_mask: np.ndarray,
    holdout: np.ndarray,
    runtime: float,
    steps: int,
    beam_pixels: float,
) -> BackboneFit:
    valid = np.isfinite(target)
    train = valid & train_mask
    test = valid & holdout
    residual = observed - target
    train_rmse = float(np.sqrt(np.mean(residual[train] ** 2)))
    holdout_rmse = float(np.sqrt(np.mean(residual[test] ** 2)))
    beam_rmse = float(np.sqrt(np.mean(residual[valid] ** 2)))
    correlation = float(np.corrcoef(observed[valid], target[valid])[0, 1])
    gy, gx = np.gradient(raw)
    gradient = np.hypot(gx, gy)
    edge_width = max(1, int(round(beam_pixels)))
    edge = np.zeros(raw.shape, dtype=bool)
    edge[:edge_width] = edge[-edge_width:] = True
    edge[:, :edge_width] = edge[:, -edge_width:] = True
    edge_gradient_ratio = float(
        np.sqrt(np.mean(gradient[edge] ** 2))
        / max(np.sqrt(np.mean(gradient[~edge] ** 2)), np.finfo(float).eps)
    )
    orientation = np.arctan2(gy, gx)
    gradient_weight = gradient**2
    axial_gradient_moment = float(
        np.sum(gradient_weight * np.cos(4.0 * orientation))
        / max(np.sum(gradient_weight), np.finfo(float).eps)
    )
    smooth = gaussian_filter(raw, max(beam_pixels / 2.354820045, 1.0), mode="reflect")
    high_frequency_fraction = float(
        np.var(raw - smooth) / max(np.var(raw), np.finfo(float).eps)
    )
    statements = [
        "Fourier features and fine latent grids were disabled.",
        "The fit used a beam forward operator, a common representation band-limit, and a curvature prior.",
        "Modes finer than the representation band-limit are not interpreted as measured morphology.",
    ]
    if method == "restricted_neural_field":
        statements.append(
            "The neural field used rotated bicubic 4/8/12 latent grids; the finest grid is comparable to the number of beams across the field."
        )
    if edge_gradient_ratio > 1.5:
        statements.append(
            f"Edge gradients are elevated ({edge_gradient_ratio:.2f}x interior); this candidate is flagged."
        )
    if abs(axial_gradient_moment) > 0.2:
        statements.append(
            "Axis-aligned gradient moment is "
            f"{axial_gradient_moment:.3f}; this candidate is flagged."
        )
    if high_frequency_fraction > 0.05:
        statements.append(
            f"Sub-beam variance fraction is {high_frequency_fraction:.3f}; this candidate is flagged."
        )
    return BackboneFit(
        method=method,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=float(runtime),
        steps=int(steps),
        train_rmse=train_rmse,
        holdout_rmse=holdout_rmse,
        beam_rmse=beam_rmse,
        beam_correlation=correlation,
        edge_gradient_ratio=edge_gradient_ratio,
        axial_gradient_moment=axial_gradient_moment,
        representation_fwhm_pixels=float(beam_pixels / 4.0),
        high_frequency_fraction=high_frequency_fraction,
        statements=tuple(statements),
    )


def compare_restricted_backbones(
    calibrated_lowres: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    lowres_beam_fwhm_arcmin: float,
    steps: int = 350,
    learning_rate: float = 0.03,
    torch_threads: int = 4,
    seed: int = 4703,
    methods: tuple[str, ...] = ("restricted_neural_field", "coarse_grid"),
) -> dict[str, tuple[np.ndarray, np.ndarray, BackboneFit]]:
    """Fit the requested restricted backbone parameterizations."""

    target = np.asarray(calibrated_lowres, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError("calibrated_lowres must be 2-D")
    beam_pixels = 60.0 * float(lowres_beam_fwhm_arcmin) / float(pixel_scale_arcsec)
    sigma = beam_pixels / 2.354820045
    holdout = checkerboard_holdout(target.shape, max(2, int(round(beam_pixels))))
    train_mask = ~holdout
    mean = float(np.nanmean(target))
    scale = float(np.nanstd(target))
    available_factories = {
        "restricted_neural_field": lambda: _RestrictedNeuralBackbone(mean, scale),
        "coarse_grid": lambda: _CoarseGridBackbone(mean, scale),
    }
    unknown = set(methods) - set(available_factories)
    if unknown:
        raise ValueError("Unknown backbone method(s): " + ", ".join(sorted(unknown)))
    factories = {name: available_factories[name] for name in methods}
    results: dict[str, tuple[np.ndarray, np.ndarray, BackboneFit]] = {}
    for index, (name, factory) in enumerate(factories.items()):
        torch.manual_seed(int(seed) + index)
        cv_model = factory()
        _, cv_observed, _, _ = _fit_model(
            cv_model,
            target,
            beam_sigma_pixels=sigma,
            train_mask=train_mask,
            steps=max(40, int(steps * 0.7)),
            learning_rate=learning_rate,
            torch_threads=torch_threads,
        )
        torch.manual_seed(int(seed) + index)
        full_model = factory()
        raw, observed, completed, runtime = _fit_model(
            full_model,
            target,
            beam_sigma_pixels=sigma,
            train_mask=np.ones(target.shape, dtype=bool),
            steps=steps,
            learning_rate=learning_rate,
            torch_threads=torch_threads,
        )
        # Metrics use the held-out response from the CV fit, while beam/edge
        # diagnostics use the final all-pixel candidate.
        metrics = _fit_metrics(
            name,
            full_model,
            target,
            raw,
            observed,
            np.ones(target.shape, dtype=bool),
            holdout,
            runtime,
            completed,
            beam_pixels,
        )
        cv_residual = cv_observed - target
        cv_holdout_rmse = float(
            np.sqrt(np.nanmean(cv_residual[holdout & np.isfinite(target)] ** 2))
        )
        metrics = BackboneFit(
            **{
                **metrics.metadata(),
                "holdout_rmse": cv_holdout_rmse,
            }
        )
        results[name] = (raw.astype(np.float32), observed.astype(np.float32), metrics)
    return results
