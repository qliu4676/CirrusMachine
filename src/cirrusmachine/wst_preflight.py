"""Data-dependent preflight analysis for single-image WST synthesis.

The preflight deliberately stops before image optimization.  It measures which
parts of a source-cleaned observation are sufficiently stable to become
synthesis targets, estimates a smooth ensemble power spectrum, and benchmarks
the differentiable scattering operator.  Mask geometry is used only to
estimate target uncertainty; it is never exposed to a synthesized image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Iterable

import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import torch

from .power_spectrum import power_spectrum_2d, radial_power_spectrum
from .scattering_mean import (
    MeanFeature,
    ScatteringMean2D,
    isotropic_feature_set,
    largest_dyadic_j,
    within_band_pairs,
)


@dataclass(frozen=True)
class WSTPreflightConfig:
    """Resolved assumptions that determine reusable preflight targets."""

    binning: int = 3
    pixel_scale_arcsec: float = 7.5
    J: int = 8
    candidate_L: tuple[int, ...] = (4, 8)
    patch_pixels: int = 240
    patch_stride: int = 120
    boundary_sigma: float = 1.0
    guard_sigma: float = 2.0
    null_mask_count: int = 24
    mask_weight_exponent: float = 2.0
    mask_shift_limit_sigma: float = 0.5
    mask_unstable_fraction: float = 0.10
    boundary_limit_sigma: float = 1.0
    native_reliable_pixels: int = 4
    quantile_count: int = 41
    random_seed: int = 4703
    optimizer_steps: int = 300
    torch_threads: int = 4
    algorithm_version: int = 3

    @property
    def pixel_scale_arcmin(self) -> float:
        return self.pixel_scale_arcsec / 60.0

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible cache metadata."""

        payload = asdict(self)
        payload["candidate_L"] = list(self.candidate_L)
        payload["assumptions"] = {
            "independent_phases": True,
            "latent_cirrus": True,
            "no_deconvolution": True,
            "no_positivity_constraint": True,
            "mask_only_in_target_uncertainty": True,
            "md07_omitted": True,
            "optimizer": "Adam",
        }
        return payload


def crop_and_bin(
    image: np.ndarray,
    mask: np.ndarray,
    factor: int,
    *,
    output_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to a binning multiple and return mean image and mask fraction."""

    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape or image.ndim != 2:
        raise ValueError("image and mask must be matching 2-D arrays")
    if factor < 1:
        raise ValueError("factor must be positive")
    if output_pixels is None:
        ny = image.shape[0] // factor * factor
        nx = image.shape[1] // factor * factor
    else:
        ny = nx = int(output_pixels) * factor
        if image.shape[0] < ny or image.shape[1] < nx:
            raise ValueError(
                f"Input {image.shape} is smaller than requested {(ny, nx)} crop"
            )
    image = image[:ny, :nx]
    mask = mask[:ny, :nx]
    binned = image.reshape(ny // factor, factor, nx // factor, factor).mean((1, 3))
    mask_fraction = mask.reshape(
        ny // factor, factor, nx // factor, factor
    ).mean((1, 3))
    return binned, mask_fraction


def patch_coordinates(
    shape: tuple[int, int], patch_pixels: int, stride: int
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic overlapping patch grid including both edges."""

    ny, nx = shape
    if patch_pixels > min(shape):
        raise ValueError("patch_pixels exceeds the image")
    ys = list(range(0, ny - patch_pixels + 1, stride))
    xs = list(range(0, nx - patch_pixels + 1, stride))
    if ys[-1] != ny - patch_pixels:
        ys.append(ny - patch_pixels)
    if xs[-1] != nx - patch_pixels:
        xs.append(nx - patch_pixels)
    return tuple((y, x) for y in ys for x in xs)


def extract_patches(
    image: np.ndarray,
    coords: Iterable[tuple[int, int]],
    patch_pixels: int,
) -> np.ndarray:
    """Extract a stack of square patches."""

    return np.stack(
        [
            image[y : y + patch_pixels, x : x + patch_pixels]
            for y, x in coords
        ]
    )


def _normalized_patch_weights(
    mask_fraction: np.ndarray,
    coords: tuple[tuple[int, int], ...],
    patch_pixels: int,
    exponent: float,
) -> np.ndarray:
    reliability = np.clip(1.0 - mask_fraction, 0.0, 1.0) ** exponent
    weights = np.asarray(
        [
            reliability[y : y + patch_pixels, x : x + patch_pixels].mean()
            for y, x in coords
        ],
        dtype=float,
    )
    return weights / weights.sum()


def _null_mask_weights(
    mask_fraction: np.ndarray,
    coords: tuple[tuple[int, int], ...],
    patch_pixels: int,
    exponent: float,
    count: int,
    seed: int,
) -> np.ndarray:
    """Translate/rotate a mask while preserving its area and morphology."""

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        transformed = np.rot90(mask_fraction, int(rng.integers(0, 4)))
        if transformed.shape != mask_fraction.shape:
            transformed = transformed.T
        shift = (
            int(rng.integers(0, mask_fraction.shape[0])),
            int(rng.integers(0, mask_fraction.shape[1])),
        )
        transformed = np.roll(transformed, shift, axis=(0, 1))
        rows.append(
            _normalized_patch_weights(
                transformed, coords, patch_pixels, exponent
            )
        )
    return np.asarray(rows)


def _weighted_average(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(samples) * np.asarray(weights)[:, None], axis=0)


def _safe_scale(samples: np.ndarray) -> np.ndarray:
    scale = np.asarray(mad_std(samples, axis=0, ignore_nan=True), dtype=float)
    fallback = np.nanstd(samples, axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, fallback)
    return np.maximum(scale, 1e-6)


def _feature_table(features: tuple[MeanFeature, ...]) -> dict[str, np.ndarray]:
    return {
        "order": np.asarray([feature.order for feature in features], dtype=np.int8),
        "j1": np.asarray([feature.j1 for feature in features], dtype=np.int8),
        "j2": np.asarray([feature.j2 for feature in features], dtype=np.int8),
        "delta_l": np.asarray(
            [feature.delta_l for feature in features], dtype=np.int8
        ),
        "label": np.asarray([feature.label for feature in features]),
    }


def measure_wst_stability(
    image: np.ndarray,
    mask_fraction: np.ndarray,
    config: WSTPreflightConfig,
    L: int,
) -> dict[str, np.ndarray]:
    """Measure patch, mask, and boundary sensitivity of reduced WST features."""

    coords = patch_coordinates(
        image.shape, config.patch_pixels, config.patch_stride
    )
    patches = extract_patches(image, coords, config.patch_pixels)
    patch_J = min(config.J, largest_dyadic_j(patches.shape[-2:]))
    patch_features = isotropic_feature_set(
        range(patch_J), within_band_pairs(range(patch_J)), L
    )
    patch_calculator = ScatteringMean2D(
        (config.patch_pixels, config.patch_pixels),
        patch_J,
        L,
        mask_boundary=True,
        boundary_sigma=config.boundary_sigma,
        device="cpu",
    )
    with torch.no_grad():
        patch_values = (
            patch_calculator.log_features(patches, patch_features).cpu().numpy()
        )

    weights = _normalized_patch_weights(
        mask_fraction,
        coords,
        config.patch_pixels,
        config.mask_weight_exponent,
    )
    null_weights = _null_mask_weights(
        mask_fraction,
        coords,
        config.patch_pixels,
        config.mask_weight_exponent,
        config.null_mask_count,
        config.random_seed + L,
    )
    unweighted = np.mean(patch_values, axis=0)
    weighted = _weighted_average(patch_values, weights)
    null_centers = np.stack(
        [_weighted_average(patch_values, row) for row in null_weights]
    )
    uncertainty = _safe_scale(patch_values)
    mask_shift_sigma = np.abs(weighted - unweighted) / uncertainty
    null_shift_sigma = np.abs(null_centers - unweighted[None]) / uncertainty
    null_p95 = np.quantile(null_shift_sigma, 0.95, axis=0)

    full_features = isotropic_feature_set(
        range(config.J), within_band_pairs(range(config.J)), L
    )
    full_plain = ScatteringMean2D(
        image.shape,
        config.J,
        L,
        mask_boundary=False,
        device="cpu",
    )
    full_interior = ScatteringMean2D(
        image.shape,
        config.J,
        L,
        mask_boundary=True,
        boundary_sigma=config.boundary_sigma,
        device="cpu",
    )
    with torch.no_grad():
        plain_values = (
            full_plain.log_features(image, full_features)[0].cpu().numpy()
        )
        interior_values = (
            full_interior.log_features(image, full_features)[0].cpu().numpy()
        )

    patch_index = {feature: i for i, feature in enumerate(patch_features)}
    full_index = {feature: i for i, feature in enumerate(full_features)}
    target = np.full(len(full_features), np.nan)
    spread = np.full(len(full_features), np.nan)
    mask_sigma = np.full(len(full_features), np.nan)
    null_sigma = np.full(len(full_features), np.nan)
    boundary_sigma = np.full(len(full_features), np.nan)
    patch_supported = np.zeros(len(full_features), dtype=bool)
    for feature, output_index in full_index.items():
        if feature not in patch_index:
            continue
        input_index = patch_index[feature]
        target[output_index] = weighted[input_index]
        spread[output_index] = uncertainty[input_index]
        mask_sigma[output_index] = mask_shift_sigma[input_index]
        null_sigma[output_index] = null_p95[input_index]
        boundary_sigma[output_index] = (
            abs(plain_values[output_index] - interior_values[output_index])
            / uncertainty[input_index]
        )
        patch_supported[output_index] = True

    mask_unstable = (
        np.nanmean(mask_sigma > config.mask_shift_limit_sigma)
        > config.mask_unstable_fraction
    )
    if mask_unstable:
        # Roll back mask-weighted centers while retaining mask sensitivity as
        # uncertainty, exactly as agreed during design review.
        for feature, output_index in full_index.items():
            if feature in patch_index:
                target[output_index] = unweighted[patch_index[feature]]
    target = np.where(patch_supported, target, interior_values)
    spread = np.where(
        patch_supported,
        np.maximum(spread, np.abs(interior_values - plain_values)),
        np.nan,
    )

    table = _feature_table(full_features)
    return {
        **{f"feature_{key}": value for key, value in table.items()},
        "target": target,
        "patch_uncertainty": spread,
        "mask_shift_sigma": mask_sigma,
        "null_mask_p95_sigma": null_sigma,
        "boundary_shift_sigma": boundary_sigma,
        "patch_supported": patch_supported.astype(np.int8),
        "full_plain": plain_values,
        "full_interior": interior_values,
        "mask_weighting_accepted": np.asarray(not mask_unstable, dtype=np.int8),
        "patch_values": patch_values,
        "patch_weights": weights,
        "null_patch_weights": null_weights,
        "patch_coords": np.asarray(coords, dtype=np.int16),
        "patch_pixels": np.asarray(config.patch_pixels, dtype=np.int16),
    }


def analyze_power_spectrum(
    image: np.ndarray,
    mask_fraction: np.ndarray,
    config: WSTPreflightConfig,
) -> dict[str, np.ndarray]:
    """Estimate a smooth hybrid ensemble spectrum and sample its variance."""

    full_power = power_spectrum_2d(image)
    frequency, radial, radial_std = radial_power_spectrum(
        full_power, config.pixel_scale_arcmin, nbins=55
    )
    coords = patch_coordinates(
        image.shape, config.patch_pixels, config.patch_stride
    )
    patches = extract_patches(image, coords, config.patch_pixels)
    patch_radial = []
    patch_frequency = None
    for patch in patches:
        pf, pp, _ = radial_power_spectrum(
            power_spectrum_2d(patch),
            config.pixel_scale_arcmin,
            nbins=38,
        )
        if patch_frequency is None:
            patch_frequency = pf
        patch_radial.append(np.interp(patch_frequency, pf, pp))
    patch_frequency = np.asarray(patch_frequency)
    patch_radial = np.asarray(patch_radial)
    patch_log_scatter = _safe_scale(np.log(patch_radial))

    log_frequency = np.log(frequency)
    log_power = np.log(np.maximum(radial, np.finfo(float).tiny))
    # The spline is the non-parametric correction to a globally physical
    # power-law trend. Its smoothing prevents individual modes from becoming
    # synthesis targets.
    large_scale = (
        (frequency >= 1.0 / 30.0)
        & (frequency <= 1.0 / 4.0)
    )
    slope, intercept = np.polyfit(
        log_frequency[large_scale], log_power[large_scale], 1
    )
    residual = log_power - (intercept + slope * log_frequency)
    correction = UnivariateSpline(
        log_frequency,
        residual,
        s=max(1.0, 0.35 * residual.size),
        k=3,
    )
    fitted_log_power = intercept + slope * log_frequency + correction(log_frequency)
    local_slope = slope + correction.derivative()(log_frequency)

    rng = np.random.default_rng(config.random_seed)
    fy = np.fft.fftshift(
        np.fft.fftfreq(image.shape[0], d=config.pixel_scale_arcmin)
    )
    fx = np.fft.fftshift(
        np.fft.fftfreq(image.shape[1], d=config.pixel_scale_arcmin)
    )
    radii = np.hypot(fy[:, None], fx[None, :])
    edges = np.geomspace(frequency.min() * 0.8, frequency.max() * 1.1, frequency.size + 1)
    mode_count = np.histogram(radii[radii > 0], bins=edges)[0]
    mode_count = np.maximum(mode_count, 1)
    sampled_radial = np.stack(
        [
            np.exp(fitted_log_power)
            * rng.gamma(shape=mode_count, scale=1.0 / mode_count)
            for _ in range(8)
        ]
    )

    theta = np.arctan2(fy[:, None], fx[None, :])
    quadrupole = np.full(frequency.size, np.nan)
    axial_ratio = np.full(frequency.size, np.nan)
    for i in range(frequency.size):
        use = (radii >= edges[i]) & (radii < edges[i + 1])
        values = full_power[use]
        if values.size < 4 or not np.isfinite(values).any():
            continue
        angles = theta[use]
        quadrupole[i] = abs(np.sum(values * np.exp(2j * angles)) / np.sum(values))
        axial = np.abs(np.sin(2.0 * angles)) < np.sin(np.deg2rad(10.0))
        diagonal = np.abs(np.cos(2.0 * angles)) < np.sin(np.deg2rad(10.0))
        if axial.any() and diagonal.any():
            axial_ratio[i] = np.mean(values[axial]) / np.mean(values[diagonal])

    return {
        "power_2d": full_power,
        "frequency": frequency,
        "radial_power": radial,
        "radial_std": radial_std,
        "fitted_power": np.exp(fitted_log_power),
        "local_slope": local_slope,
        "large_scale_slope": np.asarray(slope),
        "sampled_radial": sampled_radial,
        "mode_count": mode_count,
        "quadrupole": quadrupole,
        "axial_ratio": axial_ratio,
        "patch_frequency": patch_frequency,
        "patch_radial": patch_radial,
        "patch_log_scatter": patch_log_scatter,
    }


def analyze_cdf(
    image: np.ndarray,
    mask_fraction: np.ndarray,
    config: WSTPreflightConfig,
    smoothing_sigmas: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
) -> dict[str, np.ndarray]:
    """Measure robust CDF stability across smoothing and spatial patches."""

    levels = np.linspace(0.005, 0.995, config.quantile_count)
    coords = patch_coordinates(
        image.shape, config.patch_pixels, config.patch_stride
    )
    weights = _normalized_patch_weights(
        mask_fraction,
        coords,
        config.patch_pixels,
        config.mask_weight_exponent,
    )
    values = []
    patch_values = []
    patch_uncertainty = []
    mask_shift = []
    null_weights = _null_mask_weights(
        mask_fraction,
        coords,
        config.patch_pixels,
        config.mask_weight_exponent,
        config.null_mask_count,
        config.random_seed + 91,
    )
    for sigma in smoothing_sigmas:
        smoothed = (
            image
            if sigma == 0.0
            else gaussian_filter(image, sigma=sigma, mode="reflect")
        )
        values.append(np.quantile(smoothed, levels))
        patches = extract_patches(smoothed, coords, config.patch_pixels)
        q = np.stack([np.quantile(patch, levels) for patch in patches])
        patch_values.append(q)
        scale = _safe_scale(q)
        patch_uncertainty.append(scale)
        unweighted = q.mean(axis=0)
        weighted = _weighted_average(q, weights)
        null_centers = np.stack([_weighted_average(q, row) for row in null_weights])
        real = np.abs(weighted - unweighted) / scale
        null = np.quantile(
            np.abs(null_centers - unweighted[None]) / scale, 0.95, axis=0
        )
        mask_shift.append(np.stack((real, null)))
    return {
        "quantile_levels": levels,
        "smoothing_sigma_pixels": np.asarray(smoothing_sigmas),
        "quantiles": np.asarray(values),
        "patch_quantiles": np.asarray(patch_values),
        "patch_uncertainty": np.asarray(patch_uncertainty),
        "mask_shift_sigma": np.asarray(mask_shift),
    }


def analyze_self_modulation(
    image: np.ndarray,
    mask_fraction: np.ndarray,
    config: WSTPreflightConfig,
    scales: Iterable[int],
) -> dict[str, np.ndarray]:
    """Test monotonic local-amplitude modulation without imposing a law."""

    scale_list = tuple(int(j) for j in scales)
    bin_centers = []
    bin_rms = []
    rho = []
    quadrant_rho = []
    accepted = []
    for j in scale_list:
        wavelength = 2**j
        fine = gaussian_filter(image, wavelength / 2.0, mode="reflect")
        coarse = gaussian_filter(image, wavelength, mode="reflect")
        detail = fine - coarse
        context = max(2.0, 1.5 * wavelength)
        local_rms = np.sqrt(
            gaussian_filter(detail**2, context, mode="reflect")
        )
        background = gaussian_filter(image, context, mode="reflect")
        contamination = gaussian_filter(mask_fraction, context, mode="reflect")
        margin = min(int(math.ceil(2 * context)), min(image.shape) // 4)
        stride = max(1, wavelength // 2)
        region = np.s_[margin:-margin:stride, margin:-margin:stride]
        x = background[region].ravel()
        y = np.log(np.maximum(local_rms[region].ravel(), 1e-8))
        reliable = contamination[region].ravel() < 0.1
        x = x[reliable]
        y = y[reliable]
        if x.size < 16:
            bin_centers.append([np.nan] * 8)
            bin_rms.append([np.nan] * 8)
            rho.append(np.nan)
            quadrant_rho.append([np.nan] * 4)
            accepted.append(False)
            continue
        coefficient = float(spearmanr(x, y).statistic)
        rho.append(coefficient)

        edges = np.quantile(x, np.linspace(0.0, 1.0, 9))
        centers = []
        responses = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            use = (x >= left) & (x <= right)
            centers.append(np.median(x[use]))
            responses.append(np.exp(np.median(y[use])))
        bin_centers.append(centers)
        bin_rms.append(responses)

        quad = []
        ny, nx = image.shape
        for y0, y1 in ((0, ny // 2), (ny // 2, ny)):
            for x0, x1 in ((0, nx // 2), (nx // 2, nx)):
                bg = background[y0:y1, x0:x1].ravel()
                lr = np.log(np.maximum(local_rms[y0:y1, x0:x1].ravel(), 1e-8))
                good = contamination[y0:y1, x0:x1].ravel() < 0.1
                if np.count_nonzero(good) < 16:
                    quad.append(np.nan)
                else:
                    quad.append(float(spearmanr(bg[good], lr[good]).statistic))
        quadrant_rho.append(quad)
        same_sign = all(
            np.isfinite(value) and np.sign(value) == np.sign(coefficient)
            for value in quad
        )
        accepted.append(
            abs(coefficient) >= 0.30
            and same_sign
            and np.median(np.abs(quad)) >= 0.20
        )

    return {
        "scale_j": np.asarray(scale_list, dtype=np.int8),
        "scale_arcmin": config.pixel_scale_arcmin * 2 ** np.asarray(scale_list),
        "intensity_bin_center": np.asarray(bin_centers),
        "local_rms_bin_median": np.asarray(bin_rms),
        "spearman_rho": np.asarray(rho),
        "quadrant_rho": np.asarray(quadrant_rho),
        "accepted": np.asarray(accepted, dtype=np.int8),
    }


def select_scale_reliability(
    config: WSTPreflightConfig,
    wst: dict[str, np.ndarray],
    power: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply explicit conservative gates and state every resulting decision."""

    scales = np.arange(config.J)
    angular_scale = config.pixel_scale_arcmin * 2**scales
    frequency = 1.0 / angular_scale
    spectral_slope = np.interp(
        frequency,
        power["frequency"],
        power["local_slope"],
        left=np.nan,
        right=np.nan,
    )
    order = wst["feature_order"]
    j2 = wst["feature_j2"]
    outer_scale = np.where(order == 1, wst["feature_j1"], j2)
    def finite_median(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else np.nan

    mask_by_scale = np.asarray(
        [
            finite_median(wst["mask_shift_sigma"][outer_scale == j])
            for j in scales
        ]
    )
    boundary_by_scale = np.asarray(
        [
            finite_median(wst["boundary_shift_sigma"][outer_scale == j])
            for j in scales
        ]
    )
    supported = np.asarray(
        [
            np.any(wst["patch_supported"][outer_scale == j] > 0)
            for j in scales
        ]
    )

    status = np.full(config.J, "constrain", dtype="<U20")
    reason = np.full(config.J, "stable within preflight gates", dtype="<U160")
    large_scale_slope = float(power["large_scale_slope"])
    for j in scales:
        if 2**j < config.native_reliable_pixels:
            status[j] = "exclude"
            reason[j] = (
                f"{2**j}-pixel scale is below the conservative "
                f"{config.native_reliable_pixels}-pixel observation-reliability floor"
            )
        elif not supported[j]:
            status[j] = "validation-only"
            reason[j] = "insufficient independent patch support at this scale"
        elif (
            np.isfinite(spectral_slope[j])
            and spectral_slope[j] > large_scale_slope + 0.5
        ):
            status[j] = "downweight"
            reason[j] = (
                f"local spectral slope {spectral_slope[j]:.2f} is flatter than "
                f"the large-scale slope {large_scale_slope:.2f} by more than 0.50"
            )
        elif mask_by_scale[j] > config.mask_shift_limit_sigma:
            status[j] = "downweight"
            reason[j] = (
                f"median mask sensitivity {mask_by_scale[j]:.2f} sigma exceeds "
                f"{config.mask_shift_limit_sigma:.2f}"
            )
        elif boundary_by_scale[j] > config.boundary_limit_sigma:
            status[j] = "downweight"
            reason[j] = (
                f"median boundary sensitivity {boundary_by_scale[j]:.2f} sigma exceeds "
                f"{config.boundary_limit_sigma:.2f}"
            )
    return {
        "j": scales,
        "scale_arcmin": angular_scale,
        "frequency_arcmin_inv": frequency,
        "local_spectral_slope": spectral_slope,
        "mask_sensitivity_sigma": mask_by_scale,
        "boundary_sensitivity_sigma": boundary_by_scale,
        "patch_supported": supported.astype(np.int8),
        "status": status,
        "reason": reason,
    }


def selected_features_from_reliability(
    reliability: dict[str, np.ndarray],
    L: int,
    wst: dict[str, np.ndarray] | None = None,
) -> tuple[MeanFeature, ...]:
    """Select stable scales and physically distinct relative angles.

    For ``L=8`` the canonical 0, 45, and 90 degree channels retain parallel,
    oblique, and perpendicular morphology at substantially lower CPU cost than
    all five mirror-unique samples.  Coefficients with extreme (>3 sigma)
    boundary or mask sensitivity are omitted when a measured WST table is
    supplied; milder sensitivity remains in the set with inflated uncertainty.
    """

    allowed = {
        int(j)
        for j, status in zip(reliability["j"], reliability["status"], strict=True)
        if status in {"constrain", "downweight"}
    }
    relative_angles = tuple(range(L // 2 + 1)) if L <= 4 else (0, L // 4, L // 2)
    features = [MeanFeature(1, j) for j in sorted(allowed)]
    for j1, j2 in within_band_pairs(allowed):
        features.extend(MeanFeature(2, j1, j2, dl) for dl in relative_angles)
    if wst is None:
        return tuple(features)
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
    selected = []
    for feature in features:
        key = (feature.order, feature.j1, feature.j2, feature.delta_l)
        index = lookup[key]
        mask_sigma = wst["mask_shift_sigma"][index]
        boundary_sigma = wst["boundary_shift_sigma"][index]
        if (
            np.isfinite(mask_sigma)
            and mask_sigma > 3.0
            or np.isfinite(boundary_sigma)
            and boundary_sigma > 3.0
        ):
            continue
        selected.append(feature)
    return tuple(selected)


def coefficient_action_counts(
    reliability: dict[str, np.ndarray],
    wst: dict[str, np.ndarray],
    L: int,
) -> dict[str, int]:
    """Count candidate coefficient actions after scale and sensitivity gates."""

    candidates = selected_features_from_reliability(reliability, L)
    kept = set(selected_features_from_reliability(reliability, L, wst))
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
    counts = {"constrain": 0, "downweight": 0, "exclude": 0}
    scale_status = {
        int(j): str(status)
        for j, status in zip(
            reliability["j"], reliability["status"], strict=True
        )
    }
    for feature in candidates:
        if feature not in kept:
            counts["exclude"] += 1
            continue
        index = lookup[
            (feature.order, feature.j1, feature.j2, feature.delta_l)
        ]
        involved_scales = (feature.j1,) if feature.order == 1 else (
            feature.j1,
            feature.j2,
        )
        if any(scale_status[scale] == "downweight" for scale in involved_scales) or (
            wst["mask_shift_sigma"][index] > 0.5
            or wst["boundary_shift_sigma"][index] > 1.0
        ):
            counts["downweight"] += 1
        else:
            counts["constrain"] += 1
    return counts


def benchmark_scattering_step(
    config: WSTPreflightConfig,
    reliability: dict[str, np.ndarray],
    L: int,
    wst: dict[str, np.ndarray] | None = None,
    benchmark_shape: tuple[int, int] = (480, 480),
) -> dict[str, np.ndarray]:
    """Time one differentiable descriptor evaluation without optimization."""

    torch.set_num_threads(config.torch_threads)
    features = selected_features_from_reliability(reliability, L, wst)
    if not features:
        raise ValueError("No scattering features passed reliability selection")
    calculator = ScatteringMean2D(
        benchmark_shape, config.J, L, device="cpu"
    )
    rng = np.random.default_rng(config.random_seed + L)
    image = torch.tensor(
        rng.normal(size=(1, *benchmark_shape)).astype(np.float32),
        requires_grad=True,
    )

    # Warm one FFT plan without constructing an optimizer or changing the image.
    warm_features = tuple(features[: min(len(features), max(1, L))])
    warm = calculator.log_features(image, warm_features).square().mean()
    warm.backward()
    image.grad = None

    started = time.perf_counter()
    values = calculator.log_features(image, features)
    loss = values.square().mean()
    loss.backward()
    elapsed = time.perf_counter() - started

    max_j = max(
        feature.j1 if feature.order == 1 else feature.j2
        for feature in features
    )
    guard = int(math.ceil(config.guard_sigma * 0.8 * 2**max_j))
    sizes = np.asarray((480, 720), dtype=float)
    guarded = sizes + 2 * guard
    base = float(np.prod(benchmark_shape))
    base_log = math.log2(max(benchmark_shape))
    scaling = guarded**2 * np.log2(guarded) / (base * base_log)
    step_seconds = elapsed * scaling
    projected = step_seconds * config.optimizer_steps * 1.25 / 60.0
    return {
        "L": np.asarray(L),
        "feature_count": np.asarray(len(features)),
        "benchmark_shape": np.asarray(benchmark_shape),
        "benchmark_seconds": np.asarray(elapsed),
        "max_constrained_j": np.asarray(max_j),
        "guard_pixels": np.asarray(guard),
        "output_sizes": sizes.astype(np.int16),
        "guarded_sizes": guarded.astype(np.int16),
        "projected_step_seconds": step_seconds,
        "projected_300_step_minutes": projected,
    }
