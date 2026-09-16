"""Physical low-resolution conditioning for WST texture synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import torch
import torch.nn.functional as torch_f

from .lowres_backbone import _huber_affine
from .wst_baseline import PeriodicGaussianSmoother


@dataclass(frozen=True)
class LocalModulationScaleFit:
    """One empirical log-local-RMS relation against a low-pass backbone."""

    scale_j: int
    scale_arcmin: float
    slope: float
    intercept: float
    uncertainty_log_rms: float
    spearman_rho: float
    quadrant_rho: tuple[float, ...]
    quadrant_slope: tuple[float, ...]
    mask_sensitivity_sigma: float
    accepted: bool
    action: str
    sample_count: int

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def _sample_grid(
    output_pixels: int, scale_j: int
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    wavelength = 2 ** int(scale_j)
    context = max(2.0, 1.5 * wavelength)
    margin = min(int(math.ceil(2.0 * context)), int(output_pixels) // 4)
    stride = max(1, wavelength // 2)
    y = np.arange(margin, output_pixels - margin, stride, dtype=int)
    x = np.arange(margin, output_pixels - margin, stride, dtype=int)
    return y, x, margin, stride, context


def fit_local_modulation_relations(
    optical: np.ndarray,
    lowpass_driver: np.ndarray,
    mask_fraction: np.ndarray,
    *,
    pixel_scale_arcmin: float,
    scales: tuple[int, ...] = (2, 3, 4, 5, 6),
    uncertainty_floor_log: float = math.log1p(0.10),
    global_rho_min: float = 0.25,
    positive_quadrants_min: int = 2,
    median_quadrant_rho_min: float = 0.10,
    sensitivity_scales: tuple[int, ...] = (2,),
) -> tuple[
    tuple[LocalModulationScaleFit, ...],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """Fit robust scale-dependent local amplitudes against a coarse driver.

    Every finite spatial sample enters the primary fit. The source mask only
    defines a sensitivity refit whose shift inflates the uncertainty.
    """

    optical = np.asarray(optical, dtype=float)
    driver = np.asarray(lowpass_driver, dtype=float)
    mask = np.asarray(mask_fraction, dtype=float)
    if optical.shape != driver.shape or optical.shape != mask.shape or optical.ndim != 2:
        raise ValueError("optical, lowpass_driver, and mask_fraction must match")
    fits: list[LocalModulationScaleFit] = []
    targets: dict[int, np.ndarray] = {}
    uncertainties: dict[int, np.ndarray] = {}
    ny, nx = optical.shape
    for scale_j in scales:
        wavelength = 2 ** int(scale_j)
        _, _, margin, stride, context = _sample_grid(ny, scale_j)
        fine = gaussian_filter(optical, wavelength / 2.0, mode="reflect")
        coarse = gaussian_filter(optical, wavelength, mode="reflect")
        detail = fine - coarse
        log_rms = np.log(
            np.sqrt(
                np.maximum(
                    gaussian_filter(detail**2, context, mode="reflect"),
                    np.finfo(float).tiny,
                )
            )
        )
        background = gaussian_filter(driver, context, mode="reflect")
        contamination = gaussian_filter(mask, context, mode="reflect")
        region = np.s_[margin : ny - margin : stride, margin : nx - margin : stride]
        x = background[region].ravel()
        y = log_rms[region].ravel()
        m = np.clip(contamination[region].ravel(), 0.0, 1.0)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y, m = x[valid], y[valid], m[valid]
        slope, intercept, _ = _huber_affine(x, y)
        prediction = slope * x + intercept
        residual = y - prediction
        residual_sigma = float(
            1.4826022185 * np.median(np.abs(residual - np.median(residual)))
        )
        alternative_slope, alternative_intercept, _ = _huber_affine(
            x, y, np.clip(1.0 - m, 0.05, 1.0)
        )
        alternative_prediction = alternative_slope * x + alternative_intercept
        mask_shift = float(np.sqrt(np.mean((alternative_prediction - prediction) ** 2)))

        coefficient = float(spearmanr(x, y).statistic)
        quadrant_rho: list[float] = []
        quadrant_slope: list[float] = []
        for y0, y1 in ((0, ny // 2), (ny // 2, ny)):
            for x0, x1 in ((0, nx // 2), (nx // 2, nx)):
                qx = background[y0:y1, x0:x1][::stride, ::stride].ravel()
                qy = log_rms[y0:y1, x0:x1][::stride, ::stride].ravel()
                good = np.isfinite(qx) & np.isfinite(qy)
                quadrant_rho.append(float(spearmanr(qx[good], qy[good]).statistic))
                try:
                    q_slope, _, _ = _huber_affine(qx[good], qy[good])
                except ValueError:
                    q_slope = np.nan
                quadrant_slope.append(float(q_slope))
        finite_slopes = np.asarray(quadrant_slope, dtype=float)
        slope_scatter = float(np.nanstd(finite_slopes, ddof=1))
        driver_scale = float(np.std(x))
        total_uncertainty = max(
            float(uncertainty_floor_log),
            math.sqrt(
                residual_sigma**2
                + mask_shift**2
                + (slope_scatter * driver_scale) ** 2
            ),
        )
        positive_quadrants = sum(value > 0.0 for value in quadrant_rho)
        robustly_accepted = bool(
            coefficient >= float(global_rho_min)
            and positive_quadrants >= int(positive_quadrants_min)
            and np.median(np.abs(quadrant_rho))
            >= float(median_quadrant_rho_min)
            and slope > 0.0
        )
        sensitivity_accepted = bool(
            int(scale_j) in sensitivity_scales
            and coefficient > 0.0
            and positive_quadrants >= int(positive_quadrants_min)
            and slope > 0.0
        )
        accepted = robustly_accepted or sensitivity_accepted
        if sensitivity_accepted and not robustly_accepted:
            sensitivity_factor = max(
                2.0,
                float(global_rho_min)
                / max(coefficient, np.finfo(float).eps),
            )
            total_uncertainty *= sensitivity_factor
            action = (
                "retained as sensitivity term with "
                f"{sensitivity_factor:.2f}x uncertainty"
            )
        elif robustly_accepted:
            action = "retained"
        else:
            action = (
                "discarded: Planck-conditioned modulation evidence failed "
                "rho/majority-quadrant robustness gates"
            )
        fit = LocalModulationScaleFit(
            scale_j=int(scale_j),
            scale_arcmin=float(pixel_scale_arcmin * wavelength),
            slope=float(slope),
            intercept=float(intercept),
            uncertainty_log_rms=float(total_uncertainty),
            spearman_rho=coefficient,
            quadrant_rho=tuple(quadrant_rho),
            quadrant_slope=tuple(quadrant_slope),
            mask_sensitivity_sigma=float(
                mask_shift / max(residual_sigma, np.finfo(float).eps)
            ),
            accepted=accepted,
            action=action,
            sample_count=int(x.size),
        )
        fits.append(fit)
        if accepted:
            targets[int(scale_j)] = (slope * x + intercept).astype(np.float32)
            uncertainties[int(scale_j)] = np.full(
                x.shape, total_uncertainty, dtype=np.float32
            )
    return tuple(fits), targets, uncertainties


def measure_local_rms_samples(
    image: np.ndarray, scales: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Measure the same local DoG RMS samples used by modulation fitting."""

    values = np.asarray(image, dtype=float)
    result: dict[int, np.ndarray] = {}
    for scale_j in scales:
        wavelength = 2 ** int(scale_j)
        _, _, margin, stride, context = _sample_grid(values.shape[0], scale_j)
        detail = gaussian_filter(values, wavelength / 2.0, mode="reflect")
        detail -= gaussian_filter(values, wavelength, mode="reflect")
        local_rms = np.sqrt(
            np.maximum(
                gaussian_filter(detail**2, context, mode="reflect"),
                np.finfo(float).tiny,
            )
        )
        region = np.s_[
            margin : values.shape[0] - margin : stride,
            margin : values.shape[1] - margin : stride,
        ]
        result[int(scale_j)] = local_rms[region].ravel()
    return result


def measure_driver_samples(
    driver: np.ndarray, scales: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Measure the scale-matched coarse-driver samples used in each fit."""

    values = np.asarray(driver, dtype=float)
    result: dict[int, np.ndarray] = {}
    for scale_j in scales:
        _, _, margin, stride, context = _sample_grid(values.shape[0], scale_j)
        background = gaussian_filter(values, context, mode="reflect")
        region = np.s_[
            margin : values.shape[0] - margin : stride,
            margin : values.shape[1] - margin : stride,
        ]
        result[int(scale_j)] = background[region].ravel()
    return result


class PlanckBeamDescriptor:
    """Return a pooled central beam response from a guarded synthesis canvas."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        beam_fwhm_pixels: float,
        guard_pixels: int,
        pool_pixels: int = 4,
    ) -> None:
        self.smoother = PeriodicGaussianSmoother(
            shape,
            float(beam_fwhm_pixels) / 2.354820045,
            device=torch.device("cpu"),
        )
        self.guard_pixels = int(guard_pixels)
        self.pool_pixels = int(pool_pixels)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        smoothed = self.smoother(image)
        guard = self.guard_pixels
        central = smoothed[
            ...,
            guard : smoothed.shape[-2] - guard,
            guard : smoothed.shape[-1] - guard,
        ]
        pooled = torch_f.avg_pool2d(
            central[:, None], self.pool_pixels, self.pool_pixels
        )[:, 0]
        return pooled.reshape(pooled.shape[0], -1)


class LocalDoGModulationDescriptor:
    """Differentiable multiscale local RMS sampled on robust interior grids."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        output_pixels: int,
        guard_pixels: int,
        scales: tuple[int, ...],
    ) -> None:
        self.shape = tuple(map(int, shape))
        self.output_pixels = int(output_pixels)
        self.guard_pixels = int(guard_pixels)
        self.scales = tuple(int(value) for value in scales)
        fy = torch.fft.fftfreq(self.shape[0])[:, None]
        fx = torch.fft.rfftfreq(self.shape[1])[None, :]
        radius2 = fy.square() + fx.square()
        detail_transfer = []
        context_transfer = []
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for scale_j in self.scales:
            wavelength = 2 ** scale_j
            fine_sigma = wavelength / 2.0
            coarse_sigma = float(wavelength)
            context_sigma = max(2.0, 1.5 * wavelength)

            def transfer(sigma: float) -> torch.Tensor:
                return torch.exp(-2.0 * math.pi**2 * sigma**2 * radius2)

            detail_transfer.append(transfer(fine_sigma) - transfer(coarse_sigma))
            context_transfer.append(transfer(context_sigma))
            y, x, _, _, _ = _sample_grid(self.output_pixels, scale_j)
            self.samples.append(
                (
                    torch.as_tensor(y + self.guard_pixels, dtype=torch.long),
                    torch.as_tensor(x + self.guard_pixels, dtype=torch.long),
                )
            )
        self.detail_transfer = torch.stack(detail_transfer).to(torch.float32)
        self.context_transfer = torch.stack(context_transfer).to(torch.float32)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft2(image, dim=(-2, -1))
        values = []
        for index, (ys, xs) in enumerate(self.samples):
            detail = torch.fft.irfft2(
                transformed * self.detail_transfer[index][None],
                s=self.shape,
                dim=(-2, -1),
            )
            energy = torch.fft.irfft2(
                torch.fft.rfft2(detail.square(), dim=(-2, -1))
                * self.context_transfer[index][None],
                s=self.shape,
                dim=(-2, -1),
            )
            log_rms = 0.5 * torch.log(
                energy.clamp_min(torch.finfo(energy.dtype).tiny)
            )
            sampled = log_rms[:, ys[:, None], xs[None, :]]
            values.append(sampled.reshape(sampled.shape[0], -1))
        return torch.cat(values, dim=1)


def concatenate_modulation_targets(
    fits: tuple[LocalModulationScaleFit, ...],
    targets: dict[int, np.ndarray],
    uncertainties: dict[int, np.ndarray],
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, dict[int, slice]]:
    """Concatenate only robustly retained scales in descriptor order."""

    scales = tuple(item.scale_j for item in fits if item.accepted)
    if not scales:
        raise ValueError("No robust local-modulation scale survived preflight")
    target_rows = []
    uncertainty_rows = []
    slices: dict[int, slice] = {}
    start = 0
    for scale in scales:
        target = np.asarray(targets[scale], dtype=np.float32).ravel()
        uncertainty = np.asarray(uncertainties[scale], dtype=np.float32).ravel()
        stop = start + target.size
        slices[scale] = slice(start, stop)
        target_rows.append(target)
        uncertainty_rows.append(uncertainty)
        start = stop
    return (
        scales,
        np.concatenate(target_rows),
        np.concatenate(uncertainty_rows),
        slices,
    )
