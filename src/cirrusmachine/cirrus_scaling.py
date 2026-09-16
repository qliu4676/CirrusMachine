"""Scale- and intensity-dependent cirrus fluctuation measurements.

The Miville-Deschenes et al. (2007; M07) prescription is implemented in
surface-brightness units.  Local empirical measurements use a one-octave
difference of Gaussians at the requested angular scale and a robust MAD
standard deviation inside larger contexts.  This convention is explicit:
its absolute normalization is not identical to the 12.5 degree map standard
deviation used to calibrate M07, although its intensity and scale trends can
be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import gaussian_filter
from scipy.stats import theilslopes


M07_REFERENCE_SCALE_DEG = 12.5
M07_BREAK_I100_MJY_SR = 10.0


@dataclass(frozen=True)
class LocalCirrusSamples:
    """Local intensity, color, and fluctuation measurements on a fixed grid."""

    x: np.ndarray
    y: np.ndarray
    intensity_250: np.ndarray
    sigma_250: np.ndarray
    sigma_250_std: np.ndarray
    i100_over_i250: np.ndarray
    valid_fraction: np.ndarray
    scale_arcmin: float
    context_arcmin: float
    spacing_arcmin: float

    @property
    def intensity_100(self) -> np.ndarray:
        """Local 100 micron intensity inferred from the Planck dust color."""

        return self.intensity_250 * self.i100_over_i250

    @property
    def sigma_100(self) -> np.ndarray:
        """Local 100 micron RMS under a locally constant dust color."""

        return self.sigma_250 * self.i100_over_i250


def m07_spectral_index(intensity_100_mjy_sr: np.ndarray | float) -> np.ndarray:
    """Return the M07 Eq. 4 power-spectrum index ``gamma(I_100)``."""

    intensity = np.asarray(intensity_100_mjy_sr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = -0.26 * np.log10(intensity) - 2.77
    return np.where(intensity > 0.0, gamma, np.nan)


def m07_variance_100_reference(
    intensity_100_mjy_sr: np.ndarray | float,
) -> np.ndarray:
    """Evaluate M07 Eqs. 7--8 at their 12.5 degree reference scale."""

    intensity = np.asarray(intensity_100_mjy_sr, dtype=float)
    low = 0.12 * intensity**2.0
    high = 0.011 * intensity**3.1
    variance = np.where(intensity < M07_BREAK_I100_MJY_SR, low, high)
    return np.where(intensity > 0.0, variance, np.nan)


def m07_sigma_lambda(
    intensity_100_mjy_sr: np.ndarray | float,
    wavelength_over_100_color: np.ndarray | float,
    scale_deg: float,
) -> np.ndarray:
    """Evaluate M07 Eqs. 17--18 in MJy/sr.

    Parameters
    ----------
    intensity_100_mjy_sr
        Local mean Galactic intensity at 100 micron after background removal.
    wavelength_over_100_color
        Modified-blackbody color ``I_lambda / I_100``.
    scale_deg
        Angular scale ``l`` in degrees.  M07 calibrated the scaling between
        roughly 5 arcmin and 12.5 degrees.
    """

    intensity = np.asarray(intensity_100_mjy_sr, dtype=float)
    color = np.asarray(wavelength_over_100_color, dtype=float)
    gamma = m07_spectral_index(intensity)
    scale_factor = (float(scale_deg) / M07_REFERENCE_SCALE_DEG) ** (
        -gamma / 2.0 - 1.0
    )
    low = 0.35 * intensity * color * scale_factor
    high = 0.10 * intensity**1.5 * color * scale_factor
    result = np.where(intensity < M07_BREAK_I100_MJY_SR, low, high)
    valid = (intensity > 0.0) & (color > 0.0) & (scale_deg > 0.0)
    return np.where(valid, result, np.nan)


def normalized_gaussian_filter(
    image: np.ndarray,
    fwhm_pixels: float,
    *,
    valid: np.ndarray | None = None,
    min_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smooth finite data without treating missing coverage as zero.

    The numerator and validity weights are convolved separately.  The returned
    weight map is the local fraction of the Gaussian kernel supported by valid
    pixels and can be used to reject footprint boundaries.
    """

    image = np.asarray(image, dtype=float)
    valid_mask = np.isfinite(image) if valid is None else np.asarray(valid, dtype=bool)
    valid_mask &= np.isfinite(image)
    sigma_pixels = float(fwhm_pixels) / 2.354820045
    if sigma_pixels <= 0.0:
        return np.where(valid_mask, image, np.nan), valid_mask.astype(float)
    numerator = gaussian_filter(
        np.where(valid_mask, image, 0.0), sigma_pixels, mode="constant", cval=0.0
    )
    weight = gaussian_filter(
        valid_mask.astype(float), sigma_pixels, mode="constant", cval=0.0
    )
    smoothed = np.divide(
        numerator,
        weight,
        out=np.full(image.shape, np.nan, dtype=float),
        where=weight >= float(min_weight),
    )
    return smoothed, weight


def highpass_at_scale(
    image: np.ndarray,
    pixel_scale_arcsec: float,
    scale_arcmin: float,
    *,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``image - Gaussian_FWHM(scale)`` with normalized mask handling."""

    image = np.asarray(image, dtype=float)
    valid_mask = np.isfinite(image) if valid is None else np.asarray(valid, dtype=bool)
    valid_mask &= np.isfinite(image)
    fwhm_pixels = float(scale_arcmin) * 60.0 / float(pixel_scale_arcsec)
    lowpass, weight = normalized_gaussian_filter(image, fwhm_pixels, valid=valid_mask)
    highpass = image - lowpass
    highpass[~valid_mask] = np.nan
    return highpass, lowpass, weight


def difference_of_gaussians_at_scale(
    image: np.ndarray,
    pixel_scale_arcsec: float,
    scale_arcmin: float,
    *,
    valid: np.ndarray | None = None,
    octave_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a one-octave Gaussian detail map centered near ``scale_arcmin``.

    The detail is the map smoothed at ``scale / octave_ratio`` minus the map
    smoothed at ``scale``.  This suppresses both the large-scale background and
    beam-scale residual noise while retaining fluctuations around the chosen
    characteristic scale.
    """

    image = np.asarray(image, dtype=float)
    valid_mask = np.isfinite(image) if valid is None else np.asarray(valid, dtype=bool)
    valid_mask &= np.isfinite(image)
    if octave_ratio <= 1.0:
        raise ValueError("octave_ratio must be greater than one")
    coarse_fwhm = float(scale_arcmin) * 60.0 / float(pixel_scale_arcsec)
    fine_fwhm = coarse_fwhm / float(octave_ratio)
    fine, fine_weight = normalized_gaussian_filter(image, fine_fwhm, valid=valid_mask)
    coarse, coarse_weight = normalized_gaussian_filter(image, coarse_fwhm, valid=valid_mask)
    detail = fine - coarse
    detail[~valid_mask] = np.nan
    return detail, coarse, np.minimum(fine_weight, coarse_weight)


def _clipped_mean(values: np.ndarray) -> float:
    clipped = sigma_clip(values, sigma=3.0, maxiters=5, masked=True)
    return float(np.ma.mean(clipped))


def sample_local_cirrus_statistics(
    image_250: np.ndarray,
    i100_over_i250: np.ndarray,
    pixel_scale_arcsec: float,
    *,
    scale_arcmin: float = 5.0,
    context_factor: float = 3.0,
    spacing_arcmin: float | None = None,
    min_valid_fraction: float = 0.90,
) -> tuple[LocalCirrusSamples, np.ndarray, np.ndarray]:
    """Measure local robust RMS at one characteristic angular scale.

    ``sigma_250`` is ``astropy.stats.mad_std`` of the scale high-pass inside a
    square context.  ``mad_std`` already includes the Gaussian-consistency
    factor 1.4826.  Contexts overlap when ``spacing_arcmin < context_arcmin``;
    this is useful for visualizing the relation but does not increase the
    number of independent scale elements.
    """

    image = np.asarray(image_250, dtype=float)
    color = np.asarray(i100_over_i250, dtype=float)
    if image.shape != color.shape:
        raise ValueError("image_250 and i100_over_i250 must have the same shape")
    if scale_arcmin <= 0.0 or context_factor <= 1.0:
        raise ValueError("scale_arcmin must be positive and context_factor > 1")
    spacing = float(scale_arcmin / 2.0 if spacing_arcmin is None else spacing_arcmin)
    context_arcmin = float(context_factor * scale_arcmin)
    valid = np.isfinite(image) & np.isfinite(color) & (color > 0.0)
    highpass, lowpass, _ = difference_of_gaussians_at_scale(
        image, pixel_scale_arcsec, scale_arcmin, valid=valid
    )

    half = max(int(round(context_arcmin * 60.0 / pixel_scale_arcsec / 2.0)), 1)
    stride = max(int(round(spacing * 60.0 / pixel_scale_arcsec)), 1)
    rows: list[tuple[float, ...]] = []
    for y in range(half, image.shape[0] - half + 1, stride):
        for x in range(half, image.shape[1] - half + 1, stride):
            ys = slice(y - half, y + half + 1)
            xs = slice(x - half, x + half + 1)
            patch_valid = valid[ys, xs] & np.isfinite(highpass[ys, xs])
            fraction = float(patch_valid.mean())
            if fraction < min_valid_fraction or patch_valid.sum() < 16:
                continue
            hp = highpass[ys, xs][patch_valid]
            local_color = color[ys, xs][patch_valid]
            local_image = image[ys, xs][patch_valid]
            rows.append(
                (
                    float(x),
                    float(y),
                    _clipped_mean(local_image),
                    float(mad_std(hp, ignore_nan=True)),
                    float(np.std(sigma_clip(hp, sigma=3.0, maxiters=5))),
                    float(np.nanmedian(local_color)),
                    fraction,
                )
            )
    if not rows:
        raise ValueError("No local contexts satisfy the requested coverage threshold")
    columns = np.asarray(rows, dtype=float).T
    samples = LocalCirrusSamples(
        x=columns[0],
        y=columns[1],
        intensity_250=columns[2],
        sigma_250=columns[3],
        sigma_250_std=columns[4],
        i100_over_i250=columns[5],
        valid_fraction=columns[6],
        scale_arcmin=float(scale_arcmin),
        context_arcmin=context_arcmin,
        spacing_arcmin=spacing,
    )
    return samples, highpass, lowpass


def robust_log_relation(
    intensity: np.ndarray, sigma: np.ndarray, *, reference_intensity: float = 1.0
) -> tuple[float, float, float, float]:
    """Fit ``sigma = amplitude * (I / I_ref)**slope`` with Theil-Sen."""

    intensity = np.asarray(intensity, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    keep = np.isfinite(intensity) & np.isfinite(sigma) & (intensity > 0.0) & (sigma > 0.0)
    if keep.sum() < 3:
        raise ValueError("At least three positive finite samples are required")
    x = np.log10(intensity[keep] / float(reference_intensity))
    y = np.log10(sigma[keep])
    slope, intercept, low_slope, high_slope = theilslopes(y, x, alpha=0.95)
    return float(10.0**intercept), float(slope), float(low_slope), float(high_slope)


def binned_relation(
    intensity: np.ndarray, sigma: np.ndarray, *, bins: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-population median relation and 16--84 percentile ranges."""

    intensity = np.asarray(intensity, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    keep = np.isfinite(intensity) & np.isfinite(sigma) & (intensity > 0.0) & (sigma > 0.0)
    order = np.argsort(intensity[keep])
    x, y = intensity[keep][order], sigma[keep][order]
    chunks = [chunk for chunk in np.array_split(np.arange(x.size), int(bins)) if chunk.size]
    return (
        np.asarray([np.median(x[c]) for c in chunks]),
        np.asarray([np.median(y[c]) for c in chunks]),
        np.asarray([np.percentile(y[c], 16.0) for c in chunks]),
        np.asarray([np.percentile(y[c], 84.0) for c in chunks]),
    )
