"""Absolute, mean-preserving power calibration for a wide-field backbone."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .cirrus_scaling import normalized_gaussian_filter
from .power_spectrum import power_spectrum_2d, radial_power_spectrum


@dataclass(frozen=True)
class BackbonePowerCalibration:
    """Power-spectrum curves and scalar used to calibrate a backbone."""

    amplitude_gain: float
    crop_size_pixels: int
    period_min_arcmin: float
    period_max_arcmin: float
    frequency_arcmin_inv: np.ndarray
    planck_power: np.ndarray
    herschel_power: np.ndarray
    fit_mask: np.ndarray
    valid_fraction: float


def apply_contrast_gain(image: np.ndarray, gain: float) -> np.ndarray:
    """Scale map fluctuations while preserving its finite-pixel mean exactly."""

    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all() or gain <= 0.0 or not np.isfinite(gain):
        raise ValueError("image must be finite and gain must be finite and positive")
    mean = float(np.mean(image, dtype=np.float64))
    return (mean + float(gain) * (image - mean)).astype(np.float32)


def calibrate_backbone_power_scalar(
    backbone: np.ndarray,
    herschel: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    crop_size_deg: float = 0.5,
    planck_beam_fwhm_arcmin: float = 5.0,
    herschel_resolution_arcsec: float = 36.0,
    period_min_arcmin: float = 6.0,
    period_max_arcmin: float = 20.0,
    radial_bins: int = 64,
) -> tuple[np.ndarray, BackbonePowerCalibration]:
    """Match Planck fluctuation power to beam-matched central Herschel data.

    A single robust geometric-mean power ratio is fitted over angular periods
    safely larger than the Planck beam. The gain is applied only about the
    map mean, preserving absolute surface-brightness calibration.
    """

    backbone = np.asarray(backbone, dtype=np.float32)
    herschel = np.asarray(herschel, dtype=np.float32)
    if backbone.shape != herschel.shape or backbone.ndim != 2:
        raise ValueError("backbone and herschel must be same-shaped 2D images")
    if pixel_scale_arcsec <= 0.0 or crop_size_deg <= 0.0:
        raise ValueError("pixel scale and crop size must be positive")
    if not planck_beam_fwhm_arcmin < period_min_arcmin < period_max_arcmin:
        raise ValueError("period range must be larger than the Planck beam")
    if herschel_resolution_arcsec > planck_beam_fwhm_arcmin * 60.0:
        raise ValueError("Herschel resolution cannot exceed Planck beam")

    size = int(round(crop_size_deg * 3600.0 / pixel_scale_arcsec))
    if size < 8 or size > min(backbone.shape):
        raise ValueError("central calibration crop is outside the image")
    y0 = (backbone.shape[0] - size) // 2
    x0 = (backbone.shape[1] - size) // 2
    region = (slice(y0, y0 + size), slice(x0, x0 + size))
    planck_crop = backbone[region]
    herschel_crop = herschel[region]
    match_fwhm = np.sqrt(
        (planck_beam_fwhm_arcmin * 60.0) ** 2 - herschel_resolution_arcsec**2
    )
    herschel_matched, support = normalized_gaussian_filter(
        herschel_crop, match_fwhm / pixel_scale_arcsec
    )
    # The convolution is normalized by ``support``. Retain points with at
    # least half a beam supported; the split-cosine window subsequently
    # removes the crop perimeter from the Fourier estimate.
    valid = np.isfinite(planck_crop) & np.isfinite(herschel_matched) & (support >= 0.5)
    if float(valid.mean()) < 0.9:
        raise ValueError("central power-calibration crop has insufficient valid coverage")

    pixel_scale_arcmin = pixel_scale_arcsec / 60.0
    frequency, planck_power, _ = radial_power_spectrum(
        power_spectrum_2d(planck_crop, valid=valid), pixel_scale_arcmin, nbins=radial_bins
    )
    _, herschel_power, _ = radial_power_spectrum(
        power_spectrum_2d(herschel_matched, valid=valid),
        pixel_scale_arcmin,
        nbins=radial_bins,
    )
    fit = (
        np.isfinite(planck_power)
        & np.isfinite(herschel_power)
        & (planck_power > 0.0)
        & (herschel_power > 0.0)
        & (frequency >= 1.0 / period_max_arcmin)
        & (frequency <= 1.0 / period_min_arcmin)
    )
    if int(fit.sum()) < 3:
        raise ValueError("fewer than three usable radial bins for power calibration")
    gain = float(np.exp(0.5 * np.median(np.log(herschel_power[fit] / planck_power[fit]))))
    result = BackbonePowerCalibration(
        amplitude_gain=gain,
        crop_size_pixels=size,
        period_min_arcmin=float(period_min_arcmin),
        period_max_arcmin=float(period_max_arcmin),
        frequency_arcmin_inv=frequency,
        planck_power=planck_power,
        herschel_power=herschel_power,
        fit_mask=fit,
        valid_fraction=float(valid.mean()),
    )
    return apply_contrast_gain(backbone, gain), result


def plot_backbone_power_calibration(result: BackbonePowerCalibration, path: str) -> None:
    """Save the beam-matched power curves and fitted scalar normalization."""

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    frequency = result.frequency_arcmin_inv
    ax.loglog(frequency, result.planck_power, label="Planck backbone, raw")
    ax.loglog(
        frequency,
        result.planck_power * result.amplitude_gain**2,
        label=f"Planck backbone × {result.amplitude_gain:.3f}²",
    )
    ax.loglog(frequency, result.herschel_power, label="Herschel, beam-matched")
    ax.axvspan(
        1.0 / result.period_max_arcmin,
        1.0 / result.period_min_arcmin,
        color="0.85",
        label="scalar-fit range",
    )
    ax.set(xlabel="spatial frequency [arcmin$^{-1}$]", ylabel="apodized power")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
