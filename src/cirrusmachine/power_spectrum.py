"""Fourier power-spectrum diagnostics for finite-footprint cirrus images."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.stats import binned_statistic


def split_cosine_bell_window(
    shape: tuple[int, int], alpha: float = 0.1, beta: float = 0.5
) -> np.ndarray:
    """Return a TurbuStat-style radial split-cosine-bell window.

    ``beta`` sets the flat inner radius as a fraction of the half-size and
    ``alpha`` sets the fractional cosine-taper width beyond that radius.
    """
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("alpha and beta must lie between 0 and 1")
    y, x = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    radial_distance = np.hypot(y - cy, x - cx)
    npts = (min(shape) - 1.0) / 2.0
    inner_radius = beta * npts
    taper_width = int(np.floor(alpha * npts))
    if taper_width == 0:
        window = np.ones(shape, dtype=float)
    else:
        window = 0.5 * (
            1.0 + np.cos(np.pi * (radial_distance - inner_radius) / taper_width)
        )
    window[radial_distance < inner_radius] = 1.0
    window[radial_distance > inner_radius + taper_width] = 0.0
    return window


def power_spectrum_2d(
    image: np.ndarray,
    valid: np.ndarray | None = None,
    alpha: float = 0.1,
    beta: float = 0.5,
) -> np.ndarray:
    """Return a centered 2D power spectrum after robust centering/apodization.

    Invalid pixels are set to the sigma-clipped median before mean subtraction.
    A radial ``SplitCosineBellWindow`` reduces FFT-boundary leakage; using the
    same validity footprint permits direct before/after comparisons.
    """
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    valid = finite if valid is None else np.asarray(valid, dtype=bool) & finite
    _, median, _ = sigma_clipped_stats(image[valid], sigma=3.0, maxiters=10)
    centered = np.where(valid, image - median, 0.0)
    window = split_cosine_bell_window(image.shape, alpha=alpha, beta=beta)
    weighted = centered * window
    normalization = max(float(np.sum((window * valid) ** 2)), 1.0)
    return np.fft.fftshift(np.abs(np.fft.fft2(weighted)) ** 2 / normalization)


def radial_power_spectrum(
    power_2d: np.ndarray,
    pixel_scale_arcmin: float,
    nbins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally average a centered 2D spectrum in logarithmic frequency bins.

    This follows TurbuStat's ``pspec`` convention of taking the mean and
    standard deviation of the 2D power in radial spatial-frequency bins.
    Empty logarithmic bins are omitted.  They are common at the lowest Fourier
    frequencies of a finite image, where only a few discrete radial modes are
    available, and otherwise produce artificial gaps in log-log diagnostics.
    """
    power_2d = np.asarray(power_2d, dtype=float)
    fy = np.fft.fftshift(np.fft.fftfreq(power_2d.shape[0], d=pixel_scale_arcmin))
    fx = np.fft.fftshift(np.fft.fftfreq(power_2d.shape[1], d=pixel_scale_arcmin))
    radial_frequency = np.hypot(fy[:, None], fx[None, :])
    finite = np.isfinite(power_2d) & (radial_frequency > 0.0)
    sampled_frequency = radial_frequency[finite]
    sampled_power = power_2d[finite]
    if not sampled_frequency.size:
        raise ValueError("power_2d has no finite nonzero-frequency modes")
    edges = np.geomspace(
        sampled_frequency.min(), sampled_frequency.max(), int(nbins) + 1
    )
    # Make the final edge inclusive of the Nyquist-corner modes.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    mean = binned_statistic(
        sampled_frequency, sampled_power, bins=edges, statistic=np.nanmean
    ).statistic
    stddev = binned_statistic(
        sampled_frequency, sampled_power, bins=edges, statistic=np.nanstd
    ).statistic
    frequency = binned_statistic(
        sampled_frequency, sampled_frequency, bins=edges, statistic=np.nanmean
    ).statistic
    populated = np.isfinite(frequency) & np.isfinite(mean) & np.isfinite(stddev)
    return frequency[populated], mean[populated], stddev[populated]
