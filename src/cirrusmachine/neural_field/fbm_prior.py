"""Anisotropic fractional-Brownian targets for contaminated fine scales."""

from __future__ import annotations

import math

import numpy as np


def measure_spectral_quadrupole(
    image: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    minimum_scale_arcsec: float,
    maximum_scale_arcsec: float,
    anisotropy_cap: float,
) -> tuple[float, float]:
    """Return spectral position angle and capped quadrupole amplitude."""

    data = np.asarray(image, dtype=np.float64)
    height, width = data.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    transformed = np.fft.fft2((data - np.mean(data)) * window)
    power = np.abs(transformed) ** 2
    fy = np.fft.fftfreq(height, d=pixel_scale_arcsec)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_arcsec)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    angle = np.arctan2(fy, fx)
    include = (
        (radius >= 1.0 / float(maximum_scale_arcsec))
        & (radius <= 1.0 / float(minimum_scale_arcsec))
        & (radius > 0)
    )
    selected = power[include]
    moment = np.sum(selected * np.exp(2j * angle[include])) / np.sum(selected)
    spectral_angle = 0.5 * np.angle(moment)
    quadrupole = min(float(np.abs(moment)), float(anisotropy_cap))
    return float(spectral_angle), quadrupole


def anisotropic_fbm(
    shape: tuple[int, int],
    *,
    slope: float,
    spectral_angle: float,
    quadrupole: float,
    seed: int,
) -> np.ndarray:
    """Generate a zero-mean, unit-RMS anisotropic fBm field."""

    rng = np.random.default_rng(int(seed))
    white = rng.normal(size=shape)
    transformed = np.fft.fft2(white)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    angle = np.arctan2(fy, fx)
    angular = 1.0 + 2.0 * float(quadrupole) * np.cos(
        2.0 * (angle - float(spectral_angle))
    )
    angular = np.clip(angular, 0.1, None)
    spectrum = np.zeros(shape, dtype=np.float64)
    nonzero = radius > 0
    spectrum[nonzero] = radius[nonzero] ** float(slope) * angular[nonzero]
    field = np.fft.ifft2(transformed * np.sqrt(spectrum)).real
    field -= np.mean(field)
    rms = math.sqrt(float(np.mean(field * field)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Invalid fBm normalization")
    return (field / rms).astype(np.float32)
