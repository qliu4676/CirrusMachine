"""Herschel beam matching, compact-source detection, and source filling."""

from __future__ import annotations

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_dilation, gaussian_filter


def valid_coverage_mask(image: np.ndarray, coverage: np.ndarray | None = None) -> np.ndarray:
    """Return finite pixels inside positive instrumental coverage."""
    valid = np.isfinite(image)
    if coverage is not None:
        valid &= np.isfinite(coverage) & (coverage > 0.0)
    return valid


def sigma_clipped_median(image: np.ndarray, valid: np.ndarray) -> float:
    """Measure a robust median from valid map pixels."""
    _, median, _ = sigma_clipped_stats(image[valid], sigma=3.0, maxiters=10)
    return float(median)


def smooth_to_fwhm(
    image: np.ndarray,
    pixel_scale_arcsec: float,
    target_fwhm_arcsec: float,
    native_fwhm_arcsec: float = 0.0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian beam-match an image, padding invalid/outside pixels by its median.

    The kernel width is the quadrature difference of target and native beams.
    Pixels outside coverage and array boundaries are represented by the
    sigma-clipped median rather than zero during convolution.
    """
    image = np.asarray(image, dtype=float)
    valid = np.isfinite(image) if valid is None else np.asarray(valid, dtype=bool)
    fill_value = sigma_clipped_median(image, valid)
    filled = np.where(valid, image, fill_value)
    kernel_fwhm = np.sqrt(max(target_fwhm_arcsec**2 - native_fwhm_arcsec**2, 0.0))
    sigma_pixels = kernel_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale_arcsec)
    if sigma_pixels == 0.0:
        return filled
    return gaussian_filter(filled, sigma_pixels, mode="constant", cval=fill_value)


def highpass_filter(
    image: np.ndarray,
    valid: np.ndarray,
    pixel_scale_arcsec: float,
    background_fwhm_arcsec: float,
) -> np.ndarray:
    """Subtract a Gaussian large-scale background using median edge padding."""
    image = np.asarray(image, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    fill_value = sigma_clipped_median(image, valid)
    filled = np.where(valid, image, fill_value)
    sigma_pixels = background_fwhm_arcsec / (
        2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale_arcsec
    )
    # Invalid coverage is already represented by the robust median. At the
    # rectangular array boundary, nearest extension avoids manufacturing a
    # sharp step between valid edge pixels and that median.
    lowpass = gaussian_filter(filled, sigma_pixels, mode="nearest")
    highpass = filled - lowpass
    highpass[~valid] = np.nan
    return highpass


def robust_affine_match(
    template: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 8,
) -> tuple[float, float]:
    """Fit ``target ~= slope * template + offset`` with residual clipping."""
    template = np.asarray(template, dtype=float)
    target = np.asarray(target, dtype=float)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(template) & np.isfinite(target)
    for _ in range(maxiters):
        x = template[keep]
        y = target[keep]
        x_center = np.mean(x)
        slope = float(np.sum((x - x_center) * (y - np.mean(y))) / np.sum((x - x_center) ** 2))
        offset = float(np.mean(y) - slope * x_center)
        residual = target - (slope * template + offset)
        _, center, scatter = sigma_clipped_stats(residual[keep], sigma=sigma, maxiters=3)
        updated = keep & (np.abs(residual - center) <= sigma * scatter)
        if np.array_equal(updated, keep):
            break
        keep = updated
    return slope, offset


def detect_compact_sources(
    residual: np.ndarray,
    valid: np.ndarray,
    pixel_scale_arcsec: float,
    beam_fwhm_arcsec: float | None = 18.0,
    threshold_sigma: float = 3.0,
    npixels: int = 5,
    dilation_pixels: int = 2,
    background_box_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect positive beam-scale residual sources with a local robust threshold.

    Returns the dilated source mask, segmentation labels, and detection image.
    """
    residual = np.asarray(residual, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    fill_value = sigma_clipped_median(residual, valid)
    prepared = np.where(valid, residual, fill_value)
    if beam_fwhm_arcsec is None or beam_fwhm_arcsec <= 0.0:
        detection_image = prepared
    else:
        beam_sigma_pixels = beam_fwhm_arcsec / (
            2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale_arcsec
        )
        kernel = Gaussian2DKernel(beam_sigma_pixels)
        detection_image = convolve(
            prepared,
            kernel,
            boundary="fill",
            fill_value=fill_value,
            normalize_kernel=True,
        )
    background = Background2D(
        detection_image,
        box_size=background_box_size,
        filter_size=3,
        mask=~valid,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
        bkg_estimator=SExtractorBackground(),
        bkg_rms_estimator=StdBackgroundRMS(),
        exclude_percentile=50.0,
    )
    threshold = background.background + threshold_sigma * background.background_rms
    segmentation = detect_sources(detection_image, threshold, n_pixels=npixels, mask=~valid)
    labels = np.zeros(residual.shape, dtype=np.int32)
    if segmentation is not None:
        labels = np.asarray(segmentation.data, dtype=np.int32)
    source_mask = labels > 0
    if dilation_pixels > 0:
        source_mask = binary_dilation(source_mask, iterations=int(dilation_pixels))
    source_mask &= valid
    return source_mask, labels, detection_image


def inpaint_sources_maskfill(
    image: np.ndarray,
    source_mask: np.ndarray,
    valid: np.ndarray,
    size: int = 9,
) -> np.ndarray:
    """Fill detected sources with maskfill's median operator and restore edges."""
    from maskfill import maskfill

    image = np.asarray(image, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    fill_value = sigma_clipped_median(image, valid)
    prepared = np.where(valid, image, fill_value)
    filled, _ = maskfill(
        prepared,
        np.asarray(source_mask, dtype=np.uint8),
        size=size,
        operator="median",
        smooth=True,
        verbose=False,
    )
    filled = np.asarray(filled, dtype=float)
    filled[~valid] = np.nan
    return filled


def add_gaussian_noise_to_mask(
    image: np.ndarray,
    source_mask: np.ndarray,
    valid: np.ndarray,
    seed: int = 4676,
) -> tuple[np.ndarray, float]:
    """Add zero-mean Gaussian noise only to filled source-mask pixels.

    The noise standard deviation is ``astropy.stats.mad_std`` measured from
    valid, unmasked pixels with ``ignore_nan=True``.
    """
    image = np.asarray(image, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    reference = valid & ~source_mask
    noise_sigma = float(mad_std(image[reference], ignore_nan=True))
    output = image.copy()
    target = valid & source_mask
    rng = np.random.default_rng(seed)
    output[target] += rng.normal(0.0, noise_sigma, int(target.sum()))
    return output, noise_sigma


def source_masked_highpass_mad_std(
    highpass: np.ndarray, source_mask: np.ndarray, valid: np.ndarray
) -> float:
    """Estimate instrumental noise from source-masked, cirrus-subtracted pixels."""

    highpass = np.asarray(highpass, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(highpass)
    reference = valid & ~source_mask
    if not np.any(reference):
        raise ValueError("No valid unmasked high-pass pixels for MAD_STD")
    return float(mad_std(highpass[reference], ignore_nan=True))
