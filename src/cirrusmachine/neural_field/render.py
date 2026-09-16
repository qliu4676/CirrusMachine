"""Chunked neural-field rendering and WCS-preserving FITS output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.utils.checkpoint import checkpoint
from astropy.io import fits

from cirrusmachine.power_spectrum import power_spectrum_2d, radial_power_spectrum

from .model import MultiResolutionLatentField, coordinate_grid
from .operators import apodize_final_residual, compose_with_backbone


def phase_preserving_power_projection(
    synthesis: np.ndarray,
    backbone: np.ndarray,
    exemplar: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    fine_scale_arcmin: float,
    coarse_scale_arcmin: float,
    iterations: int,
    radial_bins: int = 48,
) -> np.ndarray:
    """Match a smooth Planck--Herschel radial-power target without changing phase.

    Complex Planck coefficients are installed exactly below the coarse edge.
    Above it, each synthesis coefficient is multiplied only by a positive real
    amplitude correction inferred from the same apodized diagnostic spectrum
    used for validation.  Thus neural phases are retained at texture scales.
    """

    output = np.asarray(synthesis, dtype=np.float64).copy()
    base = np.asarray(backbone, dtype=np.float64)
    texture = np.asarray(exemplar, dtype=np.float64)
    if output.shape != base.shape:
        raise ValueError("Synthesis and backbone must have identical shapes")
    if iterations <= 0:
        return output.astype(np.float32)
    pixel_scale_arcmin = float(pixel_scale_arcsec) / 60.0

    def spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return radial_power_spectrum(
            power_spectrum_2d(image), pixel_scale_arcmin, nbins=radial_bins
        )[:2]

    def prepared(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frequency, power = spectrum(image)
        keep = np.isfinite(frequency) & np.isfinite(power) & (frequency > 0) & (power > 0)
        return np.log(frequency[keep]), np.log(power[keep])

    def interpolate(curve: tuple[np.ndarray, np.ndarray], frequency: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(frequency), curve[0], curve[1]))

    planck_curve = prepared(base)
    herschel_curve = prepared(texture)
    height, width = output.shape
    fy = np.fft.fftfreq(height, d=pixel_scale_arcmin)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_arcmin)[None, :]
    radius = np.hypot(fy, fx)
    lower = 1.0 / float(coarse_scale_arcmin)
    upper = 1.0 / float(fine_scale_arcmin)
    texture_modes = radius >= lower
    frequencies = radius[texture_modes]
    phase = np.clip((frequencies - lower) / (upper - lower), 0.0, 1.0)
    alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
    target_power = (
        (1.0 - alpha) * interpolate(planck_curve, frequencies)
        + alpha * interpolate(herschel_curve, frequencies)
    )
    planck_fft = np.fft.fft2(base - np.mean(base))
    for _ in range(int(iterations)):
        current_curve = prepared(output)
        transformed = np.fft.fft2(output - np.mean(output))
        current_power = interpolate(current_curve, frequencies)
        transformed[texture_modes] *= np.sqrt(
            target_power / np.maximum(current_power, np.finfo(float).tiny)
        )
        transformed[~texture_modes] = planck_fft[~texture_modes]
        output = np.fft.ifft2(transformed).real + np.mean(base)
    return output.astype(np.float32)


def normalize_backbone_transition_power(
    backbone: np.ndarray,
    reference_planck: np.ndarray,
    reference_herschel: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    fine_scale_arcmin: float,
    coarse_scale_arcmin: float,
    anchor_scale_arcmin: float,
    maximum_gain: float,
    alpha: float = 0.1,
    beta: float = 0.9,
    radial_bins: int = 64,
) -> tuple[np.ndarray, dict[str, float]]:
    """Boost Planck fluctuation amplitudes smoothly into the Herschel transition band.

    The ratio is measured on a common, finite footprint.  The transform is real
    and positive, so it preserves every Planck Fourier phase and the image mean.
    """

    pixel_scale_arcmin = float(pixel_scale_arcsec) / 60.0
    valid = np.isfinite(reference_planck) & np.isfinite(reference_herschel)
    if valid.mean() < 0.9:
        raise ValueError("Power-normalization reference footprint is insufficiently valid")

    def prepared(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frequency, power = radial_power_spectrum(
            power_spectrum_2d(image, valid=valid, alpha=alpha, beta=beta),
            pixel_scale_arcmin,
            nbins=radial_bins,
        )[:2]
        keep = np.isfinite(frequency) & np.isfinite(power) & (frequency > 0) & (power > 0)
        return np.log(frequency[keep]), np.log(power[keep])

    planck_curve = prepared(reference_planck)
    herschel_curve = prepared(reference_herschel)
    lower = 1.0 / float(coarse_scale_arcmin)
    upper = 1.0 / float(fine_scale_arcmin)
    anchor = 1.0 / float(anchor_scale_arcmin)
    if not 0 < anchor < lower < upper:
        raise ValueError("Power-normalization scales must satisfy anchor > coarse > fine")
    nodes = np.geomspace(lower, upper, 96)
    log_ratio = np.interp(np.log(nodes), herschel_curve[0], herschel_curve[1]) - np.interp(
        np.log(nodes), planck_curve[0], planck_curve[1]
    )
    node_gain = np.clip(np.exp(0.5 * log_ratio), 1.0, float(maximum_gain))
    lower_gain = float(node_gain[0])
    upper_gain = float(node_gain[-1])
    height, width = backbone.shape
    fy = np.fft.fftfreq(height, d=pixel_scale_arcmin)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_arcmin)[None, :]
    radius = np.hypot(fy, fx)
    gain = np.ones_like(radius)
    ramp = np.clip((radius - anchor) / (lower - anchor), 0.0, 1.0)
    gain = 1.0 + (lower_gain - 1.0) * (0.5 - 0.5 * np.cos(np.pi * ramp))
    in_band = (radius >= lower) & (radius <= upper)
    gain[in_band] = np.interp(np.log(radius[in_band]), np.log(nodes), node_gain)
    gain[radius > upper] = upper_gain
    mean = float(np.mean(backbone))
    transformed = np.fft.fft2(np.asarray(backbone, dtype=np.float64) - mean)
    normalized = np.fft.ifft2(transformed * gain).real + mean
    median_gain = float(np.exp(0.5 * np.median(log_ratio)))
    return normalized.astype(np.float32), {
        "power_normalization_enabled": 1.0,
        "power_normalization_reference_valid_fraction": float(valid.mean()),
        "power_normalization_median_amplitude_gain": median_gain,
        "power_normalization_gain_10arcmin": lower_gain,
        "power_normalization_gain_3arcmin": upper_gain,
        "power_normalization_log_ratio_scatter_dex": float(
            np.std(log_ratio - np.median(log_ratio)) / np.log(10.0)
        ),
    }


def render_raw_field(
    model: MultiResolutionLatentField,
    shape: tuple[int, int],
    *,
    coordinate_chunk: int = 262144,
) -> torch.Tensor:
    """Evaluate one continuous field in chunks without detaching gradients."""

    device = next(model.parameters()).device
    coordinates = coordinate_grid(shape, device=device).reshape(-1, 2)
    pieces = []
    for start in range(0, coordinates.shape[0], int(coordinate_chunk)):
        chunk = coordinates[start : start + coordinate_chunk]
        if (
            torch.is_grad_enabled()
            and model.training
            and getattr(model, "fourier_feature_count", 0) > 0
        ):
            pieces.append(
                checkpoint(
                    model,
                    chunk.detach().requires_grad_(True),
                    use_reentrant=True,
                )
            )
        else:
            pieces.append(model(chunk))
    return torch.cat(pieces).reshape(shape)


def render_synthesis(
    model: MultiResolutionLatentField,
    backbone: torch.Tensor,
    *,
    transition_pixels: float,
    transition_width_fraction: float,
    transition_range_pixels: tuple[float, float] | None,
    apodization_fraction: float,
    boundary_padding_pixels: int,
    boundary_taper_fraction: float,
    complementary_backbone: bool,
    resolution_fwhm_pixels: float,
    coordinate_chunk: int,
    render_margin_pixels: int = 0,
    final_split_cosine_alpha: float = 0.0,
    final_split_cosine_beta: float = 0.9,
    final_apodization_mode: str = "radial",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    margin = int(render_margin_pixels)
    if margin < 0 or margin >= min(backbone.shape):
        raise ValueError("render_margin_pixels must be smaller than the output dimensions")
    if margin:
        expanded_backbone = torch_f.pad(
            backbone[None, None],
            (margin, margin, margin, margin),
            mode="reflect",
        )[0, 0]
    else:
        expanded_backbone = backbone
    raw = render_raw_field(
        model, tuple(expanded_backbone.shape), coordinate_chunk=coordinate_chunk
    )
    synthesis, projected = compose_with_backbone(
        expanded_backbone,
        raw,
        transition_pixels=transition_pixels,
        transition_width_fraction=transition_width_fraction,
        transition_range_pixels=transition_range_pixels,
        apodization_fraction=apodization_fraction,
        boundary_padding_pixels=boundary_padding_pixels,
        boundary_taper_fraction=boundary_taper_fraction,
        complementary_backbone=complementary_backbone,
        resolution_fwhm_pixels=resolution_fwhm_pixels,
    )
    expanded_synthesis = synthesis
    if margin:
        crop = (slice(margin, -margin), slice(margin, -margin))
        projected = projected[crop]
        raw = raw[crop]
    projected = apodize_final_residual(
        projected,
        alpha=final_split_cosine_alpha,
        beta=final_split_cosine_beta,
        mode=final_apodization_mode,
    )
    synthesis = backbone + projected
    return synthesis, projected, raw, expanded_synthesis


def write_synthesis_fits(
    path: str | Path,
    synthesis: np.ndarray,
    residual: np.ndarray,
    backbone: np.ndarray,
    header: fits.Header,
    *,
    metadata: dict[str, str | int | float | bool],
    overwrite: bool = False,
) -> Path:
    """Write the final image and provenance arrays with one shared WCS."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    primary_header = header.copy()
    primary_header.pop("EXTNAME", None)
    primary_header["BUNIT"] = "MJy/sr"
    for key, value in metadata.items():
        primary_header[key[:8].upper()] = value
    hdus = [
        fits.PrimaryHDU(np.asarray(synthesis, dtype=np.float32), header=primary_header),
        fits.ImageHDU(np.asarray(residual, dtype=np.float32), header=header, name="RESIDUAL"),
        fits.ImageHDU(np.asarray(backbone, dtype=np.float32), header=header, name="BACKBONE"),
    ]
    for hdu in hdus:
        hdu.header["BUNIT"] = "MJy/sr"
    fits.HDUList(hdus).writeto(output, overwrite=overwrite)
    return output
