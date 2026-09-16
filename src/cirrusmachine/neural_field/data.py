"""A2390 loading, WCS-preserving crops, masks, and patch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from cirrusmachine.planck_dust import modified_blackbody_band_to_I100

from .config import ExperimentConfig


@dataclass(frozen=True)
class NeuralFieldData:
    backbone: np.ndarray
    output_header: fits.Header
    exemplar: np.ndarray
    exemplar_header: fits.Header
    exemplar_valid: np.ndarray
    exemplar_training: np.ndarray
    exemplar_holdout: np.ndarray
    texture_exemplars: tuple[np.ndarray, ...]
    texture_training: tuple[np.ndarray, ...]
    texture_holdout: tuple[np.ndarray, ...]
    exemplar_planck: np.ndarray
    exemplar_band_to_i100: np.ndarray
    dust_temperature: np.ndarray
    dust_beta: np.ndarray
    pixel_scale_arcsec: float
    radiance_slope: float
    radiance_offset: float


def centered_square_slices(shape: tuple[int, int], size: int) -> tuple[slice, slice]:
    """Return deterministic centered square slices."""

    if size > min(shape):
        raise ValueError(f"Square size {size} exceeds shape {shape}")
    y0 = (shape[0] - size) // 2
    x0 = (shape[1] - size) // 2
    return slice(y0, y0 + size), slice(x0, x0 + size)


def sliced_header(header: fits.Header, slices: tuple[slice, slice]) -> fits.Header:
    """Copy metadata while replacing celestial WCS with a sliced WCS."""

    result = header.copy()
    result.update(WCS(header).celestial.slice(slices).to_header(relax=True))
    result["NAXIS1"] = slices[1].stop - slices[1].start
    result["NAXIS2"] = slices[0].stop - slices[0].start
    return result


def holdout_masks(
    shape: tuple[int, int], origin: tuple[int, int], size: int, buffer_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out pixels and the larger region forbidden during training."""

    y0, x0 = map(int, origin)
    if size < 1 or y0 < 0 or x0 < 0 or y0 + size > shape[0] or x0 + size > shape[1]:
        raise ValueError("The configured holdout lies outside the exemplar")
    holdout = np.zeros(shape, dtype=bool)
    holdout[y0 : y0 + size, x0 : x0 + size] = True
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[
        max(0, y0 - buffer_pixels) : min(shape[0], y0 + size + buffer_pixels),
        max(0, x0 - buffer_pixels) : min(shape[1], x0 + size + buffer_pixels),
    ] = True
    return holdout, forbidden


def _reproject(array: np.ndarray, header: fits.Header, target: fits.Header, shape: tuple[int, int]) -> np.ndarray:
    projected, footprint = reproject_interp(
        (np.asarray(array, dtype=float), WCS(header).celestial),
        WCS(target).celestial,
        shape_out=shape,
        order="bilinear",
    )
    projected[footprint <= 0.0] = np.nan
    return projected


def load_a2390(config: ExperimentConfig) -> NeuralFieldData:
    """Load the approved affine Planck backbone and Herschel cutout 4."""

    inputs = config.inputs
    with fits.open(Path(inputs.products), memmap=True) as hdul:
        radiance_hdu = hdul[inputs.planck_hdu]
        radiance = np.asarray(radiance_hdu.data, dtype=np.float32)
        full_header = radiance_hdu.header.copy()
        primary = hdul[0].header
        slope = float(
            inputs.radiance_slope
            if inputs.radiance_slope is not None
            else primary["RADSLOPE"]
        )
        offset = float(
            inputs.radiance_offset
            if inputs.radiance_offset is not None
            else primary["RADOFF"]
        )
        temperature_full = np.asarray(hdul[inputs.temperature_hdu].data, dtype=np.float32)
        beta_full = np.asarray(hdul[inputs.beta_hdu].data, dtype=np.float32)
    slices = centered_square_slices(radiance.shape, inputs.target_square_pixels)
    backbone = slope * radiance[slices] + offset
    output_header = sliced_header(full_header, slices)
    temperature = temperature_full[slices]
    beta = beta_full[slices]

    with fits.open(Path(inputs.herschel_cutout), memmap=True) as hdul:
        exemplar = np.asarray(hdul[0].data, dtype=np.float32)
        exemplar_header = hdul[0].header.copy()
    valid = np.isfinite(exemplar)
    holdout, forbidden = holdout_masks(
        exemplar.shape,
        config.split.holdout_origin,
        config.split.holdout_size,
        config.split.buffer_pixels,
    )
    training = valid & ~forbidden
    holdout &= valid
    texture_paths = inputs.herschel_cutouts or (inputs.herschel_cutout,)
    texture_exemplars = []
    texture_training = []
    texture_holdout = []
    for path in texture_paths:
        with fits.open(Path(path), memmap=True) as hdul:
            texture = np.asarray(hdul[0].data, dtype=np.float32)
        texture_valid = np.isfinite(texture)
        texture_hold, texture_forbidden = holdout_masks(
            texture.shape,
            config.split.holdout_origin,
            config.split.holdout_size,
            config.split.buffer_pixels,
        )
        texture_hold &= texture_valid
        texture_mask = texture_valid & ~texture_forbidden
        texture_exemplars.append(texture)
        texture_training.append(texture_mask)
        texture_holdout.append(texture_hold)
    exemplar_planck = _reproject(
        slope * radiance + offset,
        full_header,
        exemplar_header,
        exemplar.shape,
    ).astype(np.float32)
    conversion_full = modified_blackbody_band_to_I100(
        temperature_full, beta_full, 250.0
    )
    exemplar_conversion = _reproject(
        conversion_full,
        full_header,
        exemplar_header,
        exemplar.shape,
    ).astype(np.float32)
    pixel_scale = float(np.mean(proj_plane_pixel_scales(WCS(output_header).celestial)) * 3600.0)
    if not np.isclose(pixel_scale, inputs.pixel_scale_arcsec, rtol=0.01):
        raise ValueError(
            f"Configured pixel scale {inputs.pixel_scale_arcsec} differs from WCS {pixel_scale}"
        )
    for name, value in {
        "backbone": backbone,
        "temperature": temperature,
        "beta": beta,
        "exemplar_planck": exemplar_planck,
        "exemplar_conversion": exemplar_conversion,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains invalid pixels in the selected footprint")
    return NeuralFieldData(
        backbone=backbone.astype(np.float32),
        output_header=output_header,
        exemplar=exemplar,
        exemplar_header=exemplar_header,
        exemplar_valid=valid,
        exemplar_training=training,
        exemplar_holdout=holdout,
        texture_exemplars=tuple(texture_exemplars),
        texture_training=tuple(texture_training),
        texture_holdout=tuple(texture_holdout),
        exemplar_planck=exemplar_planck,
        exemplar_band_to_i100=exemplar_conversion,
        dust_temperature=temperature.astype(np.float32),
        dust_beta=beta.astype(np.float32),
        pixel_scale_arcsec=pixel_scale,
        radiance_slope=slope,
        radiance_offset=offset,
    )


def valid_patch_origins(
    mask: np.ndarray, patch_size: int, minimum_valid_fraction: float = 1.0
) -> np.ndarray:
    """Return all top-left coordinates satisfying a mask-fraction threshold."""

    valid = np.asarray(mask, dtype=np.int32)
    size = int(patch_size)
    if size > min(valid.shape):
        raise ValueError("patch_size exceeds the image")
    integral = np.pad(valid, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y = np.arange(valid.shape[0] - size + 1)
    x = np.arange(valid.shape[1] - size + 1)
    counts = (
        integral[y[:, None] + size, x[None, :] + size]
        - integral[y[:, None], x[None, :] + size]
        - integral[y[:, None] + size, x[None, :]]
        + integral[y[:, None], x[None, :]]
    )
    keep = counts >= float(minimum_valid_fraction) * size * size
    return np.argwhere(keep).astype(np.int64)


def sample_origins(
    origins: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample reproducibly, with replacement only when necessary."""

    if len(origins) == 0:
        raise ValueError("No valid patch origins are available")
    indices = rng.choice(len(origins), size=int(count), replace=len(origins) < count)
    return np.asarray(origins[indices], dtype=np.int64)
