"""Generic single-image neural-field texture synthesis with Morlet SWD."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import random
import time
from typing import Any, TypeVar, get_type_hints

import numpy as np
import torch
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.ndimage import gaussian_filter

from cirrusmachine.scattering_mean import ScatteringMean2D

from .features import (
    OrientedMorletFeatures,
    RobustNormalizer,
    correlation_floor_weights,
    log_aligned_cross_scale_coupling,
    log_cross_scale_envelope_coupling,
    log_oriented_wavelet_energy,
    log_wavelet_energy_by_scale,
    radial_log_power_physical,
    subsample_rows,
)
from .fbm_prior import anisotropic_fbm, measure_spectral_quadrupole
from .losses import sliced_wasserstein_loss
from .model import MultiResolutionLatentField
from .operators import gaussian_resolution_transfer
from .render import render_raw_field


@dataclass(frozen=True)
class SingleImageInputConfig:
    path: str
    hdu: int | str = 0
    pixel_scale_arcsec: float | None = None
    label: str = "Input image"
    block_average_factor: int = 1
    pre_bin_gaussian_fwhm_arcsec: float | None = None
    texture_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SingleImageBackboneConfig:
    """Optional fixed output backbone on a footprint larger than the texture input."""

    path: str | None = None
    hdu: int | str = 0
    affine_from_primary: bool = False
    value_scale: float = 1.0
    value_offset: float = 0.0


@dataclass(frozen=True)
class SingleImageGeometryConfig:
    crop_size: int = 150
    canvas_size: int = 300


@dataclass(frozen=True)
class SingleImageMorletConfig:
    training_scales: tuple[int, ...] = (2, 3, 4, 5)
    diagnostic_scales: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    orientations: int = 4
    boundary_sigma: float = 2.0
    target_padding_pixels: int = 75
    output_samples: int = 4096
    projections: int = 64
    quantiles: int = 128
    projection_chunk: int = 4096
    output_sampling_region: str = "crop"
    safe_canvas_radius: float = 0.8


@dataclass(frozen=True)
class SingleImageArchitectureConfig:
    grid_resolutions: tuple[int, ...] = (16, 32, 64, 128)
    channels_per_level: int = 4
    hidden_features: int = 64
    hidden_layers: int = 2
    output_scale: float = 1.0
    include_coordinates: bool = False
    rotate_and_offset_grids: bool = False
    grid_frame_seed: int | None = None
    fine_branch: str = "latent_grid"
    fourier_feature_count: int = 0
    fourier_min_scale_arcsec: float | None = None
    fourier_max_scale_arcsec: float | None = None
    fourier_seed: int | None = None


@dataclass(frozen=True)
class SingleImageDecompositionConfig:
    enabled: bool = False
    gaussian_fwhm_pixels: float | None = None


@dataclass(frozen=True)
class SingleImageInitializationConfig:
    path: str | None = None
    hdu: int | str = 0
    canvas_extension: str = "reflect"
    value_scale: float = 1.0
    value_offset: float = 0.0
    prefit_steps: int = 300
    learning_rate: float = 0.003


@dataclass(frozen=True)
class SingleImageOptimizerConfig:
    steps: int = 300
    learning_rate: float = 0.003
    gradient_clip: float = 5.0
    mixed_precision: bool = True


@dataclass(frozen=True)
class SingleImagePowerConfig:
    """Soft residual-spectrum constraint over a physically trusted scale band.

    ``minimum_scale_arcsec`` is the smallest trustworthy spatial scale; modes
    at finer scales are deliberately excluded.  ``maximum_scale_arcsec`` is
    the largest scale assigned to the neural residual (normally the fixed
    backbone split).  A null maximum includes every available larger scale.
    """

    enabled: bool = False
    weight: float = 0.0
    bins: int = 8
    minimum_scale_arcsec: float = 0.0
    maximum_scale_arcsec: float | None = None
    huber_delta: float = 1.0


@dataclass(frozen=True)
class SingleImageWaveletPowerConfig:
    """Absolute orientation-resolved Morlet band-energy constraint."""

    enabled: bool = False
    weight: float = 0.0
    huber_delta: float = 1.0


@dataclass(frozen=True)
class SingleImageCrossScaleConfig:
    """Normalized fine/coarse Morlet-energy concentration constraint."""

    enabled: bool = False
    weight: float = 0.0
    fine_scales: tuple[int, ...] = ()
    coarse_scales: tuple[int, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()
    aligned_orientations: bool = False
    correlation_weight_eta: float = 0.0
    maximum_uniform_multiple: float = 4.0
    huber_delta: float = 1.0


@dataclass(frozen=True)
class SingleImageFbmPriorConfig:
    """Anisotropic fBm replacement for contaminated Morlet scales."""

    enabled: bool = False
    replaced_scales: tuple[int, ...] = ()
    trusted_scales: tuple[int, ...] = ()
    realizations: int = 4
    slope: float = -3.0
    anisotropy_cap: float = 0.25
    seed_offset: int = 90000


@dataclass(frozen=True)
class SingleImageRuntimeConfig:
    device: str = "cpu"
    seed: int = 4676
    coordinate_chunk: int = 262144
    checkpoint_interval: int = 100
    diagnostic_interval: int = 100
    validation_interval: int = 5
    deterministic: bool = True


@dataclass(frozen=True)
class SingleImageOutputConfig:
    root: str
    run_name: str = "single_image_morlet"
    overwrite: bool = False


@dataclass(frozen=True)
class SingleImageConfig:
    input: SingleImageInputConfig
    output: SingleImageOutputConfig
    geometry: SingleImageGeometryConfig = field(default_factory=SingleImageGeometryConfig)
    backbone: SingleImageBackboneConfig = field(default_factory=SingleImageBackboneConfig)
    morlet: SingleImageMorletConfig = field(default_factory=SingleImageMorletConfig)
    architecture: SingleImageArchitectureConfig = field(
        default_factory=SingleImageArchitectureConfig
    )
    decomposition: SingleImageDecompositionConfig = field(
        default_factory=SingleImageDecompositionConfig
    )
    initialization: SingleImageInitializationConfig = field(
        default_factory=SingleImageInitializationConfig
    )
    power: SingleImagePowerConfig = field(default_factory=SingleImagePowerConfig)
    wavelet_power: SingleImageWaveletPowerConfig = field(
        default_factory=SingleImageWaveletPowerConfig
    )
    cross_scale: SingleImageCrossScaleConfig = field(
        default_factory=SingleImageCrossScaleConfig
    )
    fbm_prior: SingleImageFbmPriorConfig = field(
        default_factory=SingleImageFbmPriorConfig
    )
    optimizer: SingleImageOptimizerConfig = field(
        default_factory=SingleImageOptimizerConfig
    )
    runtime: SingleImageRuntimeConfig = field(default_factory=SingleImageRuntimeConfig)

    def validate(self) -> None:
        geometry = self.geometry
        if geometry.crop_size < 16:
            raise ValueError("geometry.crop_size must be at least 16")
        if geometry.canvas_size <= geometry.crop_size:
            raise ValueError("geometry.canvas_size must exceed crop_size")
        if (geometry.canvas_size - geometry.crop_size) % 2:
            raise ValueError("Canvas and crop sizes must have the same parity")
        morlet = self.morlet
        if self.input.block_average_factor < 1:
            raise ValueError("input.block_average_factor must be positive")
        if (
            self.input.pre_bin_gaussian_fwhm_arcsec is not None
            and self.input.pre_bin_gaussian_fwhm_arcsec <= 0
        ):
            raise ValueError("input.pre_bin_gaussian_fwhm_arcsec must be positive")
        if self.backbone.path is None and self.backbone.affine_from_primary:
            raise ValueError("backbone.affine_from_primary requires backbone.path")
        if not morlet.training_scales:
            raise ValueError("At least one training Morlet scale is required")
        if min(morlet.training_scales) < 0:
            raise ValueError("Morlet scales cannot be negative")
        if not set(morlet.training_scales).issubset(morlet.diagnostic_scales):
            raise ValueError("Diagnostic scales must include every training scale")
        if morlet.orientations < 1:
            raise ValueError("At least one Morlet orientation is required")
        if morlet.target_padding_pixels >= geometry.crop_size:
            raise ValueError("Target reflection padding must be smaller than the input")
        valid = morlet_valid_center_mask(
            (geometry.crop_size, geometry.crop_size),
            morlet.training_scales,
            morlet.boundary_sigma,
        )
        if not np.any(valid):
            raise ValueError("Morlet boundary exclusion leaves no target responses")
        if morlet.output_samples < 1 or morlet.projections < 1 or morlet.quantiles < 2:
            raise ValueError("Morlet SWD sampling parameters must be positive")
        if morlet.output_sampling_region not in {"crop", "safe_canvas"}:
            raise ValueError(
                "morlet.output_sampling_region must be 'crop' or 'safe_canvas'"
            )
        if not 0 < morlet.safe_canvas_radius < 1:
            raise ValueError("morlet.safe_canvas_radius must lie in (0, 1)")
        architecture = self.architecture
        if architecture.fine_branch not in {"latent_grid", "fourier"}:
            raise ValueError(
                "architecture.fine_branch must be 'latent_grid' or 'fourier'"
            )
        if architecture.fine_branch == "fourier":
            if architecture.fourier_feature_count < 1:
                raise ValueError("The Fourier branch requires Fourier features")
            if (
                architecture.fourier_min_scale_arcsec is None
                or architecture.fourier_max_scale_arcsec is None
                or architecture.fourier_min_scale_arcsec <= 0
                or architecture.fourier_max_scale_arcsec
                <= architecture.fourier_min_scale_arcsec
            ):
                raise ValueError(
                    "Fourier physical scales must satisfy 0 < min < max"
                )
        if (
            self.decomposition.gaussian_fwhm_pixels is not None
            and self.decomposition.gaussian_fwhm_pixels <= 0
        ):
            raise ValueError("decomposition.gaussian_fwhm_pixels must be positive")
        initialization = self.initialization
        if initialization.path is not None:
            if initialization.canvas_extension not in {"reflect", "reproject"}:
                raise ValueError(
                    "initialization.canvas_extension must be 'reflect' or 'reproject'"
                )
            if initialization.prefit_steps < 1 or initialization.learning_rate <= 0:
                raise ValueError(
                    "Initialization prefit steps and learning rate must be positive"
                )
        if self.optimizer.steps < 1 or self.optimizer.learning_rate <= 0:
            raise ValueError("Optimizer steps and learning rate must be positive")
        if self.optimizer.gradient_clip < 0:
            raise ValueError("Gradient clipping cannot be negative")
        if self.power.enabled:
            if self.power.weight <= 0:
                raise ValueError("power.weight must be positive when power is enabled")
            if self.power.bins < 2 or self.power.minimum_scale_arcsec <= 0:
                raise ValueError("Power loss needs at least two bins and a positive minimum scale")
            if (
                self.power.maximum_scale_arcsec is not None
                and self.power.maximum_scale_arcsec <= self.power.minimum_scale_arcsec
            ):
                raise ValueError("Power maximum scale must exceed the minimum scale")
            if self.power.huber_delta <= 0:
                raise ValueError("power.huber_delta must be positive")
        if self.wavelet_power.enabled:
            if self.wavelet_power.weight <= 0:
                raise ValueError(
                    "wavelet_power.weight must be positive when enabled"
                )
            if self.wavelet_power.huber_delta <= 0:
                raise ValueError("wavelet_power.huber_delta must be positive")
        if self.cross_scale.enabled:
            if self.cross_scale.weight <= 0:
                raise ValueError("cross_scale.weight must be positive when enabled")
            trained = set(morlet.training_scales)
            labels = self.cross_scale.pairs or tuple(
                (fine, coarse)
                for fine in self.cross_scale.fine_scales
                for coarse in self.cross_scale.coarse_scales
            )
            if not labels or any(
                fine not in trained or coarse not in trained or fine >= coarse
                for fine, coarse in labels
            ):
                raise ValueError("Every cross-scale fine scale must be below coarse")
            if not 0 <= self.cross_scale.correlation_weight_eta <= 1:
                raise ValueError("Cross-scale correlation eta must lie in [0, 1]")
            if self.cross_scale.maximum_uniform_multiple < 1:
                raise ValueError("Cross-scale weight cap must be >= uniform")
            if self.cross_scale.huber_delta <= 0:
                raise ValueError("cross_scale.huber_delta must be positive")
        if self.fbm_prior.enabled:
            trained = set(morlet.training_scales)
            if (
                not self.fbm_prior.replaced_scales
                or not self.fbm_prior.trusted_scales
                or not set(self.fbm_prior.replaced_scales).issubset(trained)
                or not set(self.fbm_prior.trusted_scales).issubset(trained)
                or set(self.fbm_prior.replaced_scales)
                & set(self.fbm_prior.trusted_scales)
            ):
                raise ValueError(
                    "fBm replaced/trusted scales must be disjoint trained scales"
                )
            if self.fbm_prior.realizations < 1 or self.fbm_prior.slope >= 0:
                raise ValueError("fBm needs realizations and a negative power slope")
            if not 0 <= self.fbm_prior.anisotropy_cap < 0.5:
                raise ValueError("fBm anisotropy_cap must lie in [0, 0.5)")
        if self.runtime.coordinate_chunk < 1:
            raise ValueError("Coordinate chunks must be positive")
        if (
            self.runtime.checkpoint_interval < 1
            or self.runtime.diagnostic_interval < 1
            or self.runtime.validation_interval < 1
        ):
            raise ValueError("Runtime intervals must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any], section: str) -> T:
    if not isinstance(values, dict):
        raise TypeError(f"Configuration section {section!r} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    converted: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, list) and "tuple" in str(hints.get(key)):
            value = tuple(
                tuple(item) if isinstance(item, list) else item for item in value
            )
        converted[key] = value
    return cls(**converted)


def load_single_image_config(path: str | Path) -> SingleImageConfig:
    """Load and validate a generic single-image YAML configuration."""

    with Path(path).open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping")
    section_types = {
        "input": SingleImageInputConfig,
        "geometry": SingleImageGeometryConfig,
        "backbone": SingleImageBackboneConfig,
        "morlet": SingleImageMorletConfig,
        "architecture": SingleImageArchitectureConfig,
        "decomposition": SingleImageDecompositionConfig,
        "initialization": SingleImageInitializationConfig,
        "power": SingleImagePowerConfig,
        "wavelet_power": SingleImageWaveletPowerConfig,
        "cross_scale": SingleImageCrossScaleConfig,
        "fbm_prior": SingleImageFbmPriorConfig,
        "optimizer": SingleImageOptimizerConfig,
        "runtime": SingleImageRuntimeConfig,
        "output": SingleImageOutputConfig,
    }
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    missing = {"input", "output"} - set(raw)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    kwargs = {
        name: _construct(section_types[name], values, name)
        for name, values in raw.items()
    }
    config = SingleImageConfig(**kwargs)
    config.validate()
    return config


def save_single_image_config(config: SingleImageConfig, path: str | Path) -> None:
    """Save the fully resolved configuration without Python YAML tags."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(config.as_dict(), stream, sort_keys=False)


def central_square_slices(
    shape: tuple[int, int], size: int
) -> tuple[slice, slice]:
    """Return slices for an exactly centered square."""

    if size > min(shape):
        raise ValueError("Central crop exceeds the array")
    if (shape[0] - size) % 2 or (shape[1] - size) % 2:
        raise ValueError("Central crop requires matching parity")
    y0 = (shape[0] - size) // 2
    x0 = (shape[1] - size) // 2
    return slice(y0, y0 + size), slice(x0, x0 + size)


def center_on_crop(
    canvas: np.ndarray, crop: tuple[slice, slice]
) -> tuple[np.ndarray, float]:
    """Subtract the scalar mean measured only on the delivered crop."""

    value = np.asarray(canvas)
    removed = float(np.mean(value[crop], dtype=np.float64))
    return value - removed, removed


def restore_crop_mean(centered_crop: np.ndarray, input_mean: float) -> np.ndarray:
    """Restore an exact scalar mean after synthesis."""

    value = np.asarray(centered_crop, dtype=np.float64)
    restored = value - np.mean(value, dtype=np.float64) + float(input_mean)
    return restored.astype(np.float32)


def block_average_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping square blocks without changing surface brightness."""

    value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError("Block averaging requires a two-dimensional image")
    if factor < 1:
        raise ValueError("Block-average factor must be positive")
    if factor == 1:
        return value.astype(np.float32, copy=True)
    ny, nx = value.shape
    if ny % factor or nx % factor:
        raise ValueError(
            f"Image shape {value.shape} is not divisible by factor {factor}"
        )
    reduced = value.reshape(
        ny // factor, factor, nx // factor, factor
    ).mean(axis=(1, 3), dtype=np.float64)
    return reduced.astype(np.float32)


def block_average_header(
    header: fits.Header, factor: int, shape: tuple[int, int]
) -> fits.Header:
    """Update a FITS WCS for pixels centered on block-averaged input pixels."""

    result = header.copy()
    if factor < 1:
        raise ValueError("Block-average factor must be positive")
    if factor > 1:
        for axis in (1, 2):
            crpix = f"CRPIX{axis}"
            if crpix in result:
                result[crpix] = (float(result[crpix]) - 0.5) / factor + 0.5
        cd_keys = [
            f"CD{world}_{pixel}"
            for world in (1, 2)
            for pixel in (1, 2)
        ]
        if any(key in result for key in cd_keys):
            for key in cd_keys:
                if key in result:
                    result[key] = float(result[key]) * factor
        else:
            for axis in (1, 2):
                cdelt = f"CDELT{axis}"
                if cdelt in result:
                    result[cdelt] = float(result[cdelt]) * factor
    result["NAXIS1"] = int(shape[1])
    result["NAXIS2"] = int(shape[0])
    return result


def gaussian_backbone_residual(
    image: np.ndarray, fwhm_pixels: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image exactly into a Gaussian backbone and zero-mean residual."""

    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Gaussian decomposition requires a finite 2D image")
    if fwhm_pixels <= 0:
        raise ValueError("Gaussian FWHM must be positive")
    sigma = float(fwhm_pixels) / 2.354820045
    smoothed = gaussian_filter(value, sigma=sigma, mode="reflect").astype(np.float32)
    residual = value - smoothed
    residual_mean = float(np.mean(residual, dtype=np.float64))
    residual = (residual - residual_mean).astype(np.float32)
    backbone = (smoothed + residual_mean).astype(np.float32)
    # Absorb float32 roundoff into the backbone so the decomposition is exact.
    backbone = (value - residual).astype(np.float32)
    return backbone, residual


def _header_pixel_scale_arcsec(header: fits.Header) -> float:
    """Return the finite mean celestial pixel scale recorded in a FITS header."""

    try:
        scales = proj_plane_pixel_scales(WCS(header).celestial) * 3600.0
    except Exception as error:
        raise ValueError("FITS WCS does not provide a usable pixel scale") from error
    if not np.isfinite(scales).all():
        raise ValueError("FITS WCS does not provide a finite pixel scale")
    return float(np.mean(scales))


def morlet_margin(max_scale: int, boundary_sigma: float = 2.0) -> int:
    """Return the conservative real-space support used by Morlet features."""

    return int(math.ceil(float(boundary_sigma) * 0.8 * 2 ** int(max_scale)))


def morlet_valid_center_mask(
    shape: tuple[int, int],
    scales: tuple[int, ...],
    boundary_sigma: float = 2.0,
) -> np.ndarray:
    """Mark response centers whose largest Morlet support stays in the image."""

    if not scales:
        raise ValueError("At least one scale is required")
    margin = morlet_margin(max(scales), boundary_sigma)
    mask = np.zeros(shape, dtype=bool)
    if 2 * margin < min(shape):
        mask[margin : shape[0] - margin, margin : shape[1] - margin] = True
    return mask


def _resolve_device(requested: str) -> torch.device:
    name = requested.lower()
    if name in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"gpu", "mps"} and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _gradient_norm(parameters) -> torch.Tensor:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return torch.zeros(())
    return torch.stack(values).sum().sqrt()


def _extended_header(header: fits.Header, margin: int, size: int) -> fits.Header:
    result = header.copy()
    if "CRPIX1" in result:
        result["CRPIX1"] = float(result["CRPIX1"]) + margin
    if "CRPIX2" in result:
        result["CRPIX2"] = float(result["CRPIX2"]) + margin
    result["NAXIS1"] = size
    result["NAXIS2"] = size
    return result


class SingleImageNeuralFieldTrainer:
    """Fit one latent-grid field to a single image's Morlet distribution."""

    def __init__(self, config: SingleImageConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.runtime.device)
        with fits.open(config.input.path, memmap=True) as hdul:
            hdu = hdul[config.input.hdu]
            source = np.asarray(hdu.data, dtype=np.float32)
            source_header = hdu.header.copy()
        self.pre_bin_gaussian_fwhm_pixels: float | None = None
        if config.input.pre_bin_gaussian_fwhm_arcsec is not None:
            source_pixel_scale = _header_pixel_scale_arcsec(source_header)
            self.pre_bin_gaussian_fwhm_pixels = (
                float(config.input.pre_bin_gaussian_fwhm_arcsec) / source_pixel_scale
            )
            source = gaussian_filter(
                source,
                sigma=self.pre_bin_gaussian_fwhm_pixels / 2.354820045,
                mode="reflect",
            ).astype(np.float32)
        self.input_image = block_average_image(
            source, config.input.block_average_factor
        )
        texture_header = block_average_header(
            source_header,
            config.input.block_average_factor,
            self.input_image.shape,
        )
        self._texture_header_for_scale = texture_header
        expected = (config.geometry.crop_size, config.geometry.crop_size)
        if config.backbone.path is None and self.input_image.shape != expected:
            raise ValueError(
                f"Configured crop is {expected}, but input is {self.input_image.shape}"
            )
        if not np.isfinite(self.input_image).all():
            raise ValueError("Single-image baseline requires a fully finite input")
        self.texture_mean = float(np.mean(self.input_image, dtype=np.float64))
        self.decomposition_fwhm_pixels = self._decomposition_fwhm_pixels()
        if config.backbone.path is not None:
            if self.decomposition_fwhm_pixels is not None:
                raise ValueError(
                    "Use either an external backbone or decomposition, not both"
                )
            self.backbone, self.input_header = self._load_external_backbone()
            self.target_centered = self.input_image - self.texture_mean
            self.input_mean = float(np.mean(self.backbone, dtype=np.float64))
        elif self.decomposition_fwhm_pixels is None:
            self.backbone = None
            self.target_centered = self.input_image - self.texture_mean
            self.input_header = texture_header
            self.input_mean = self.texture_mean
        else:
            self.backbone, self.target_centered = gaussian_backbone_residual(
                self.input_image, self.decomposition_fwhm_pixels
            )
            self.input_header = texture_header
            self.input_mean = self.texture_mean
        self.texture_images = [self.input_image]
        self.texture_centered_images = [self.target_centered]
        configured_textures = config.input.texture_paths or (config.input.path,)
        for path in dict.fromkeys(configured_textures):
            if Path(path) == Path(config.input.path):
                continue
            with fits.open(path, memmap=True) as hdul:
                value = np.asarray(hdul[config.input.hdu].data, dtype=np.float32)
                value_header = hdul[config.input.hdu].header.copy()
            if config.input.pre_bin_gaussian_fwhm_arcsec is not None:
                value = gaussian_filter(
                    value,
                    sigma=(
                        float(config.input.pre_bin_gaussian_fwhm_arcsec)
                        / _header_pixel_scale_arcsec(value_header)
                        / 2.354820045
                    ),
                    mode="reflect",
                ).astype(np.float32)
            value = block_average_image(value, config.input.block_average_factor)
            if value.shape != self.input_image.shape or not np.isfinite(value).all():
                raise ValueError("Every texture exemplar must be finite and match input.shape")
            self.texture_images.append(value)
            self.texture_centered_images.append(
                value - float(np.mean(value, dtype=np.float64))
            )
        self.crop = central_square_slices(
            (config.geometry.canvas_size, config.geometry.canvas_size),
            config.geometry.crop_size,
        )
        self.pixel_scale_arcsec = self._pixel_scale()
        self.initialization_canvas = self._load_initialization_canvas()
        self.target_mask = morlet_valid_center_mask(
            self.input_image.shape,
            config.morlet.training_scales,
            config.morlet.boundary_sigma,
        )
        self.output_mask = np.zeros(
            (config.geometry.canvas_size, config.geometry.canvas_size), dtype=bool
        )
        if config.morlet.output_sampling_region == "crop":
            self.output_mask[self.crop] = True
        else:
            coordinates = np.linspace(
                -1.0, 1.0, config.geometry.canvas_size, dtype=np.float32
            )
            yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
            self.output_mask = (
                xx * xx + yy * yy
                <= float(config.morlet.safe_canvas_radius) ** 2
            )
        self.target_descriptor, self.output_descriptor = self._build_descriptors()
        observed_target = torch.cat(
            [
                self.target_descriptor.features(value, center_mask=self.target_mask)
                for value in self.texture_centered_images
            ],
            dim=0,
        )
        target = self._apply_fbm_prior(observed_target)
        self.target_raw_features = target.detach()
        self.target_log_wavelet_energy = log_oriented_wavelet_energy(target).detach()
        scale_count = len(config.morlet.training_scales)
        self.target_log_wavelet_energy_by_scale = log_wavelet_energy_by_scale(
            target,
            scales=scale_count,
            orientations=config.morlet.orientations,
        ).detach()
        scale_index = {
            scale: index
            for index, scale in enumerate(config.morlet.training_scales)
        }
        self.cross_scale_labels = config.cross_scale.pairs or tuple(
            (fine, coarse)
            for fine in config.cross_scale.fine_scales
            for coarse in config.cross_scale.coarse_scales
        )
        self.cross_scale_pairs = tuple(
            (scale_index[fine], scale_index[coarse])
            for fine, coarse in self.cross_scale_labels
        )
        coupling_function = (
            log_aligned_cross_scale_coupling
            if config.cross_scale.aligned_orientations
            else log_cross_scale_envelope_coupling
        )
        self.target_cross_scale_coupling = (
            coupling_function(
                target,
                scales=scale_count,
                orientations=config.morlet.orientations,
                pairs=self.cross_scale_pairs,
            ).detach()
            if self.cross_scale_pairs
            else None
        )
        self.cross_scale_weights = (
            correlation_floor_weights(
                self.target_cross_scale_coupling,
                eta=config.cross_scale.correlation_weight_eta,
                maximum_uniform_multiple=config.cross_scale.maximum_uniform_multiple,
            )
            if self.cross_scale_pairs
            else None
        )
        self.normalizer = RobustNormalizer.fit(target.detach().cpu().numpy())
        self.target_features = self.normalizer.apply(target).detach()
        self.power_edges, self.target_power = self._build_power_target()
        validation_generator = torch.Generator(device=self.device).manual_seed(
            config.runtime.seed + 20000
        )
        self.validation_target_features = subsample_rows(
            self.target_features, config.morlet.output_samples, validation_generator
        ).detach()

    def _apply_fbm_prior(self, observed_target: torch.Tensor) -> torch.Tensor:
        """Replace configured contaminated scales with fitted anisotropic fBm."""

        setup = self.config.fbm_prior
        self.fbm_images: list[np.ndarray] = []
        self.fbm_prior_info: dict[str, Any] | None = None
        if not setup.enabled:
            return observed_target
        scales = self.config.morlet.training_scales
        orientations = self.config.morlet.orientations
        trusted_physical = [
            2.35 * self.pixel_scale_arcsec * 2 ** scale
            for scale in setup.trusted_scales
        ]
        angle, quadrupole = measure_spectral_quadrupole(
            self.target_centered,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            minimum_scale_arcsec=min(trusted_physical),
            maximum_scale_arcsec=max(trusted_physical),
            anisotropy_cap=setup.anisotropy_cap,
        )
        unit_images = [
            anisotropic_fbm(
                self.input_image.shape,
                slope=setup.slope,
                spectral_angle=angle,
                quadrupole=quadrupole,
                seed=self.config.runtime.seed + setup.seed_offset + index,
            )
            for index in range(setup.realizations)
        ]
        unit_feature_list = [
            self.target_descriptor.features(value, center_mask=self.target_mask)
            for value in unit_images
        ]
        unit_features = torch.cat(unit_feature_list, dim=0)
        observed_energy = log_wavelet_energy_by_scale(
            observed_target, scales=len(scales), orientations=orientations
        )
        unit_energy = log_wavelet_energy_by_scale(
            unit_features, scales=len(scales), orientations=orientations
        )
        index = {scale: position for position, scale in enumerate(scales)}
        trusted_index = torch.as_tensor(
            [index[scale] for scale in setup.trusted_scales],
            device=self.device,
        )
        log_scale = 0.5 * torch.mean(
            observed_energy[trusted_index] - unit_energy[trusted_index]
        )
        scaled_features = unit_features + log_scale
        amplitude = float(torch.exp(log_scale).cpu())
        self.fbm_images = [(amplitude * value).astype(np.float32) for value in unit_images]
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.runtime.seed + setup.seed_offset + 1000
        )
        sampled_fbm = subsample_rows(
            scaled_features, observed_target.shape[0], generator
        )
        hybrid = observed_target.clone()
        for scale in setup.replaced_scales:
            position = index[scale]
            section = slice(position * orientations, (position + 1) * orientations)
            hybrid[:, section] = sampled_fbm[:, section]
        scaled_energy = log_wavelet_energy_by_scale(
            scaled_features, scales=len(scales), orientations=orientations
        )
        fbm_energy_samples = torch.stack(
            [
                log_wavelet_energy_by_scale(
                    value + log_scale,
                    scales=len(scales),
                    orientations=orientations,
                )
                for value in unit_feature_list
            ]
        )
        observed_blocks = torch.tensor_split(observed_target, 4, dim=0)
        observed_energy_samples = torch.stack(
            [
                log_wavelet_energy_by_scale(
                    value, scales=len(scales), orientations=orientations
                )
                for value in observed_blocks
            ]
        )
        self.fbm_prior_info = {
            "spectral_angle_degrees": math.degrees(angle) % 180.0,
            "quadrupole": quadrupole,
            "amplitude": amplitude,
            "observed_log_energy": observed_energy.detach().cpu().numpy(),
            "fbm_log_energy": scaled_energy.detach().cpu().numpy(),
            "fbm_log_energy_std": fbm_energy_samples.std(dim=0).detach().cpu().numpy(),
            "observed_log_energy_std": observed_energy_samples.std(dim=0)
            .detach()
            .cpu()
            .numpy(),
            "scales": np.asarray(scales),
            "trusted_scales": np.asarray(setup.trusted_scales),
            "replaced_scales": np.asarray(setup.replaced_scales),
            "slope": setup.slope,
        }
        return hybrid

    def _load_external_backbone(self) -> tuple[np.ndarray, fits.Header]:
        """Load a centred fixed backbone with its output WCS and calibration."""

        setup = self.config.backbone
        assert setup.path is not None
        with fits.open(setup.path, memmap=True) as hdul:
            hdu = hdul[setup.hdu]
            value = np.asarray(hdu.data, dtype=np.float32)
            header = hdu.header.copy()
            primary = hdul[0].header
        scale = float(setup.value_scale)
        offset = float(setup.value_offset)
        if setup.affine_from_primary:
            offset += scale * float(primary["RADOFF"])
            scale *= float(primary["RADSLOPE"])
        value = scale * value + offset
        crop = central_square_slices(value.shape, self.config.geometry.crop_size)
        value = value[crop].astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("External backbone crop contains invalid pixels")
        output_header = header.copy()
        output_header.update(WCS(header).celestial.slice(crop).to_header(relax=True))
        output_header["NAXIS1"] = value.shape[1]
        output_header["NAXIS2"] = value.shape[0]
        output_header["BUNIT"] = "MJy/sr"
        output_scale = _header_pixel_scale_arcsec(output_header)
        try:
            source_scale = _header_pixel_scale_arcsec(
                self._texture_header_for_scale
            )
        except ValueError:
            source_scale = self.config.input.pixel_scale_arcsec
        if source_scale is not None and not np.isclose(output_scale, source_scale, rtol=0.01):
            raise ValueError("Texture and external backbone pixel scales differ")
        return value, output_header

    def _decomposition_fwhm_pixels(self) -> float | None:
        setup = self.config.decomposition
        if not setup.enabled:
            return None
        if setup.gaussian_fwhm_pixels is not None:
            return float(setup.gaussian_fwhm_pixels)
        return float(2 ** max(self.config.morlet.training_scales))

    def _load_initialization_canvas(self) -> np.ndarray | None:
        """Reproject an optional initialization image and force zero crop mean."""

        setup = self.config.initialization
        if setup.path is None:
            return None
        with fits.open(setup.path, memmap=True) as hdul:
            source = (
                setup.value_scale
                * np.asarray(hdul[setup.hdu].data, dtype=np.float32)
                + setup.value_offset
            ).astype(np.float32)
            source_header = hdul[setup.hdu].header.copy()
        margin = (self.config.geometry.canvas_size - self.config.geometry.crop_size) // 2
        if setup.canvas_extension == "reflect":
            projected_crop, footprint = reproject_interp(
                (source, WCS(source_header).celestial),
                WCS(self.input_header).celestial,
                shape_out=self.input_image.shape,
                order="bilinear",
            )
            if np.any(footprint <= 0.0) or not np.isfinite(projected_crop).all():
                raise ValueError(
                    "Initialization image does not fully cover the input grid"
                )
            projected_crop = (
                projected_crop
                - np.mean(projected_crop, dtype=np.float64)
            ).astype(np.float32)
            projected = np.pad(
                projected_crop,
                ((margin, margin), (margin, margin)),
                mode="reflect",
            )
            footprint = np.ones_like(projected)
        else:
            target_header = _extended_header(
                self.input_header, margin, self.config.geometry.canvas_size
            )
            projected, footprint = reproject_interp(
                (source, WCS(source_header).celestial),
                WCS(target_header).celestial,
                shape_out=(
                    self.config.geometry.canvas_size,
                    self.config.geometry.canvas_size,
                ),
                order="bilinear",
            )
        if np.any(footprint <= 0.0) or not np.isfinite(projected).all():
            raise ValueError(
                "Initialization image does not fully cover the expanded canvas"
            )
        centered, _ = center_on_crop(projected.astype(np.float32), self.crop)
        return centered.astype(np.float32)

    def _pixel_scale(self) -> float:
        configured = self.config.input.pixel_scale_arcsec
        if configured is not None:
            return float(configured)
        return _header_pixel_scale_arcsec(self.input_header)

    def _build_power_target(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build a fixed log-power target for the neural high-pass residual."""

        setup = self.config.power
        if not setup.enabled:
            return None, None
        largest = setup.maximum_scale_arcsec
        if largest is None:
            largest = max(self.input_image.shape) * self.pixel_scale_arcsec
        edges = torch.logspace(
            math.log10(60.0 / float(largest)),
            math.log10(60.0 / float(setup.minimum_scale_arcsec)),
            int(setup.bins) + 1,
            device=self.device,
        )
        observed = torch.stack(
            [
                radial_log_power_physical(
                    torch.as_tensor(value, device=self.device),
                    pixel_scale_arcmin=self.pixel_scale_arcsec / 60.0,
                    frequency_edges=edges,
                )[0]
                for value in self.texture_centered_images
            ]
        ).mean(dim=0, keepdim=True).detach()
        if not self.config.fbm_prior.enabled:
            return edges, observed
        fbm = torch.stack(
            [
                radial_log_power_physical(
                    torch.as_tensor(value, device=self.device),
                    pixel_scale_arcmin=self.pixel_scale_arcsec / 60.0,
                    frequency_edges=edges,
                )[0]
                for value in self.fbm_images
            ]
        ).mean(dim=0, keepdim=True).detach()
        replaced = sorted(self.config.fbm_prior.replaced_scales)
        trusted = sorted(self.config.fbm_prior.trusted_scales)
        fine_scale = 2.35 * self.pixel_scale_arcsec * 2 ** max(replaced)
        coarse_scale = 2.35 * self.pixel_scale_arcsec * 2 ** min(trusted)
        frequency = torch.sqrt(edges[:-1] * edges[1:])
        low = 60.0 / coarse_scale
        high = 60.0 / fine_scale
        fraction = ((frequency - low) / (high - low)).clamp(0.0, 1.0)
        blend = 0.5 - 0.5 * torch.cos(torch.pi * fraction)
        target = (1.0 - blend[None]) * observed + blend[None] * fbm
        if self.fbm_prior_info is not None:
            self.fbm_prior_info["power_frequency_arcmin_inverse"] = (
                frequency.detach().cpu().numpy()
            )
            self.fbm_prior_info["observed_log_power"] = (
                observed[0].detach().cpu().numpy()
            )
            self.fbm_prior_info["fbm_log_power"] = fbm[0].detach().cpu().numpy()
            self.fbm_prior_info["hybrid_log_power"] = (
                target[0].detach().cpu().numpy()
            )
        return edges, target

    def _build_descriptors(
        self,
    ) -> tuple[OrientedMorletFeatures, OrientedMorletFeatures]:
        setup = self.config.morlet
        j_count = max(setup.diagnostic_scales) + 1
        target_calc = ScatteringMean2D(
            self.input_image.shape,
            j_count,
            setup.orientations,
            pad_pixels=setup.target_padding_pixels,
            mask_boundary=False,
            device=str(self.device),
        )
        output_shape = (
            self.config.geometry.canvas_size,
            self.config.geometry.canvas_size,
        )
        output_calc = ScatteringMean2D(
            output_shape,
            j_count,
            setup.orientations,
            pad_pixels=0,
            mask_boundary=False,
            device=str(self.device),
        )
        kwargs = {
            "spatial_stride": 1,
            "boundary_sigma": setup.boundary_sigma,
            "orientation_pool": False,
        }
        return (
            OrientedMorletFeatures(
                target_calc, setup.training_scales, **kwargs
            ),
            OrientedMorletFeatures(
                output_calc, setup.training_scales, **kwargs
            ),
        )

    def _build_model(self) -> MultiResolutionLatentField:
        setup = self.config.architecture
        canvas_extent_arcsec = (
            (self.config.geometry.canvas_size - 1) * self.pixel_scale_arcsec
        )
        if setup.fine_branch == "fourier":
            assert setup.fourier_min_scale_arcsec is not None
            assert setup.fourier_max_scale_arcsec is not None
            fourier_min_cycles = (
                canvas_extent_arcsec / setup.fourier_max_scale_arcsec
            )
            fourier_max_cycles = min(
                canvas_extent_arcsec / setup.fourier_min_scale_arcsec,
                0.5 * (self.config.geometry.canvas_size - 1),
            )
            fourier_count = setup.fourier_feature_count
        else:
            fourier_min_cycles = 1.0
            fourier_max_cycles = 2.0
            fourier_count = 0
        return MultiResolutionLatentField(
            setup.grid_resolutions,
            setup.channels_per_level,
            setup.hidden_features,
            setup.hidden_layers,
            setup.output_scale,
            setup.include_coordinates,
            setup.rotate_and_offset_grids,
            (
                self.config.runtime.seed
                if setup.grid_frame_seed is None
                else setup.grid_frame_seed
            ),
            fourier_count,
            fourier_min_cycles,
            fourier_max_cycles,
            (
                self.config.runtime.seed + 70000
                if setup.fourier_seed is None
                else setup.fourier_seed
            ),
        ).to(self.device)

    def _render_centered(
        self, model: MultiResolutionLatentField
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.config.geometry.canvas_size
        raw = render_raw_field(
            model,
            (size, size),
            coordinate_chunk=self.config.runtime.coordinate_chunk,
        )
        centered = self._project_and_center(raw)
        return centered, centered[self.crop]

    def _project_and_center(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the configured high-pass on the expanded canvas and center the crop."""

        projected = raw.float()
        if self.decomposition_fwhm_pixels is not None:
            transfer = gaussian_resolution_transfer(
                tuple(projected.shape[-2:]),
                self.decomposition_fwhm_pixels,
                device=projected.device,
                dtype=projected.dtype,
            )
            projected = torch.fft.ifftn(
                torch.fft.fftn(projected, dim=(-2, -1)) * (1.0 - transfer),
                dim=(-2, -1),
            ).real
        return projected - projected[self.crop].mean()

    def _per_scale_swd(
        self,
        output_features: torch.Tensor,
        step: int,
    ) -> dict[int, float]:
        setup = self.config.morlet
        result: dict[int, float] = {}
        with torch.no_grad():
            for index, scale in enumerate(setup.training_scales):
                section = slice(
                    index * setup.orientations,
                    (index + 1) * setup.orientations,
                )
                generator = torch.Generator(device=self.device).manual_seed(
                    self.config.runtime.seed + 50000 + 100 * step + int(scale)
                )
                value = sliced_wasserstein_loss(
                    output_features[:, section],
                    self.target_features[:, section],
                    projection_count=setup.projections,
                    quantile_count=setup.quantiles,
                    generator=generator,
                    chunk_size=setup.projection_chunk,
                )
                result[int(scale)] = float(value.cpu())
        return result

    def _power_loss(self, centered_crop: torch.Tensor) -> torch.Tensor:
        """Return the weighted, differentiable residual radial-power mismatch."""

        if self.power_edges is None or self.target_power is None:
            return torch.zeros((), device=self.device, dtype=centered_crop.dtype)
        measured = radial_log_power_physical(
            centered_crop,
            pixel_scale_arcmin=self.pixel_scale_arcsec / 60.0,
            frequency_edges=self.power_edges,
        )
        return torch.nn.functional.huber_loss(
            measured,
            self.target_power.to(dtype=measured.dtype),
            delta=self.config.power.huber_delta,
            reduction="mean",
        )

    def _wavelet_power_loss(
        self, output_log_features: torch.Tensor
    ) -> torch.Tensor:
        """Match absolute Morlet energy with equal weight for every scale."""

        setup = self.config.wavelet_power
        if not setup.enabled:
            return torch.zeros(
                (), device=output_log_features.device, dtype=output_log_features.dtype
            )
        measured = log_wavelet_energy_by_scale(
            output_log_features,
            scales=len(self.config.morlet.training_scales),
            orientations=self.config.morlet.orientations,
        )
        return torch.nn.functional.huber_loss(
            measured,
            self.target_log_wavelet_energy_by_scale.to(dtype=measured.dtype),
            delta=setup.huber_delta,
            reduction="mean",
        )

    def _wavelet_power_components(
        self, output_log_features: torch.Tensor
    ) -> torch.Tensor:
        measured = log_wavelet_energy_by_scale(
            output_log_features,
            scales=len(self.config.morlet.training_scales),
            orientations=self.config.morlet.orientations,
        )
        return torch.nn.functional.huber_loss(
            measured,
            self.target_log_wavelet_energy_by_scale.to(dtype=measured.dtype),
            delta=self.config.wavelet_power.huber_delta,
            reduction="none",
        )

    def _cross_scale_loss(self, output_log_features: torch.Tensor) -> torch.Tensor:
        setup = self.config.cross_scale
        if not setup.enabled:
            return torch.zeros(
                (), device=output_log_features.device, dtype=output_log_features.dtype
            )
        coupling_function = (
            log_aligned_cross_scale_coupling
            if setup.aligned_orientations
            else log_cross_scale_envelope_coupling
        )
        measured = coupling_function(
            output_log_features,
            scales=len(self.config.morlet.training_scales),
            orientations=self.config.morlet.orientations,
            pairs=self.cross_scale_pairs,
        )
        components = torch.nn.functional.huber_loss(
            measured,
            self.target_cross_scale_coupling.to(dtype=measured.dtype),
            delta=setup.huber_delta,
            reduction="none",
        )
        return torch.sum(
            components * self.cross_scale_weights.to(dtype=components.dtype)
        )

    def _cross_scale_components(
        self, output_log_features: torch.Tensor
    ) -> torch.Tensor:
        coupling_function = (
            log_aligned_cross_scale_coupling
            if self.config.cross_scale.aligned_orientations
            else log_cross_scale_envelope_coupling
        )
        measured = coupling_function(
            output_log_features,
            scales=len(self.config.morlet.training_scales),
            orientations=self.config.morlet.orientations,
            pairs=self.cross_scale_pairs,
        )
        return torch.nn.functional.huber_loss(
            measured,
            self.target_cross_scale_coupling.to(dtype=measured.dtype),
            delta=self.config.cross_scale.huber_delta,
            reduction="none",
        )

    @torch.no_grad()
    def _fixed_validation_score(
        self, model: MultiResolutionLatentField
    ) -> dict[str, float]:
        """Evaluate the seed-fixed composite validation objective."""

        centered_canvas, centered_crop = self._render_centered(model)
        output_features = self.output_descriptor.features(
            centered_canvas, center_mask=self.output_mask
        )
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.runtime.seed + 30000
        )
        sampled_output = subsample_rows(
            output_features, self.config.morlet.output_samples, generator
        )
        standardized_output = self.normalizer.apply(sampled_output)
        swd = sliced_wasserstein_loss(
            standardized_output,
            self.validation_target_features,
            projection_count=self.config.morlet.projections,
            quantile_count=self.config.morlet.quantiles,
            generator=torch.Generator(device=self.device).manual_seed(
                self.config.runtime.seed + 40000
            ),
            chunk_size=self.config.morlet.projection_chunk,
        )
        wavelet_power = self._wavelet_power_loss(sampled_output)
        cross_scale = self._cross_scale_loss(sampled_output)
        radial_power = self._power_loss(centered_crop)
        score = (
            swd
            + self.config.wavelet_power.weight * wavelet_power
            + self.config.cross_scale.weight * cross_scale
            + self.config.power.weight * radial_power
        )
        return {
            "score": float(score.cpu()),
            "swd": float(swd.cpu()),
            "wavelet_power": float(wavelet_power.cpu()),
            "cross_scale": float(cross_scale.cpu()),
            "radial_power": float(radial_power.cpu()),
        }

    def _prefit_initialization(
        self,
        model: MultiResolutionLatentField,
        directory: Path,
    ) -> tuple[list[dict[str, float]], np.ndarray | None]:
        """Fit model parameters to the zero-mean initialization canvas."""

        if self.initialization_canvas is None:
            return [], None
        setup = self.config.initialization
        target = torch.as_tensor(
            self.initialization_canvas, device=self.device, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=setup.learning_rate)
        history: list[dict[str, float]] = []
        for step in range(setup.prefit_steps):
            optimizer.zero_grad(set_to_none=True)
            centered_canvas, _ = self._render_centered(model)
            loss = torch.mean((centered_canvas - target) ** 2)
            loss.backward()
            grad_norm = _gradient_norm(model.parameters()).to(self.device)
            if self.config.optimizer.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.gradient_clip
                )
            optimizer.step()
            history.append(
                {
                    "step": float(step),
                    "mse": float(loss.detach().cpu()),
                    "gradient_norm": float(grad_norm.cpu()),
                }
            )
        with torch.no_grad():
            fitted, _ = self._render_centered(model)
        fitted_np = fitted.detach().cpu().numpy().astype(np.float32)
        with (directory / "initialization_history.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        np.save(directory / "initialization_target_canvas.npy", self.initialization_canvas)
        np.save(directory / "initialization_model_canvas.npy", fitted_np)
        return history, fitted_np

    def profile_training_step(self) -> dict[str, float | str]:
        """Run one unsaved forward/backward step for device and memory profiling."""

        config = self.config
        _seed_everything(config.runtime.seed, config.runtime.deterministic)
        model = self._build_model()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optimizer.learning_rate
        )
        use_amp = config.optimizer.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            centered_canvas, _ = self._render_centered(model)
        output_features = self.output_descriptor.features(
            centered_canvas, center_mask=self.output_mask
        )
        generator = torch.Generator(device=self.device).manual_seed(
            config.runtime.seed + 1000
        )
        standardized_output = self.normalizer.apply(
            subsample_rows(output_features, config.morlet.output_samples, generator)
        )
        loss = sliced_wasserstein_loss(
            standardized_output,
            self.target_features,
            projection_count=config.morlet.projections,
            quantile_count=config.morlet.quantiles,
            generator=torch.Generator(device=self.device).manual_seed(
                config.runtime.seed + 10000
            ),
            chunk_size=config.morlet.projection_chunk,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = _gradient_norm(model.parameters()).to(self.device)
        if config.optimizer.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.gradient_clip
            )
        scaler.step(optimizer)
        scaler.update()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result: dict[str, float | str] = {
            "device": str(self.device),
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.cpu()),
            "step_seconds": float(time.perf_counter() - started),
        }
        if self.device.type == "cuda":
            result["allocated_gpu_memory_mb"] = float(
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
            result["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
        return result

    def train(self) -> dict[str, Any]:
        """Run the configured single realization and write all core products."""

        config = self.config
        seed = config.runtime.seed
        _seed_everything(seed, config.runtime.deterministic)
        directory = Path(config.output.root) / config.output.run_name / f"seed_{seed}"
        if directory.exists() and any(directory.iterdir()) and not config.output.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        save_single_image_config(config, directory / "config.resolved.yaml")

        model = self._build_model()
        initialization_history, fitted_initialization = self._prefit_initialization(
            model, directory
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optimizer.learning_rate
        )
        use_amp = config.optimizer.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        history: list[dict[str, float]] = []
        best_validation_score = float("inf")
        best_validation_components: dict[str, float] = {}
        best_step = 0
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for step in range(config.optimizer.steps):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                centered_canvas, centered_crop = self._render_centered(model)
            output_features = self.output_descriptor.features(
                centered_canvas, center_mask=self.output_mask
            )
            sample_generator = torch.Generator(device=self.device).manual_seed(
                seed + 1000 + step
            )
            sampled_output = subsample_rows(
                output_features, config.morlet.output_samples, sample_generator
            )
            standardized_output = self.normalizer.apply(sampled_output)
            projection_generator = torch.Generator(device=self.device).manual_seed(
                seed + 10000 + step
            )
            morlet_loss = sliced_wasserstein_loss(
                standardized_output,
                self.target_features,
                projection_count=config.morlet.projections,
                quantile_count=config.morlet.quantiles,
                generator=projection_generator,
                chunk_size=config.morlet.projection_chunk,
            )
            wavelet_power_loss = self._wavelet_power_loss(sampled_output)
            cross_scale_loss = self._cross_scale_loss(sampled_output)
            power_loss = self._power_loss(centered_crop)
            loss = (
                morlet_loss
                + config.wavelet_power.weight * wavelet_power_loss
                + config.cross_scale.weight * cross_scale_loss
                + config.power.weight * power_loss
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = _gradient_norm(model.parameters()).to(self.device)
            if config.optimizer.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()
            per_scale = self._per_scale_swd(standardized_output.detach(), step)
            with torch.no_grad():
                wavelet_components = self._wavelet_power_components(
                    sampled_output.detach()
                )
                cross_components = (
                    self._cross_scale_components(sampled_output.detach())
                    if config.cross_scale.enabled
                    else torch.empty(0, device=self.device)
                )
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "morlet_loss": float(morlet_loss.detach().cpu()),
                "wavelet_power_loss": float(wavelet_power_loss.detach().cpu()),
                "cross_scale_loss": float(cross_scale_loss.detach().cpu()),
                "power_loss": float(power_loss.detach().cpu()),
                "gradient_norm": float(grad_norm.cpu()),
                "crop_rms": float(centered_crop.detach().square().mean().sqrt().cpu()),
                "step_seconds": float(time.perf_counter() - step_started),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            if (
                (step + 1) % config.runtime.validation_interval == 0
                or step + 1 == config.optimizer.steps
            ):
                validation = self._fixed_validation_score(model)
                row["validation_score"] = validation["score"]
                row["validation_swd"] = validation["swd"]
                row["validation_wavelet_power"] = validation["wavelet_power"]
                row["validation_cross_scale"] = validation["cross_scale"]
                row["validation_radial_power"] = validation["radial_power"]
                if validation["score"] < best_validation_score:
                    best_validation_score = validation["score"]
                    best_validation_components = validation
                    best_step = step + 1
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": best_step,
                            "validation_score": best_validation_score,
                            "validation_swd": validation["swd"],
                            "validation_wavelet_power": validation[
                                "wavelet_power"
                            ],
                            "validation_cross_scale": validation["cross_scale"],
                            "validation_radial_power": validation["radial_power"],
                        },
                        directory / "best.pt",
                    )
            else:
                row["validation_score"] = float("nan")
                row["validation_swd"] = float("nan")
                row["validation_wavelet_power"] = float("nan")
                row["validation_cross_scale"] = float("nan")
                row["validation_radial_power"] = float("nan")
            row.update({f"morlet_j{scale}": value for scale, value in per_scale.items()})
            row.update(
                {
                    f"wavelet_power_j{scale}": float(wavelet_components[index].cpu())
                    for index, scale in enumerate(config.morlet.training_scales)
                }
            )
            row.update(
                {
                    f"cross_scale_j{fine}_j{coarse}": float(
                        cross_components[index].cpu()
                    )
                    for index, (fine, coarse) in enumerate(self.cross_scale_labels)
                }
            )
            history.append(row)

            if (step + 1) % config.runtime.checkpoint_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step + 1,
                    },
                    directory / f"checkpoint_{step + 1:06d}.pt",
                )
            if (step + 1) % config.runtime.diagnostic_interval == 0:
                intermediate = directory / "intermediate"
                intermediate.mkdir(exist_ok=True)
                np.save(
                    intermediate / f"central_crop_{step + 1:06d}.npy",
                    centered_crop.detach().cpu().numpy().astype(np.float32),
                )
                print(
                    f"seed={seed} step={step + 1}/{config.optimizer.steps} "
                    f"loss={row['loss']:.6g} grad={row['gradient_norm']:.4g} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

        torch.save(
            {"model": model.state_dict(), "step": len(history)},
            directory / "final.pt",
        )
        with (directory / "loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        selected = torch.load(directory / "best.pt", map_location=self.device)
        model.load_state_dict(selected["model"])
        model.eval()
        with torch.no_grad():
            size = config.geometry.canvas_size
            raw_canvas = render_raw_field(
                model,
                (size, size),
                coordinate_chunk=config.runtime.coordinate_chunk,
            )
            centered_canvas = self._project_and_center(raw_canvas)
            centered_crop = centered_canvas[self.crop]
        raw_canvas_np = raw_canvas.cpu().numpy().astype(np.float32)
        centered_canvas_np = centered_canvas.cpu().numpy().astype(np.float32)
        centered_crop_np = centered_crop.cpu().numpy().astype(np.float32)
        if self.backbone is None:
            synthesis = restore_crop_mean(centered_crop_np, self.input_mean)
        else:
            synthesis = restore_crop_mean(
                self.backbone + centered_crop_np, self.input_mean
            )
        restored_canvas = centered_canvas_np + self.input_mean
        margin = (config.geometry.canvas_size - config.geometry.crop_size) // 2

        primary_header = self.input_header.copy()
        primary_header["METHOD"] = "NF1IMGSW"
        primary_header["RANDSEED"] = seed
        primary_header["NSTEP"] = config.optimizer.steps
        primary_header["INPMEAN"] = self.input_mean
        primary_header["CANVSIZE"] = config.geometry.canvas_size
        primary_header["MORJMIN"] = min(config.morlet.training_scales)
        primary_header["MORJMAX"] = max(config.morlet.training_scales)
        primary_header["GRIDROT"] = config.architecture.rotate_and_offset_grids
        primary_header["OUTSAMP"] = config.morlet.output_sampling_region
        primary_header["WVLPWR"] = config.wavelet_power.enabled
        primary_header["XSCAT"] = config.cross_scale.enabled
        primary_header["NFBRANCH"] = config.architecture.fine_branch
        primary_header["FBMPRIOR"] = config.fbm_prior.enabled
        primary_header["INIT"] = (
            str(config.initialization.hdu)
            if config.initialization.path is not None
            else "random"
        )
        primary_header["INITMEAN"] = 0.0
        if self.pre_bin_gaussian_fwhm_pixels is not None:
            primary_header["PREFWHM"] = (
                float(config.input.pre_bin_gaussian_fwhm_arcsec),
                "Pre-bin Gaussian FWHM [arcsec]",
            )
        if self.decomposition_fwhm_pixels is not None:
            primary_header["SPLITJ"] = max(config.morlet.training_scales)
            primary_header["SPLTFWH"] = (
                self.decomposition_fwhm_pixels,
                "Gaussian split FWHM [pixel]",
            )
        products = [
            fits.PrimaryHDU(synthesis, header=primary_header),
            fits.ImageHDU(
                centered_crop_np,
                header=self.input_header,
                name="CENTERED",
            ),
        ]
        if self.backbone is not None:
            products.extend(
                [
                    fits.ImageHDU(
                        self.backbone, header=self.input_header, name="BACKBONE"
                    ),
                    fits.ImageHDU(
                        self.target_centered,
                        header=self.input_header,
                        name="TARGET_HP",
                    ),
                ]
            )
        fits.HDUList(products).writeto(
            directory / "synthesis.fits", overwrite=config.output.overwrite
        )
        fits.PrimaryHDU(
            restored_canvas,
            header=_extended_header(
                self.input_header, margin, config.geometry.canvas_size
            ),
        ).writeto(directory / "full_canvas.fits", overwrite=config.output.overwrite)
        np.save(directory / "centered_canvas.npy", centered_canvas_np)
        np.save(directory / "raw_canvas.npy", raw_canvas_np)

        from .single_image_diagnostics import create_single_image_diagnostics

        metrics = create_single_image_diagnostics(
            directory=directory,
            input_image=self.input_image,
            synthesis=synthesis,
            centered_canvas=centered_canvas_np,
            restored_canvas=restored_canvas,
            crop=self.crop,
            target_mask=self.target_mask,
            input_mean=self.texture_mean,
            output_mean=self.input_mean,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            config=config,
            history=history,
            model=model,
            target_calculator=self.target_descriptor.calculator,
            output_calculator=self.output_descriptor.calculator,
            normalizer=self.normalizer,
            target_features=self.target_features,
            target_raw_features=self.target_raw_features,
            device=self.device,
            initialization_target=self.initialization_canvas,
            initialization_model=fitted_initialization,
            initialization_history=initialization_history,
            backbone=self.backbone,
            target_residual=self.target_centered,
            raw_canvas=raw_canvas_np,
            decomposition_fwhm_pixels=self.decomposition_fwhm_pixels,
            fbm_prior_info=self.fbm_prior_info,
        )
        metrics.update(
            {
                "device": str(self.device),
                "runtime_seconds": float(time.perf_counter() - started),
                "mean_step_seconds": float(
                    np.mean([row["step_seconds"] for row in history])
                ),
                "input_mean": self.input_mean,
                "synthesis_mean": float(np.mean(synthesis, dtype=np.float64)),
                "target_response_count": int(self.target_features.shape[0]),
                "texture_exemplar_count": float(len(self.texture_images)),
                "best_step": float(best_step),
                "best_validation_score": float(best_validation_score),
                "best_validation_swd": float(
                    best_validation_components.get("swd", float("nan"))
                ),
                "best_validation_wavelet_power": float(
                    best_validation_components.get(
                        "wavelet_power", float("nan")
                    )
                ),
                "best_validation_cross_scale": float(
                    best_validation_components.get("cross_scale", float("nan"))
                ),
                "best_validation_radial_power": float(
                    best_validation_components.get("radial_power", float("nan"))
                ),
                "selected_checkpoint": "best.pt",
                "initialization_enabled": float(
                    self.initialization_canvas is not None
                ),
                "decomposition_enabled": float(self.backbone is not None),
                "pre_bin_gaussian_fwhm_arcsec": config.input.pre_bin_gaussian_fwhm_arcsec,
            }
        )
        if self.decomposition_fwhm_pixels is not None:
            metrics["decomposition_fwhm_pixels"] = self.decomposition_fwhm_pixels
            metrics["decomposition_fwhm_arcsec"] = (
                self.decomposition_fwhm_pixels * self.pixel_scale_arcsec
            )
        if self.fbm_prior_info is not None:
            metrics["fbm_spectral_angle_degrees"] = float(
                self.fbm_prior_info["spectral_angle_degrees"]
            )
            metrics["fbm_quadrupole"] = float(self.fbm_prior_info["quadrupole"])
            metrics["fbm_amplitude"] = float(self.fbm_prior_info["amplitude"])
        if initialization_history:
            metrics["initialization_final_mse"] = initialization_history[-1]["mse"]
        if self.device.type == "cuda":
            metrics["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return metrics
