"""Differentiable M07 local-amplitude modulation for cirrus synthesis.

The constraint compares the local RMS of one-octave Gaussian detail maps with
the surface-brightness fluctuation predicted by Miville-Deschenes et al.
(2007; M07).  M07's scale law is evaluated without clipping outside its
published 5 arcmin--12.5 degree calibration range; those scales are explicitly
flagged as extrapolations in the target metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as torch_f

from .cirrus_scaling import m07_sigma_lambda


M07_MIN_SCALE_ARCMIN = 5.0
M07_MAX_SCALE_ARCMIN = 12.5 * 60.0


@dataclass(frozen=True)
class M07ModulationConfig:
    """Physical and numerical choices for the local-amplitude constraint.

    ``band_to_I100`` converts a map in the synthesis-band units to equivalent
    100 micron MJy/sr. It may be a scalar (future optical/DGL use) or a spatial
    color map (FIR modified-blackbody use). ``amplitude_correction`` is a
    separate arbitrary multiplier on the target RMS.
    """

    band_to_I100: float | np.ndarray
    amplitude_correction: float = 1.0
    scale_indices: tuple[int, ...] = (4, 5, 6)
    context_factor: float = 3.0
    spacing_factor: float = 0.5
    extrapolate: bool = True
    relation_color_mode: str = "global_median"

    def __post_init__(self) -> None:
        if self.amplitude_correction <= 0.0:
            raise ValueError("amplitude_correction must be positive")
        if not self.scale_indices or min(self.scale_indices) < 0:
            raise ValueError("scale_indices must contain non-negative dyadic indices")
        if self.context_factor <= 1.0:
            raise ValueError("context_factor must be greater than one")
        if not 0.0 < self.spacing_factor <= self.context_factor:
            raise ValueError("spacing_factor must be positive and no larger than context_factor")
        if self.relation_color_mode not in {"global_median", "local"}:
            raise ValueError("relation_color_mode must be 'global_median' or 'local'")


@dataclass(frozen=True)
class M07ScaleTarget:
    """Fixed local intensity and target-RMS arrays for one dyadic scale."""

    j: int
    scale_pixels: int
    scale_arcmin: float
    context_pixels: int
    stride_pixels: int
    local_band_mean: np.ndarray
    local_I100: np.ndarray
    target_sigma_band: np.ndarray
    extrapolated: bool


def _as_image(array: float | np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 0:
        value = np.full(shape, float(value), dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return value


def _gaussian_kernel1d(fwhm_pixels: float, device: torch.device) -> torch.Tensor:
    sigma = float(fwhm_pixels) / 2.354820045
    radius = max(int(math.ceil(4.0 * sigma)), 1)
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_gaussian(image: torch.Tensor, fwhm_pixels: float) -> torch.Tensor:
    kernel = _gaussian_kernel1d(fwhm_pixels, image.device)
    radius = kernel.numel() // 2
    if radius >= min(image.shape[-2:]):
        raise ValueError("Gaussian kernel is too large for reflection support")
    data = image[:, None] if image.ndim == 3 else image
    horizontal = torch_f.conv2d(
        torch_f.pad(data, (radius, radius, 0, 0), mode="reflect"),
        kernel[None, None, None, :],
    )
    vertical = torch_f.conv2d(
        torch_f.pad(horizontal, (0, 0, radius, radius), mode="reflect"),
        kernel[None, None, :, None],
    )
    return vertical[:, 0] if image.ndim == 3 else vertical


def _local_mean(image: torch.Tensor, kernel_pixels: int, stride_pixels: int) -> torch.Tensor:
    data = image[:, None] if image.ndim == 3 else image
    pooled = torch_f.avg_pool2d(data, kernel_size=kernel_pixels, stride=stride_pixels)
    return pooled[:, 0] if image.ndim == 3 else pooled


class M07LocalRMSConstraint:
    """Return log local-RMS residuals relative to fixed M07 target maps.

    The Planck-band backbone and its ``band_to_I100`` conversion are fixed.
    For dyadic index ``j``, the angular scale is ``2**j`` pixels. The output
    detail is ``G(scale/2) - G(scale)`` and its local RMS is measured in square
    contexts of width ``context_factor * scale``. No padded values enter these
    contexts; reflection is used only to support the two Gaussian convolutions.
    """

    def __init__(
        self,
        planck_band_map: np.ndarray,
        pixel_scale_arcsec: float,
        config: M07ModulationConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        planck = np.asarray(planck_band_map, dtype=np.float32)
        if planck.ndim != 2 or not np.isfinite(planck).all() or np.any(planck <= 0.0):
            raise ValueError("planck_band_map must be a finite positive 2D image")
        if pixel_scale_arcsec <= 0.0:
            raise ValueError("pixel_scale_arcsec must be positive")
        self.shape = planck.shape
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)
        self.config = config
        self.device = torch.device(device)
        conversion = _as_image(config.band_to_I100, planck.shape, "band_to_I100")
        self.band_to_I100 = conversion
        self.relation_band_over_i100 = float(np.median(1.0 / conversion))
        planck_tensor = torch.as_tensor(planck, dtype=torch.float32, device=self.device)[None]
        i100_tensor = torch.as_tensor(
            planck * conversion, dtype=torch.float32, device=self.device
        )[None]
        self.targets: tuple[M07ScaleTarget, ...] = self._build_targets(
            planck_tensor, i100_tensor
        )
        self._target_tensors = tuple(
            torch.as_tensor(target.target_sigma_band, dtype=torch.float32, device=self.device)
            for target in self.targets
        )

    def _build_targets(
        self, planck_band: torch.Tensor, i100: torch.Tensor
    ) -> tuple[M07ScaleTarget, ...]:
        targets = []
        for j in self.config.scale_indices:
            scale_pixels = 2**int(j)
            context_pixels = int(round(self.config.context_factor * scale_pixels))
            stride_pixels = max(int(round(self.config.spacing_factor * scale_pixels)), 1)
            if context_pixels > min(self.shape):
                raise ValueError(f"j={j} context is larger than the synthesis image")
            local_band = _local_mean(planck_band, context_pixels, stride_pixels)[0]
            local_i100 = _local_mean(i100, context_pixels, stride_pixels)[0]
            local_band_np = local_band.detach().cpu().numpy()
            local_i100_np = local_i100.detach().cpu().numpy()
            if self.config.relation_color_mode == "global_median":
                # Keep the M07 envelope a single-valued, smooth function of
                # I100. Pixel-to-pixel T/beta color variations are used to
                # construct I100, but must not turn this relation into a
                # spatially zigzagging lookup table.
                band_over_i100: float | np.ndarray = self.relation_band_over_i100
            else:
                band_over_i100 = local_band_np / local_i100_np
            scale_arcmin = scale_pixels * self.pixel_scale_arcsec / 60.0
            outside = not (M07_MIN_SCALE_ARCMIN <= scale_arcmin <= M07_MAX_SCALE_ARCMIN)
            if outside and not self.config.extrapolate:
                raise ValueError(
                    f"j={j} ({scale_arcmin:g} arcmin) lies outside the M07 calibration range"
                )
            sigma = self.config.amplitude_correction * m07_sigma_lambda(
                local_i100_np, band_over_i100, scale_arcmin / 60.0
            )
            targets.append(
                M07ScaleTarget(
                    j=int(j),
                    scale_pixels=scale_pixels,
                    scale_arcmin=float(scale_arcmin),
                    context_pixels=context_pixels,
                    stride_pixels=stride_pixels,
                    local_band_mean=local_band_np,
                    local_I100=local_i100_np,
                    target_sigma_band=np.asarray(sigma, dtype=np.float32),
                    extrapolated=outside,
                )
            )
        return tuple(targets)

    def local_rms(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Measure differentiable local RMS maps for all configured scales."""

        if image.shape[-2:] != self.shape:
            raise ValueError(f"Image shape {image.shape[-2:]} does not match {self.shape}")
        results = []
        for target in self.targets:
            fine = _separable_gaussian(image, target.scale_pixels / 2.0)
            coarse = _separable_gaussian(image, float(target.scale_pixels))
            detail = fine - coarse
            local_first = _local_mean(
                detail, target.context_pixels, target.stride_pixels
            )
            local_second = _local_mean(
                detail.square(), target.context_pixels, target.stride_pixels
            )
            variance = (local_second - local_first.square()).clamp_min(0.0)
            results.append(torch.sqrt(variance + torch.finfo(image.dtype).eps))
        return tuple(results)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return one row of concatenated log(output RMS / M07 target RMS)."""

        residuals = []
        for measured, target in zip(self.local_rms(image), self._target_tensors, strict=True):
            residuals.append(torch.log(measured) - torch.log(target)[None])
        return torch.cat([value.reshape(value.shape[0], -1) for value in residuals], dim=1)

    def validation(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Measure output and target arrays for diagnostics without gradients."""

        tensor = torch.as_tensor(image, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad():
            measured = self.local_rms(tensor)
        result: dict[str, np.ndarray] = {}
        for target, output in zip(self.targets, measured, strict=True):
            prefix = f"j{target.j}"
            result[f"{prefix}_local_band_mean"] = target.local_band_mean
            result[f"{prefix}_local_I100"] = target.local_I100
            result[f"{prefix}_target_sigma"] = target.target_sigma_band
            result[f"{prefix}_output_sigma"] = output[0].detach().cpu().numpy()
        return result


def standardized_log_envelope_residual(
    measured_sigma: torch.Tensor,
    target_sigma: torch.Tensor,
    *,
    target_scale_floor_log: float = math.log1p(0.05),
) -> torch.Tensor:
    """Return centered local log-RMS residuals in target-contrast units.

    Centering removes the per-scale global amplitude, which is handled by
    :class:`M07LocalRMSConstraint`. The denominator is the target log-envelope
    standard deviation with a configurable floor, preventing nearly uniform
    Planck fields from producing an arbitrarily strong modulation loss.
    """

    measured_log = torch.log(measured_sigma)
    target_log = torch.log(target_sigma)
    measured_centered = measured_log - measured_log.mean(dim=(-2, -1), keepdim=True)
    target_centered = target_log - target_log.mean()
    target_scale = torch.std(target_log, unbiased=False).clamp_min(
        float(target_scale_floor_log)
    )
    return (measured_centered - target_centered[None]) / target_scale


class M07RelativeEnvelopeConstraint:
    """Emphasize spatial RMS modulation independently of absolute amplitude."""

    def __init__(
        self,
        rms_constraint: M07LocalRMSConstraint,
        *,
        target_scale_floor_log: float = math.log1p(0.05),
    ) -> None:
        if target_scale_floor_log <= 0.0:
            raise ValueError("target_scale_floor_log must be positive")
        self.rms_constraint = rms_constraint
        self.target_scale_floor_log = float(target_scale_floor_log)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        residuals = []
        for measured, target in zip(
            self.rms_constraint.local_rms(image),
            self.rms_constraint._target_tensors,
            strict=True,
        ):
            residual = standardized_log_envelope_residual(
                measured,
                target,
                target_scale_floor_log=self.target_scale_floor_log,
            )
            residuals.append(residual.reshape(residual.shape[0], -1))
        return torch.cat(residuals, dim=1)
