"""Local isotropic wavelet-amplitude distributions for texture synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from astropy.stats import mad_std

from .scattering_mean import ScatteringMean2D


class IsotropicLocalAmplitude2D:
    """Return spatial samples of orientation-averaged log Morlet amplitudes."""

    def __init__(
        self,
        calculator: ScatteringMean2D,
        scales: tuple[int, ...],
        *,
        boundary_sigma: float = 2.0,
        spatial_stride: int = 2,
    ) -> None:
        self.calculator = calculator
        self.scales = tuple(int(scale) for scale in scales)
        self.spatial_stride = int(spatial_stride)
        if not self.scales or min(self.scales) < 0 or max(self.scales) >= calculator.J:
            raise ValueError("Local-amplitude scales must lie inside the calculator J range.")
        if self.spatial_stride < 1:
            raise ValueError("spatial_stride must be positive.")
        self.margin = int(math.ceil(boundary_sigma * 0.8 * 2 ** max(self.scales)))
        if 2 * self.margin >= min(calculator.shape):
            raise ValueError("Largest local-amplitude scale leaves no valid patch interior.")

    def features(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return ``(spatial samples, scales)`` log-amplitude feature vectors."""

        data = self.calculator._prepare(images)
        transformed = torch.fft.fftn(data, dim=(-2, -1))
        p = self.calculator.pad_pixels
        ny, nx = self.calculator.shape
        margin = self.margin
        amplitudes = []
        for scale in self.scales:
            response = torch.fft.ifftn(
                transformed[:, None] * self.calculator.filters[scale][None],
                dim=(-2, -1),
            ).abs().mean(dim=1)
            interior = response[
                :,
                p + margin : p + ny - margin : self.spatial_stride,
                p + margin : p + nx - margin : self.spatial_stride,
            ]
            amplitudes.append(torch.log(interior.clamp_min(torch.finfo(interior.dtype).tiny)))
        stacked = torch.stack(amplitudes, dim=-1)
        return stacked.reshape(-1, len(self.scales))


def fixed_random_projections(feature_count: int, count: int, seed: int) -> np.ndarray:
    """Return reproducible unit vectors for sliced-Wasserstein projections."""

    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, feature_count))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32)


def amplitude_distribution_target(
    features: np.ndarray,
    projections: np.ndarray,
    quantile_count: int,
) -> dict[str, np.ndarray]:
    """Build robust normalization and projected target quantiles."""

    features = np.asarray(features, dtype=float)
    center = np.median(features, axis=0)
    scale = np.asarray(mad_std(features, axis=0, ignore_nan=True), dtype=float)
    scale = np.maximum(scale, np.finfo(float).eps)
    standardized = (features - center) / scale
    projected = standardized @ np.asarray(projections, dtype=float).T
    levels = np.linspace(0.0, 1.0, int(quantile_count), dtype=float)
    quantiles = np.quantile(projected, levels, axis=0)
    return {
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "projections": np.asarray(projections, dtype=np.float32),
        "quantile_levels": levels.astype(np.float32),
        "target_quantiles": quantiles.astype(np.float32),
        "feature_variance": np.var(features, axis=0, ddof=1).astype(np.float32),
    }


def projected_amplitude_quantiles(
    descriptor: IsotropicLocalAmplitude2D,
    images: np.ndarray | torch.Tensor,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    projections: np.ndarray,
    quantile_levels: np.ndarray,
) -> torch.Tensor:
    """Measure standardized projected local-amplitude quantiles."""

    features = descriptor.features(images)
    center_tensor = torch.as_tensor(center, dtype=features.dtype, device=features.device)
    scale_tensor = torch.as_tensor(scale, dtype=features.dtype, device=features.device)
    projection_tensor = torch.as_tensor(
        projections, dtype=features.dtype, device=features.device
    )
    level_tensor = torch.as_tensor(
        quantile_levels, dtype=features.dtype, device=features.device
    )
    projected = ((features - center_tensor) / scale_tensor) @ projection_tensor.T
    return torch.quantile(projected, level_tensor, dim=0)


@dataclass
class RandomPatchAmplitudeQuantiles:
    """Differentiable projected quantiles from random equal-footprint patches."""

    descriptor: IsotropicLocalAmplitude2D
    center: np.ndarray
    scale: np.ndarray
    projections: np.ndarray
    quantile_levels: np.ndarray
    patch_shape: tuple[int, int] = (150, 150)
    batch_size: int = 4
    seed: int = 4678

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        ny, nx = image.shape[-2:]
        ph, pw = self.patch_shape
        patches = []
        for _ in range(self.batch_size):
            y = int(self.rng.integers(0, ny - ph + 1))
            x = int(self.rng.integers(0, nx - pw + 1))
            patches.append(image[:, y : y + ph, x : x + pw])
        quantiles = projected_amplitude_quantiles(
            self.descriptor,
            torch.cat(patches, dim=0),
            center=self.center,
            scale=self.scale,
            projections=self.projections,
            quantile_levels=self.quantile_levels,
        )
        return quantiles.reshape(1, -1)
