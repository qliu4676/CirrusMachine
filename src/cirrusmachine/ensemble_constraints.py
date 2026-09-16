"""Small-ensemble targets and equal-footprint patch validation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from astropy.stats import mad_std

from .scattering_mean import FullMeanFeature, MeanFeature, ScatteringMean2D


Feature = MeanFeature | FullMeanFeature


def measure_mean_features(
    calculator: ScatteringMean2D,
    images: np.ndarray | torch.Tensor,
    features: tuple[Feature, ...],
    mode: str,
) -> torch.Tensor:
    """Dispatch to isotropic or orientation-resolved scattering means."""

    if mode == "isotropic":
        return calculator.log_features(images, features)  # type: ignore[arg-type]
    if mode == "full":
        return calculator.log_full_features(images, features)  # type: ignore[arg-type]
    raise ValueError(f"Unknown scattering-mean mode {mode!r}")


def robust_ensemble_target(
    images: np.ndarray,
    *,
    calculator: ScatteringMean2D,
    features: tuple[Feature, ...],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log-median, Astropy MAD standard deviation, and all measurements."""

    with torch.no_grad():
        values = measure_mean_features(calculator, images, features, mode).cpu().numpy()
    center = np.median(values, axis=0)
    uncertainty = np.asarray(mad_std(values, axis=0, ignore_nan=True), dtype=float)
    return center, uncertainty, values


def fixed_patch_grid(
    image_shape: tuple[int, int], patch_shape: tuple[int, int], count_per_axis: int = 3
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic evenly spaced validation grid."""

    ny, nx = image_shape
    ph, pw = patch_shape
    if ph > ny or pw > nx:
        raise ValueError("Patch shape exceeds image shape.")
    ys = np.rint(np.linspace(0, ny - ph, count_per_axis)).astype(int)
    xs = np.rint(np.linspace(0, nx - pw, count_per_axis)).astype(int)
    return tuple((int(y), int(x)) for y in ys for x in xs)


def extract_patch_stack(
    image: np.ndarray | torch.Tensor,
    coords: Iterable[tuple[int, int]],
    patch_shape: tuple[int, int],
) -> np.ndarray | torch.Tensor:
    """Extract equally shaped patches while preserving NumPy/Torch type."""

    ph, pw = patch_shape
    patches = [image[..., y : y + ph, x : x + pw] for y, x in coords]
    if isinstance(image, torch.Tensor):
        return torch.cat(patches, dim=0)
    return np.stack(patches, axis=0)

