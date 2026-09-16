"""Radiance-conditioned scattering-mean targets from a small cutout ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from astropy.stats import mad_std

from .scattering_mean import FullMeanFeature, MeanFeature


Feature = MeanFeature | FullMeanFeature


@dataclass(frozen=True)
class ConditionalWSTFit:
    """Shrinkage log-linear WST target as a function of log Planck radiance."""

    intercept_log: np.ndarray
    slope_log_radiance: np.ndarray
    uncertainty_log: np.ndarray
    reference_log_radiance: float
    training_log_radiance: np.ndarray

    def predict(self, radiance: np.ndarray | float) -> np.ndarray:
        """Evaluate the extrapolating target at radiance values."""

        z = np.log(np.asarray(radiance, dtype=float)) - self.reference_log_radiance
        return self.intercept_log + np.asarray(z)[..., None] * self.slope_log_radiance


def _group_key(feature: Feature) -> tuple[int, int, int]:
    """Group coefficients by order and dyadic scale band for slope shrinkage."""

    return (feature.order, feature.j1, feature.j2)


def fit_conditional_wst_targets(
    values_log: np.ndarray,
    radiance: np.ndarray,
    features: tuple[Feature, ...],
    *,
    shrinkage: float = 0.5,
    uncertainty_floor_log: float = math.log1p(0.05),
) -> ConditionalWSTFit:
    """Fit per-coefficient trends with group-level slope shrinkage.

    Four cutouts cannot robustly determine an independent orientation-specific
    slope. Raw least-squares slopes are therefore blended with the median slope
    of their scattering order/scale band. The relation deliberately extrapolates
    when evaluated outside the training radiance range.
    """

    values = np.asarray(values_log, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    if values.ndim != 2 or values.shape != (radiance.size, len(features)):
        raise ValueError("values_log must have shape (cutouts, features)")
    if radiance.size < 3 or np.any(~np.isfinite(radiance)) or np.any(radiance <= 0.0):
        raise ValueError("at least three finite positive radiance values are required")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    x = np.log(radiance)
    reference = float(np.median(x))
    regression_centered = x - float(np.mean(x))
    denominator = float(np.sum(regression_centered**2))
    if denominator <= np.finfo(float).eps:
        raise ValueError("cutout radiance values have no usable spread")
    raw_slope = regression_centered @ (values - np.mean(values, axis=0)) / denominator
    group_slope: dict[tuple[int, int, int], float] = {}
    for key in {_group_key(feature) for feature in features}:
        indices = [index for index, feature in enumerate(features) if _group_key(feature) == key]
        group_slope[key] = float(np.median(raw_slope[indices]))
    shared = np.asarray([group_slope[_group_key(feature)] for feature in features])
    slope = (1.0 - shrinkage) * raw_slope + shrinkage * shared
    centered = x - reference
    intercept = np.median(values - centered[:, None] * slope[None], axis=0)
    residual = values - (intercept[None] + centered[:, None] * slope[None])
    uncertainty = np.maximum(
        np.asarray(mad_std(residual, axis=0, ignore_nan=True), dtype=float),
        float(uncertainty_floor_log),
    )
    return ConditionalWSTFit(
        intercept_log=intercept.astype(np.float32),
        slope_log_radiance=slope.astype(np.float32),
        uncertainty_log=uncertainty.astype(np.float32),
        reference_log_radiance=reference,
        training_log_radiance=x.astype(np.float32),
    )
