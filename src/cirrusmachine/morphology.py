"""Topology diagnostics for possible loop-like synthesis artifacts."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from scipy.ndimage import label


def robust_standardize(image: np.ndarray) -> np.ndarray:
    """Center by the median and scale by Astropy's MAD standard deviation."""

    image = np.asarray(image, dtype=float)
    center = float(np.nanmedian(image))
    scale = float(mad_std(image, ignore_nan=True))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(image))
    return (image - center) / max(scale, np.finfo(float).eps)


def euler_characteristic(binary: np.ndarray) -> int:
    """Return foreground components minus enclosed background holes."""

    binary = np.asarray(binary, dtype=bool)
    _, components = label(binary, structure=np.ones((3, 3), dtype=int))
    background_labels, background_count = label(
        ~binary, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    if background_count == 0:
        holes = 0
    else:
        boundary = np.unique(
            np.concatenate(
                (
                    background_labels[0],
                    background_labels[-1],
                    background_labels[:, 0],
                    background_labels[:, -1],
                )
            )
        )
        exterior = set(int(value) for value in boundary if value > 0)
        holes = background_count - len(exterior)
    return int(components - holes)


def perimeter_area_ratio(binary: np.ndarray) -> float:
    """Return the four-neighbour excursion perimeter divided by its area."""

    binary = np.asarray(binary, dtype=np.int8)
    area = int(binary.sum())
    if area == 0:
        return np.nan
    padded = np.pad(binary, 1)
    perimeter = np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum()
    return float(perimeter / area)


def morphology_curves(
    images: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure Euler characteristic and perimeter/area over standardized levels."""

    stack = np.asarray(images, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    if thresholds is None:
        thresholds = np.linspace(-1.5, 1.5, 13)
    euler = np.empty((len(stack), len(thresholds)), dtype=float)
    perimeter_area = np.empty_like(euler)
    for index, image in enumerate(stack):
        standardized = robust_standardize(image)
        for level, threshold in enumerate(thresholds):
            excursion = standardized >= threshold
            euler[index, level] = euler_characteristic(excursion)
            perimeter_area[index, level] = perimeter_area_ratio(excursion)
    return np.asarray(thresholds), euler, perimeter_area


def plot_morphology_comparison(
    herschel: np.ndarray,
    treatments: Mapping[str, np.ndarray],
    path: str,
) -> dict[str, dict[str, float]]:
    """Plot ensemble envelopes and return compact deviation scores."""

    thresholds, euler_ref, perimeter_ref = morphology_curves(herschel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    scores: dict[str, dict[str, float]] = {}
    references = ((axes[0], euler_ref, "Euler characteristic"),
                  (axes[1], perimeter_ref, "perimeter / area"))
    for ax, reference, ylabel in references:
        median = np.nanmedian(reference, axis=0)
        lower = np.nanmin(reference, axis=0)
        upper = np.nanmax(reference, axis=0)
        ax.fill_between(thresholds, lower, upper, color="0.8", label="Herschel range")
        ax.plot(thresholds, median, color="black", lw=2, label="Herschel median")
        ax.set(xlabel="threshold [robust σ]", ylabel=ylabel)
        ax.axhline(0, color="0.6", lw=0.6)
    for name, patches in treatments.items():
        _, euler, perimeter = morphology_curves(patches, thresholds)
        axes[0].plot(thresholds, np.nanmedian(euler, axis=0), label=name)
        axes[1].plot(thresholds, np.nanmedian(perimeter, axis=0), label=name)
        euler_scale = np.maximum(np.nanmax(euler_ref, axis=0) - np.nanmin(euler_ref, axis=0), 1.0)
        perimeter_scale = np.maximum(
            np.nanmax(perimeter_ref, axis=0) - np.nanmin(perimeter_ref, axis=0), 0.05
        )
        scores[name] = {
            "euler_envelope_distance": float(
                np.nanmean(np.abs(np.nanmedian(euler, axis=0) - np.nanmedian(euler_ref, axis=0)) / euler_scale)
            ),
            "perimeter_envelope_distance": float(
                np.nanmean(
                    np.abs(np.nanmedian(perimeter, axis=0) - np.nanmedian(perimeter_ref, axis=0))
                    / perimeter_scale
                )
            ),
        }
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return scores
