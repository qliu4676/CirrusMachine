"""Sparse, coverage-aware cutout selection for cirrus synthesis inputs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def largest_connected_area(mask: np.ndarray) -> int:
    """Return the pixel area of the largest 8-connected true region."""
    labels, count = label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    if count == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


def select_sparse_cutouts(
    image: np.ndarray,
    cutout_shape: tuple[int, int],
    count: int,
    max_nan_fraction: float = 0.01,
    min_separation_pixels: float = 0.0,
    seed: int = 4676,
    max_attempts: int = 1000,
) -> list[tuple[slice, slice]]:
    """Randomly select valid cutouts with separated centers.

    Candidate windows are required to have a total NaN fraction no larger than
    ``max_nan_fraction``. This is stricter than imposing the same limit only on
    the largest connected NaN region.
    """
    image = np.asarray(image)
    ny, nx = image.shape
    height, width = map(int, cutout_shape)
    if height > ny or width > nx:
        raise ValueError("cutout shape exceeds image shape")

    bad = (~np.isfinite(image)).astype(np.int32)
    integral = np.pad(bad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0 = np.arange(ny - height + 1)
    x0 = np.arange(nx - width + 1)
    y1 = y0 + height
    x1 = x0 + width
    nan_counts = (
        integral[y1[:, None], x1]
        - integral[y0[:, None], x1]
        - integral[y1[:, None], x0]
        + integral[y0[:, None], x0]
    )
    eligible = np.argwhere(nan_counts <= max_nan_fraction * height * width)
    if eligible.size == 0:
        raise ValueError("no cutout satisfies the NaN-fraction limit")
    centers = np.column_stack(
        (eligible[:, 0] + (height - 1) / 2.0, eligible[:, 1] + (width - 1) / 2.0)
    )

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        chosen = [int(rng.integers(len(eligible)))]
        while len(chosen) < count:
            distances = np.min(
                np.linalg.norm(
                    centers[:, None, :] - centers[np.asarray(chosen)][None, :, :], axis=2
                ),
                axis=1,
            )
            available = np.flatnonzero(distances >= min_separation_pixels)
            available = available[~np.isin(available, chosen)]
            if available.size == 0:
                break
            # Retain randomness while favouring the more widely separated half.
            cutoff = np.percentile(distances[available], 50.0)
            pool = available[distances[available] >= cutoff]
            chosen.append(int(rng.choice(pool)))
        if len(chosen) == count:
            slices = []
            for index in chosen:
                start_y, start_x = eligible[index]
                slices.append(
                    (slice(int(start_y), int(start_y + height)), slice(int(start_x), int(start_x + width)))
                )
            return slices
    raise ValueError(
        f"could not place {count} cutouts at minimum separation {min_separation_pixels:.1f} pixels"
    )


def fill_nan_pixels_maskfill(image: np.ndarray, size: int = 9) -> np.ndarray:
    """Fill NaN pixels using maskfill's median operator and smoothing."""
    from maskfill import maskfill

    image = np.asarray(image, dtype=float)
    nan_mask = ~np.isfinite(image)
    if not nan_mask.any():
        return image.copy()
    filled, _ = maskfill(
        image,
        nan_mask.astype(np.uint8),
        size=size,
        operator="median",
        smooth=True,
        verbose=False,
    )
    filled = np.asarray(filled, dtype=float)
    if not np.isfinite(filled).all():
        raise ValueError("maskfill did not fill every NaN pixel")
    return filled
