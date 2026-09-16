"""Hybrid scattering-mean targets across a Herschel/Planck transition scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridSMeanTarget:
    """Hybrid log S1/S2 target and its transition diagnostics."""

    log_s1: np.ndarray
    log_s2: np.ndarray
    s1_planck_offset: np.ndarray
    s2_planck_offset: np.ndarray
    cross_s2_slope: np.ndarray


def _pivoted_line(values: np.ndarray, scales: np.ndarray, pivot: int) -> tuple[float, float]:
    """Return slope and value at ``pivot`` for a finite one-dimensional fit."""

    values = np.asarray(values, dtype=float)
    scales = np.asarray(scales, dtype=float)
    keep = np.isfinite(values)
    if keep.sum() < 2:
        raise ValueError("At least two finite scales are needed for a linear fit")
    slope, intercept = np.polyfit(scales[keep] - pivot, values[keep], 1)
    return float(slope), float(intercept)


def build_hybrid_smean_target(
    herschel_s1: np.ndarray,
    herschel_s2: np.ndarray,
    planck_s1: np.ndarray,
    planck_s2: np.ndarray,
    *,
    transition_j: int,
    target_j: int,
    herschel_min_j: int = 0,
) -> HybridSMeanTarget:
    """Combine Herschel small scales with Planck large scales in log space.

    Planck S1 and Planck large-large S2 are offset to the observed Herschel
    coefficient at ``transition_j`` (used for calibration only, not as a
    small-scale target).  This makes every normalized Planck path continuous
    at the join.  Pivoted linear fits remain useful diagnostics of the local
    trend but are not allowed to introduce a normalization jump. Cross-scale
    paths are extrapolated from the Herschel S2 sequence in ``j2``.  A path
    with only one Herschel sample (the ``j1=transition_j-1`` case) borrows the
    orientation-specific median slope from the better-sampled paths, while
    remaining anchored to its observed Herschel coefficient.
    """

    h1, h2 = np.asarray(herschel_s1, float), np.asarray(herschel_s2, float)
    p1, p2 = np.asarray(planck_s1, float), np.asarray(planck_s2, float)
    if h1.ndim != 2 or p1.ndim != 2 or h2.ndim != 4 or p2.ndim != 4:
        raise ValueError("S1 must be (J,L) and S2 must be (J,J,L,L)")
    if not 1 <= transition_j < target_j <= p1.shape[0]:
        raise ValueError("transition_j must lie inside the Planck target scales")
    if not 0 <= herschel_min_j < transition_j:
        raise ValueError("herschel_min_j must lie in [0, transition_j)")
    if h1.shape[0] <= transition_j or h2.shape[0] <= transition_j:
        raise ValueError("Herschel input must include the transition calibration scale")
    if h1.shape[1] != p1.shape[1] or h2.shape[2:] != p2.shape[2:]:
        raise ValueError("Herschel and Planck orientations must agree")

    L = h1.shape[1]
    small = np.arange(herschel_min_j, transition_j)
    large = np.arange(transition_j, target_j)
    # The bundled direct estimator requires every native path. Paths below the
    # instrument-supported Herschel scale are intentionally retained from the
    # Planck initialization rather than treated as Herschel constraints.
    log_s1 = np.log(p1[:target_j]).copy()
    log_s2 = np.log(p2[:target_j, :target_j]).copy()
    log_s1[small] = np.log(h1[small])
    for j1 in small:
        for j2 in range(j1, transition_j):
            log_s2[j1, j2] = np.log(h2[j1, j2])

    s1_offset = np.empty(L)
    for l in range(L):
        s1_offset[l] = np.log(h1[transition_j, l]) - np.log(p1[transition_j, l])
    log_s1[transition_j:] = np.log(p1[transition_j:]) + s1_offset[None]

    s2_offset = np.empty((L, L))
    for l1 in range(L):
        for l2 in range(L):
            s2_offset[l1, l2] = (
                np.log(h2[transition_j, transition_j, l1, l2])
                - np.log(p2[transition_j, transition_j, l1, l2])
            )
            for j1 in large:
                for j2 in range(j1, target_j):
                    log_s2[j1, j2, l1, l2] = np.log(p2[j1, j2, l1, l2]) + s2_offset[l1, l2]

    cross_slope = np.empty((transition_j, L, L))
    for l1 in range(L):
        for l2 in range(L):
            fitted = []
            for j1 in range(herschel_min_j, transition_j - 1):
                scales = np.arange(j1, transition_j)
                slope, _ = _pivoted_line(np.log(h2[j1, scales, l1, l2]), scales, transition_j)
                fitted.append(slope)
                cross_slope[j1, l1, l2] = slope
            fallback = float(np.median(fitted))
            j1 = transition_j - 1
            cross_slope[j1, l1, l2] = fallback
            anchor = float(np.log(h2[j1, j1, l1, l2]))
            for j2 in large:
                log_s2[j1, j2, l1, l2] = anchor + fallback * (j2 - j1)
            for j1 in range(herschel_min_j, transition_j - 1):
                slope = cross_slope[j1, l1, l2]
                # Preserve the least-squares slope but anchor at the final
                # observed small-scale path. This prevents an unconstrained
                # intercept from creating a jump exactly at the stitch.
                anchor = float(np.log(h2[j1, transition_j - 1, l1, l2]))
                for j2 in large:
                    log_s2[j1, j2, l1, l2] = anchor + slope * (j2 - (transition_j - 1))
    return HybridSMeanTarget(log_s1, log_s2, s1_offset, s2_offset, cross_slope)


def flatten_bundled_smean(log_s1: np.ndarray, log_s2: np.ndarray) -> np.ndarray:
    """Flatten J,L log means using the bundled ``Scattering2d`` ordering."""

    J = log_s1.shape[0]
    valid = np.zeros(log_s2.shape[:2], dtype=bool)
    valid[np.triu_indices(J)] = True
    if np.any(~np.isfinite(log_s1)) or np.any(~np.isfinite(log_s2[valid])):
        raise ValueError("All native j2>=j1 target paths must be finite")
    return np.concatenate((log_s1.ravel(), log_s2[valid].ravel()))[None].astype(np.float32)


def bundled_smean_selection(
    J: int, L: int, *, transition_j: int, include_cross_s2: bool = True
) -> np.ndarray:
    """Select native bundled coefficients, optionally omitting cross-scale S2.

    The ordering matches :func:`flatten_bundled_smean` and the bundled
    ``Scattering2d.scattering_coef()['for_synthesis']`` vector.
    """

    selection = [True] * (J * L)
    for j1 in range(J):
        for j2 in range(j1, J):
            keep = include_cross_s2 or not (j1 < transition_j <= j2)
            selection.extend([keep] * (L * L))
    return np.asarray(selection, dtype=bool)
