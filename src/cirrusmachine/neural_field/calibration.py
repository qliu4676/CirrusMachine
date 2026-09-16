"""Adapters around the existing M07 calibration implementation."""

from __future__ import annotations

import numpy as np

from cirrusmachine.m07_modulation import M07LocalRMSConstraint, M07ModulationConfig
from cirrusmachine.planck_dust import modified_blackbody_band_to_I100


def band_to_i100(temperature: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return the existing Planck-MBB 250-to-100 micron conversion map."""

    return modified_blackbody_band_to_I100(temperature, beta, 250.0).astype(np.float32)


def fit_md07_amplitude_correction(
    exemplar: np.ndarray,
    exemplar_planck: np.ndarray,
    conversion: float | np.ndarray,
    pixel_scale_arcsec: float,
    *,
    scale_indices: tuple[int, ...] = (4, 5, 6),
    context_factor: float = 3.0,
    spacing_factor: float = 0.5,
    extrapolate: bool = True,
) -> tuple[float, dict[str, float]]:
    """Fit one robust multiplicative correction from log RMS ratios."""

    logs = []
    metrics: dict[str, float] = {}
    used = []
    for j in scale_indices:
        context_pixels = int(round(context_factor * 2**int(j)))
        if context_pixels > min(exemplar.shape):
            metrics[f"j{j}_fit_skipped_context_too_large"] = 1.0
            continue
        descriptor = M07LocalRMSConstraint(
            exemplar_planck,
            pixel_scale_arcsec,
            M07ModulationConfig(
                band_to_I100=conversion,
                amplitude_correction=1.0,
                scale_indices=(int(j),),
                context_factor=context_factor,
                spacing_factor=spacing_factor,
                extrapolate=extrapolate,
            ),
            device="cpu",
        )
        validation = descriptor.validation(np.asarray(exemplar, dtype=np.float32))
        target = descriptor.targets[0]
        prefix = f"j{target.j}"
        measured = validation[f"{prefix}_output_sigma"]
        expected = validation[f"{prefix}_target_sigma"]
        log_ratio = np.log(measured / expected)
        finite = np.isfinite(log_ratio)
        logs.append(log_ratio[finite])
        used.append(target.j)
        metrics[f"{prefix}_median_ratio"] = float(np.exp(np.median(log_ratio[finite])))
        metrics[f"{prefix}_extrapolated"] = float(target.extrapolated)
    if not logs:
        raise ValueError("The exemplar is too small to fit any configured MD07 scale")
    correction = float(np.exp(np.median(np.concatenate(logs))))
    metrics["fitted_correction"] = correction
    metrics["fit_scale_count"] = float(len(used))
    return correction, metrics
