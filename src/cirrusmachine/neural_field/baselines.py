"""Non-neural baselines for controlled A2390 comparisons."""

from __future__ import annotations

import numpy as np
import torch

from .operators import project_residual


def random_phase_herschel_baseline(
    backbone: np.ndarray,
    exemplar: np.ndarray,
    *,
    transition_pixels: float,
    resolution_fwhm_pixels: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add random-phase high frequencies with the tiled exemplar amplitude spectrum.

    This is deliberately a baseline, not the proposed synthesis method.  The
    exemplar is periodically extended only to estimate a same-shape Fourier
    amplitude; all phases are randomized before the approved projection.
    """

    base = np.asarray(backbone, dtype=np.float32)
    texture = np.asarray(exemplar, dtype=np.float32)
    repeats = (
        int(np.ceil(base.shape[0] / texture.shape[0])),
        int(np.ceil(base.shape[1] / texture.shape[1])),
    )
    tiled = np.tile(texture - np.mean(texture), repeats)[: base.shape[0], : base.shape[1]]
    amplitude = np.abs(np.fft.rfft2(tiled))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=amplitude.shape)
    randomized = np.fft.irfft2(amplitude * np.exp(1j * phase), s=base.shape).real
    projected = project_residual(
        torch.as_tensor(randomized, dtype=torch.float32),
        transition_pixels=transition_pixels,
        resolution_fwhm_pixels=resolution_fwhm_pixels,
    ).numpy()
    exemplar_projected = project_residual(
        torch.as_tensor(texture - np.mean(texture), dtype=torch.float32),
        transition_pixels=transition_pixels,
        resolution_fwhm_pixels=resolution_fwhm_pixels,
    ).numpy()
    target_rms = float(np.sqrt(np.mean(exemplar_projected**2)))
    current_rms = max(float(np.sqrt(np.mean(projected**2))), np.finfo(float).eps)
    projected *= target_rms / current_rms
    return base + projected, projected
