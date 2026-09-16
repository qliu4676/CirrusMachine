"""Local isotropic phase-harmonic correlations between adjacent scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from astropy.stats import mad_std

from .scattering_mean import ScatteringMean2D


class IsotropicPhaseHarmonic2D:
    """Measure normalized ``H1(W_j) x H2(W_j+1)`` correlations.

    The second phase harmonic aligns the coarse wavelet phase frequency with
    the adjacent finer wavelet. Real and imaginary parts are retained after
    averaging over orientations, making the descriptor rotation-averaged but
    phase-sensitive.
    """

    def __init__(
        self,
        calculator: ScatteringMean2D,
        scale_pairs: tuple[tuple[int, int], ...],
        *,
        boundary_sigma: float = 2.0,
    ) -> None:
        self.calculator = calculator
        self.scale_pairs = tuple((int(a), int(b)) for a, b in scale_pairs)
        if not self.scale_pairs or any(b != a + 1 for a, b in self.scale_pairs):
            raise ValueError("Phase-harmonic scale pairs must be adjacent.")
        largest = max(b for _, b in self.scale_pairs)
        if largest >= calculator.J:
            raise ValueError("Phase-harmonic scale lies outside calculator J.")
        self.margin = int(np.ceil(boundary_sigma * 0.8 * 2**largest))
        if 2 * self.margin >= min(calculator.shape):
            raise ValueError("Phase-harmonic scales leave no valid patch interior.")

    def features(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return real/imaginary normalized correlations per input image."""

        data = self.calculator._prepare(images)
        transformed = torch.fft.fftn(data, dim=(-2, -1))
        needed = sorted({scale for pair in self.scale_pairs for scale in pair})
        p = self.calculator.pad_pixels
        ny, nx = self.calculator.shape
        m = self.margin
        responses = {}
        for scale in needed:
            response = torch.fft.ifftn(
                transformed[:, None] * self.calculator.filters[scale][None],
                dim=(-2, -1),
            )
            responses[scale] = response[:, :, p + m : p + ny - m, p + m : p + nx - m]

        values = []
        epsilon = torch.finfo(data.dtype).eps
        for fine_scale, coarse_scale in self.scale_pairs:
            fine = responses[fine_scale]
            coarse = responses[coarse_scale]
            coarse_h2 = coarse.square() / coarse.abs().clamp_min(epsilon)
            correlation = (fine * coarse_h2.conj()).mean(dim=(-2, -1))
            normalization = torch.sqrt(
                fine.abs().square().mean(dim=(-2, -1))
                * coarse_h2.abs().square().mean(dim=(-2, -1))
            ).clamp_min(epsilon)
            isotropic = (correlation / normalization).mean(dim=1)
            values.extend((isotropic.real, isotropic.imag))
        return torch.stack(values, dim=-1)


def phase_harmonic_ensemble_target(features: np.ndarray) -> dict[str, np.ndarray]:
    """Return robust center, MAD scale, variance, and all cutout values."""

    values = np.asarray(features, dtype=float)
    center = np.median(values, axis=0)
    scale = np.asarray(mad_std(values, axis=0, ignore_nan=True), dtype=float)
    scale = np.maximum(scale, 0.05)
    return {
        "all": values.astype(np.float32),
        "center": center.astype(np.float32),
        "mad_std": scale.astype(np.float32),
        "variance": np.var(values, axis=0, ddof=1).astype(np.float32),
    }


@dataclass
class RandomPatchPhaseHarmonics:
    """Uncertainty-standardized phase harmonics on random output patches."""

    descriptor: IsotropicPhaseHarmonic2D
    center: np.ndarray
    scale: np.ndarray
    patch_shape: tuple[int, int] = (150, 150)
    batch_size: int = 4
    seed: int = 4681

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
        values = self.descriptor.features(torch.cat(patches, dim=0))
        center = torch.as_tensor(self.center, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return (values - center) / scale
