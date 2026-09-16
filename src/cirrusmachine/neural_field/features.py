"""Raw-patch and orientation-resolved Morlet feature definitions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from cirrusmachine.scattering_mean import ScatteringMean2D


def extract_patch_vectors(image: torch.Tensor, origins: np.ndarray, patch_size: int) -> torch.Tensor:
    """Return mean-centered signed flattened patches with gradients intact."""

    if image.ndim == 2:
        image = image[None]
    size = int(patch_size)
    patches = [
        image[:, int(y) : int(y) + size, int(x) : int(x) + size]
        for y, x in np.asarray(origins)
    ]
    stacked = torch.cat(patches, dim=0)
    stacked = stacked - stacked.mean(dim=(-2, -1), keepdim=True)
    return stacked.reshape(stacked.shape[0], -1)


def dihedral_augment_patches(patches: torch.Tensor) -> torch.Tensor:
    """Apply a balanced set of rotations/reflections to a patch batch.

    Transform IDs cycle through the eight symmetries of a square.  This keeps
    interpolation out of the target construction and makes the empirical
    orientation distribution isotropic under 90-degree rotations/reflections.
    """

    if patches.ndim != 3 or patches.shape[-2] != patches.shape[-1]:
        raise ValueError("Dihedral augmentation expects square (batch, y, x) patches")
    transformed = []
    for index, patch in enumerate(patches):
        transform = index % 8
        value = torch.rot90(patch, transform % 4, dims=(-2, -1))
        if transform >= 4:
            value = torch.flip(value, dims=(-1,))
        transformed.append(value)
    return torch.stack(transformed)


@dataclass(frozen=True)
class RobustNormalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, floor_fraction: float = 1.0e-3) -> "RobustNormalizer":
        values = np.asarray(features, dtype=np.float64)
        center = np.median(values, axis=0)
        scale = 1.4826022185 * np.median(np.abs(values - center), axis=0)
        floor = max(float(np.median(scale)) * floor_fraction, np.finfo(np.float32).eps)
        return cls(center.astype(np.float32), np.maximum(scale, floor).astype(np.float32))

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(self.center, device=features.device, dtype=features.dtype)
        scale = torch.as_tensor(self.scale, device=features.device, dtype=features.dtype)
        return (features - center) / scale


class OrientedMorletFeatures:
    """Spatial samples of orientation-resolved log Morlet amplitudes."""

    def __init__(
        self,
        calculator: ScatteringMean2D,
        scales: tuple[int, ...],
        *,
        spatial_stride: int = 2,
        boundary_sigma: float = 2.0,
        orientation_pool: bool = False,
    ) -> None:
        self.calculator = calculator
        self.scales = tuple(int(value) for value in scales)
        self.spatial_stride = int(spatial_stride)
        self.orientation_pool = bool(orientation_pool)
        if not self.scales or min(self.scales) < 0 or max(self.scales) >= calculator.J:
            raise ValueError("Morlet scales must lie within the calculator")
        self.margin = int(math.ceil(boundary_sigma * 0.8 * 2 ** max(self.scales)))
        if 2 * self.margin >= min(calculator.shape):
            raise ValueError("Morlet boundary margin leaves no valid interior")

    def features(
        self,
        images: np.ndarray | torch.Tensor,
        *,
        center_mask: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = self.calculator._prepare(images)
        transformed = torch.fft.fftn(data, dim=(-2, -1))
        p = self.calculator.pad_pixels
        ny, nx = self.calculator.shape
        m = self.margin
        rows = []
        for scale in self.scales:
            response = torch.fft.ifftn(
                transformed[:, None] * self.calculator.filters[scale][None],
                dim=(-2, -1),
            ).abs()
            if self.orientation_pool:
                response = response.mean(dim=1, keepdim=True)
            interior = response[
                :,
                :,
                p + m : p + ny - m : self.spatial_stride,
                p + m : p + nx - m : self.spatial_stride,
            ]
            rows.append(torch.log(interior.clamp_min(torch.finfo(interior.dtype).tiny)))
        stacked = torch.cat(rows, dim=1).permute(0, 2, 3, 1)
        feature_count = stacked.shape[-1]
        if center_mask is None:
            return stacked.reshape(-1, feature_count)
        mask = torch.as_tensor(center_mask, device=stacked.device, dtype=torch.bool)
        mask = mask[
            m : ny - m : self.spatial_stride,
            m : nx - m : self.spatial_stride,
        ]
        return stacked[:, mask].reshape(-1, feature_count)


def subsample_rows(features: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """Choose feature rows without replacement when possible."""

    if features.shape[0] <= count:
        return features
    indices = torch.randperm(features.shape[0], generator=generator, device=features.device)[:count]
    return features[indices]


def log_oriented_wavelet_energy(log_amplitudes: torch.Tensor) -> torch.Tensor:
    """Return log mean-square Morlet response for every scale/orientation.

    ``log_amplitudes`` contains the unstandardized log absolute responses
    returned by :class:`OrientedMorletFeatures`.  Computing the energy in log
    space is stable and preserves the physical band-power normalization that
    robust feature standardization deliberately removes.
    """

    if log_amplitudes.ndim != 2 or log_amplitudes.shape[0] < 1:
        raise ValueError("Wavelet energy expects a non-empty (sample, component) array")
    return torch.logsumexp(2.0 * log_amplitudes, dim=0) - math.log(
        log_amplitudes.shape[0]
    )


def log_wavelet_energy_by_scale(
    log_amplitudes: torch.Tensor, *, scales: int, orientations: int
) -> torch.Tensor:
    """Return absolute log Morlet energy separately for every scale."""

    expected = int(scales) * int(orientations)
    if log_amplitudes.ndim != 2 or log_amplitudes.shape[1] != expected:
        raise ValueError("Morlet feature count does not match scales*orientations")
    values = log_amplitudes.reshape(
        log_amplitudes.shape[0], int(scales), int(orientations)
    )
    return torch.logsumexp(2.0 * values, dim=(0, 2)) - math.log(
        values.shape[0] * values.shape[2]
    )


def log_cross_scale_envelope_coupling(
    log_amplitudes: torch.Tensor,
    *,
    scales: int,
    orientations: int,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Measure normalized fine/coarse Morlet-energy co-location.

    For each spatial response center, orientations are pooled into a local
    energy envelope. The returned log coupling is zero for independent
    envelopes and positive when fine energy preferentially occupies regions
    with strong coarse-scale energy.
    """

    expected = int(scales) * int(orientations)
    if log_amplitudes.ndim != 2 or log_amplitudes.shape[1] != expected:
        raise ValueError("Morlet feature count does not match scales*orientations")
    values = log_amplitudes.reshape(
        log_amplitudes.shape[0], int(scales), int(orientations)
    )
    log_envelope = torch.logsumexp(2.0 * values, dim=2) - math.log(
        values.shape[2]
    )
    results = []
    count_log = math.log(log_envelope.shape[0])
    for fine, coarse in pairs:
        if not 0 <= fine < scales or not 0 <= coarse < scales or fine >= coarse:
            raise ValueError("Cross-scale pairs require 0 <= fine < coarse < scales")
        joint = (
            torch.logsumexp(log_envelope[:, fine] + log_envelope[:, coarse], dim=0)
            - count_log
        )
        fine_mean = torch.logsumexp(log_envelope[:, fine], dim=0) - count_log
        coarse_mean = torch.logsumexp(log_envelope[:, coarse], dim=0) - count_log
        results.append(joint - fine_mean - coarse_mean)
    return torch.stack(results)


def log_aligned_cross_scale_coupling(
    log_amplitudes: torch.Tensor,
    *,
    scales: int,
    orientations: int,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Return same-orientation fine/coarse energy coupling."""

    expected = int(scales) * int(orientations)
    if log_amplitudes.ndim != 2 or log_amplitudes.shape[1] != expected:
        raise ValueError("Morlet feature count does not match scales*orientations")
    values = 2.0 * log_amplitudes.reshape(
        log_amplitudes.shape[0], int(scales), int(orientations)
    )
    sample_log = math.log(values.shape[0])
    sample_orientation_log = math.log(values.shape[0] * values.shape[2])
    results = []
    for fine, coarse in pairs:
        if not 0 <= fine < scales or not 0 <= coarse < scales or fine >= coarse:
            raise ValueError("Cross-scale pairs require 0 <= fine < coarse < scales")
        log_joint = torch.logsumexp(
            values[:, fine] + values[:, coarse], dim=(0, 1)
        ) - sample_orientation_log
        log_fine_by_orientation = torch.logsumexp(
            values[:, fine], dim=0
        ) - sample_log
        log_coarse_by_orientation = torch.logsumexp(
            values[:, coarse], dim=0
        ) - sample_log
        log_independent = torch.logsumexp(
            log_fine_by_orientation + log_coarse_by_orientation, dim=0
        ) - math.log(values.shape[2])
        results.append(log_joint - log_independent)
    return torch.stack(results)


def correlation_floor_weights(
    log_coupling: torch.Tensor,
    *,
    eta: float,
    maximum_uniform_multiple: float,
) -> torch.Tensor:
    """Weight strong couplings while retaining a nonzero uniform floor."""

    count = log_coupling.numel()
    if count < 1:
        raise ValueError("At least one coupling is required")
    uniform = 1.0 / count
    strength = log_coupling.detach().abs()
    if float(strength.sum()) == 0.0:
        return torch.full_like(strength, uniform)
    weights = (1.0 - float(eta)) * uniform + float(eta) * strength / strength.sum()
    cap = float(maximum_uniform_multiple) * uniform
    for _ in range(count + 1):
        over = weights > cap
        if not torch.any(over):
            break
        excess = (weights[over] - cap).sum()
        weights = torch.where(over, torch.full_like(weights, cap), weights)
        under = ~over
        if not torch.any(under):
            break
        weights = weights + under * (excess / under.sum())
    return weights / weights.sum()


def radial_log_power_features(
    patches: torch.Tensor,
    *,
    bins: int,
    minimum_frequency: float,
    maximum_frequency: float,
) -> torch.Tensor:
    """Return differentiable apodized log radial power for equal-size patches."""

    if patches.ndim == 2:
        patches = patches[None]
    if patches.ndim != 3:
        raise ValueError("Power features expect (batch, height, width)")
    if not 0 < minimum_frequency < maximum_frequency <= 0.5:
        raise ValueError("Power-frequency limits must satisfy 0 < min < max <= 0.5")
    height, width = patches.shape[-2:]
    wy = torch.hann_window(height, periodic=False, device=patches.device, dtype=patches.dtype)
    wx = torch.hann_window(width, periodic=False, device=patches.device, dtype=patches.dtype)
    window = wy[:, None] * wx[None, :]
    centered = patches - patches.mean(dim=(-2, -1), keepdim=True)
    transformed = torch.fft.rfftn(centered * window, dim=(-2, -1))
    power = transformed.abs().square() / window.square().sum().clamp_min(1.0)
    fy = torch.fft.fftfreq(height, device=patches.device, dtype=patches.dtype)[:, None]
    fx = torch.fft.rfftfreq(width, device=patches.device, dtype=patches.dtype)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    edges = torch.logspace(
        math.log10(minimum_frequency),
        math.log10(maximum_frequency),
        int(bins) + 1,
        device=patches.device,
        dtype=patches.dtype,
    )
    values = []
    for index in range(int(bins)):
        include = (radius >= edges[index]) & (radius < edges[index + 1])
        if not torch.any(include):
            raise ValueError("A radial power bin contains no Fourier samples")
        values.append(torch.log(power[:, include].mean(dim=1).clamp_min(torch.finfo(power.dtype).tiny)))
    return torch.stack(values, dim=-1)


def angular_power_anisotropy(
    patches: torch.Tensor,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> torch.Tensor:
    """Return the normalized quadrupole of patch power in a frequency band."""

    if patches.ndim == 2:
        patches = patches[None]
    if patches.ndim != 3:
        raise ValueError("Anisotropy expects (batch, height, width)")
    if not 0 < minimum_frequency < maximum_frequency <= 0.5:
        raise ValueError("Anisotropy-frequency limits must lie in (0, 0.5]")
    height, width = patches.shape[-2:]
    wy = torch.hann_window(height, periodic=False, device=patches.device, dtype=patches.dtype)
    wx = torch.hann_window(width, periodic=False, device=patches.device, dtype=patches.dtype)
    window = wy[:, None] * wx[None, :]
    centered = patches - patches.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fftn(centered * window, dim=(-2, -1)).abs().square()
    fy = torch.fft.fftfreq(height, device=patches.device, dtype=patches.dtype)[:, None]
    fx = torch.fft.fftfreq(width, device=patches.device, dtype=patches.dtype)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    include = (radius >= minimum_frequency) & (radius < maximum_frequency)
    if not torch.any(include):
        raise ValueError("The anisotropy band contains no Fourier samples")
    angle = torch.atan2(fy.expand_as(radius), fx.expand_as(radius))
    basis_real = torch.cos(2.0 * angle[include])
    basis_imag = torch.sin(2.0 * angle[include])
    selected = power[:, include]
    denominator = selected.sum(dim=1).clamp_min(torch.finfo(power.dtype).tiny)
    real = (selected * basis_real).sum(dim=1) / denominator
    imaginary = (selected * basis_imag).sum(dim=1) / denominator
    return torch.sqrt(real.square() + imaginary.square()).clamp(0.0, 1.0)


def radial_log_power_physical(
    images: torch.Tensor,
    *,
    pixel_scale_arcmin: float,
    frequency_edges: torch.Tensor,
    window_centers: tuple[tuple[float, float], ...] = ((0.5, 0.5),),
) -> torch.Tensor:
    """Differentiable radial log power in shared physical-frequency bins."""

    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3:
        raise ValueError("Global power expects (batch, height, width)")
    height, width = images.shape[-2:]
    # Match ``power_spectrum_2d`` exactly: a radial split-cosine bell with
    # beta=0.5 and alpha=0.1.  Using a different training window can hide a
    # transition bump that reappears in the published diagnostic.
    y = torch.arange(height, device=images.device, dtype=images.dtype)
    x = torch.arange(width, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    npts = (min(height, width) - 1.0) / 2.0
    inner = 0.5 * npts
    taper = max(int(math.floor(0.1 * npts)), 1)
    centered = images - images.mean(dim=(-2, -1), keepdim=True)
    fy = torch.fft.fftfreq(
        height, d=float(pixel_scale_arcmin), device=images.device, dtype=images.dtype
    )[:, None]
    fx = torch.fft.fftfreq(
        width, d=float(pixel_scale_arcmin), device=images.device, dtype=images.dtype
    )[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    edges = frequency_edges.to(device=images.device, dtype=images.dtype)
    all_windows = []
    for center_y, center_x in window_centers:
        cy = float(center_y) * (height - 1.0)
        cx = float(center_x) * (width - 1.0)
        distance = torch.sqrt((yy - cy).square() + (xx - cx).square())
        window = 0.5 * (1.0 + torch.cos(torch.pi * (distance - inner) / taper))
        window = torch.where(distance < inner, torch.ones_like(window), window)
        window = torch.where(distance > inner + taper, torch.zeros_like(window), window)
        power = (
            torch.fft.fftn(centered * window, dim=(-2, -1)).abs().square()
            / window.square().sum().clamp_min(1.0)
        )
        values = []
        for index in range(len(edges) - 1):
            include = (radius >= edges[index]) & (radius < edges[index + 1])
            if not torch.any(include):
                raise ValueError("A physical radial-power bin contains no Fourier samples")
            values.append(
                torch.log(
                    power[:, include]
                    .mean(dim=1)
                    .clamp_min(torch.finfo(power.dtype).tiny)
                )
            )
        all_windows.append(torch.stack(values, dim=-1))
    return torch.cat(all_windows, dim=0)


def boundary_gradient_excess(image: torch.Tensor, *, width: int = 8) -> torch.Tensor:
    """Penalize only excess normal gradients near the four output borders."""

    if image.ndim != 2:
        raise ValueError("Boundary gradient loss expects one 2D field")
    band = int(width)
    if band < 1 or 2 * band + 2 >= min(image.shape):
        raise ValueError("Boundary width is incompatible with the image shape")
    gx = image[:, 1:] - image[:, :-1]
    gy = image[1:, :] - image[:-1, :]
    edge_power = torch.cat(
        (
            gx[:, :band].reshape(-1),
            gx[:, -band:].reshape(-1),
            gy[:band].reshape(-1),
            gy[-band:].reshape(-1),
        )
    ).square().mean()
    interior_power = 0.5 * (
        gx[band:-band, band:-band].square().mean()
        + gy[band:-band, band:-band].square().mean()
    )
    ratio = edge_power / interior_power.detach().clamp_min(torch.finfo(image.dtype).tiny)
    return torch.relu(ratio - 1.0).square()
