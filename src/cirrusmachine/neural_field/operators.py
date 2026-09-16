"""Differentiable scale projection and safe Planck composition operators."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_f


def fourier_highpass_mask(
    shape: tuple[int, int],
    transition_pixels: float,
    *,
    width_fraction: float = 0.0,
    transition_range_pixels: tuple[float, float] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a radial high-pass mask with a transition at one spatial scale.

    ``width_fraction=0`` gives the initial single-cutoff experiment.  A later
    ablation can enable a raised-cosine transition without changing the API.
    """

    if transition_pixels <= 0:
        raise ValueError("transition_pixels must be positive")
    if not 0.0 <= width_fraction < 1.0:
        raise ValueError("width_fraction must lie in [0, 1)")
    fy = torch.fft.fftfreq(shape[0], device=device, dtype=dtype)[:, None]
    fx = torch.fft.fftfreq(shape[1], device=device, dtype=dtype)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    cutoff = 1.0 / float(transition_pixels)
    if transition_range_pixels is not None:
        fine, coarse = map(float, transition_range_pixels)
        if not 0 < fine < coarse:
            raise ValueError("transition_range_pixels requires 0 < fine < coarse")
        lower = 1.0 / coarse
        upper = 1.0 / fine
        phase = ((radius - lower) / (upper - lower)).clamp(0.0, 1.0)
        mask = 0.5 - 0.5 * torch.cos(math.pi * phase)
        mask = torch.where(radius <= lower, torch.zeros_like(mask), mask)
        mask = torch.where(radius >= upper, torch.ones_like(mask), mask)
    elif width_fraction == 0.0:
        mask = (radius >= cutoff).to(dtype)
    else:
        lower = cutoff * (1.0 - width_fraction)
        upper = cutoff * (1.0 + width_fraction)
        phase = ((radius - lower) / (upper - lower)).clamp(0.0, 1.0)
        mask = 0.5 - 0.5 * torch.cos(math.pi * phase)
        mask = torch.where(radius <= lower, torch.zeros_like(mask), mask)
        mask = torch.where(radius >= upper, torch.ones_like(mask), mask)
    mask[0, 0] = 0.0
    return mask


def apodization_window(
    shape: tuple[int, int],
    fraction: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a separable Tukey window for periodic-safe residual filtering."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Apodization fraction must lie in [0, 1]")
    if fraction == 0.0:
        return torch.ones(shape, device=device, dtype=dtype)

    def one_dimensional(size: int) -> torch.Tensor:
        coordinate = torch.arange(size, device=device, dtype=dtype)
        edge_distance = torch.minimum(coordinate, size - 1 - coordinate)
        taper = fraction * (size - 1) / 2.0
        phase = (edge_distance / max(taper, torch.finfo(dtype).eps)).clamp(0.0, 1.0)
        return torch.where(
            edge_distance < taper,
            0.5 - 0.5 * torch.cos(math.pi * phase),
            torch.ones_like(phase),
        )

    return one_dimensional(shape[0])[:, None] * one_dimensional(shape[1])[None, :]


def split_cosine_bell_window(
    shape: tuple[int, int],
    alpha: float,
    beta: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the radial SplitCosineBellWindow used by power diagnostics."""

    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("Split-cosine alpha and beta must lie in [0, 1]")
    y, x = torch.meshgrid(
        torch.arange(shape[0], device=device, dtype=dtype),
        torch.arange(shape[1], device=device, dtype=dtype),
        indexing="ij",
    )
    center_y = (shape[0] - 1.0) / 2.0
    center_x = (shape[1] - 1.0) / 2.0
    radius = torch.sqrt((y - center_y).square() + (x - center_x).square())
    half_size = (min(shape) - 1.0) / 2.0
    inner_radius = beta * half_size
    taper_width = int(math.floor(alpha * half_size))
    if taper_width == 0:
        return torch.ones(shape, device=device, dtype=dtype)
    window = 0.5 * (
        1.0 + torch.cos(math.pi * (radius - inner_radius) / taper_width)
    )
    window = torch.where(radius < inner_radius, torch.ones_like(window), window)
    return torch.where(radius > inner_radius + taper_width, torch.zeros_like(window), window)


def separable_split_cosine_window(
    shape: tuple[int, int], alpha: float, beta: float, *, device, dtype
) -> torch.Tensor:
    """Return a rectangular split-cosine window with a flat central plateau."""

    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("Split-cosine alpha and beta must lie in [0, 1]")

    def axis_window(size: int) -> torch.Tensor:
        coordinate = torch.arange(size, device=device, dtype=dtype)
        normalized = torch.abs(2.0 * coordinate / (size - 1.0) - 1.0)
        width = max(float(alpha), float(torch.finfo(dtype).eps))
        phase = ((normalized - beta) / width).clamp(0.0, 1.0)
        window = 0.5 + 0.5 * torch.cos(math.pi * phase)
        return torch.where(normalized <= beta, torch.ones_like(window), window)

    return axis_window(shape[0])[:, None] * axis_window(shape[1])[None, :]


def apodize_final_residual(
    residual: torch.Tensor, *, alpha: float, beta: float, mode: str = "radial"
) -> torch.Tensor:
    """Taper a delivered residual to zero edges while preserving zero mean."""

    if alpha == 0.0:
        return residual - residual.mean(dim=(-2, -1), keepdim=True)
    factory = split_cosine_bell_window if mode == "radial" else separable_split_cosine_window
    window = factory(tuple(residual.shape[-2:]), alpha, beta, device=residual.device, dtype=residual.dtype)
    denominator = window.sum().clamp_min(torch.finfo(residual.dtype).eps)
    offset = (residual * window).sum(dim=(-2, -1), keepdim=True) / denominator
    return (residual - offset) * window


def gaussian_resolution_transfer(
    shape: tuple[int, int], fwhm_pixels: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return the Fourier transfer of a circular Gaussian resolution limit."""

    if fwhm_pixels <= 0:
        return torch.ones(shape, device=device, dtype=dtype)
    sigma = float(fwhm_pixels) / 2.354820045
    fy = torch.fft.fftfreq(shape[0], device=device, dtype=dtype)[:, None]
    fx = torch.fft.fftfreq(shape[1], device=device, dtype=dtype)[None, :]
    return torch.exp(-2.0 * math.pi**2 * sigma**2 * (fx.square() + fy.square()))


def project_residual(
    residual: torch.Tensor,
    *,
    transition_pixels: float,
    transition_width_fraction: float = 0.0,
    transition_range_pixels: tuple[float, float] | None = None,
    apodization_fraction: float = 0.0,
    boundary_padding_pixels: int = 0,
    resolution_fwhm_pixels: float | None = None,
) -> torch.Tensor:
    """Remove DC/Planck modes and optionally enforce the Herschel resolution."""

    original_shape = tuple(residual.shape[-2:])
    padding = int(boundary_padding_pixels)
    if padding < 0 or padding >= min(original_shape):
        raise ValueError("Boundary padding must be smaller than each image dimension")
    working = residual.float()
    if padding:
        leading = working.shape[:-2]
        working = working.reshape(-1, 1, *original_shape)
        working = torch_f.pad(
            working,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        working = working.reshape(*leading, *working.shape[-2:])
    shape = tuple(working.shape[-2:])
    window = apodization_window(
        shape,
        apodization_fraction,
        device=residual.device,
        dtype=torch.float32,
    )
    transformed = torch.fft.fftn(working * window, dim=(-2, -1))
    transfer = fourier_highpass_mask(
        shape,
        transition_pixels,
        width_fraction=transition_width_fraction,
        transition_range_pixels=transition_range_pixels,
        device=residual.device,
        dtype=torch.float32,
    )
    if resolution_fwhm_pixels is not None:
        transfer = transfer * gaussian_resolution_transfer(
            shape,
            resolution_fwhm_pixels,
            device=residual.device,
            dtype=torch.float32,
        )
    # Keep filtering in float32 even when decoder evaluation used autocast.
    # FFT cutoffs, SWD sorting, and MD07 RMS are deliberately not half precision.
    projected = torch.fft.ifftn(transformed * transfer, dim=(-2, -1)).real
    if padding:
        projected = projected[..., padding:-padding, padding:-padding]
    # Cropping a reflection-padded filtered field can introduce a tiny DC
    # offset; remove it exactly before adding the unchanged Planck backbone.
    return projected - projected.mean(dim=(-2, -1), keepdim=True)


def compose_with_backbone(
    backbone: torch.Tensor,
    residual: torch.Tensor,
    *,
    transition_pixels: float,
    transition_width_fraction: float = 0.0,
    transition_range_pixels: tuple[float, float] | None = None,
    apodization_fraction: float = 0.0,
    boundary_padding_pixels: int = 0,
    boundary_taper_fraction: float = 0.0,
    complementary_backbone: bool = False,
    resolution_fwhm_pixels: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose Planck and texture with optional power-complementary blending.

    With ``complementary_backbone=False`` this retains the legacy additive
    form.  The complementary form treats the raised-cosine mask as a *power*
    fraction: Planck and texture amplitudes receive ``sqrt(1-alpha)`` and
    ``sqrt(alpha)`` respectively.  Multiplication by these real non-negative
    transfers preserves the phase of every retained coefficient.
    """

    if backbone.shape != residual.shape:
        raise ValueError("backbone and residual must have identical shapes")
    if not 0.0 <= boundary_taper_fraction <= 1.0:
        raise ValueError("boundary_taper_fraction must lie in [0, 1]")
    if not complementary_backbone:
        delta = project_residual(
            residual,
            transition_pixels=transition_pixels,
            transition_width_fraction=transition_width_fraction,
            transition_range_pixels=transition_range_pixels,
            apodization_fraction=apodization_fraction,
            boundary_padding_pixels=boundary_padding_pixels,
            resolution_fwhm_pixels=resolution_fwhm_pixels,
        )
    else:
        original_shape = tuple(backbone.shape[-2:])
        padding = int(boundary_padding_pixels)
        if padding < 0 or padding >= min(original_shape):
            raise ValueError("Boundary padding must be smaller than each image dimension")

        def padded(value: torch.Tensor) -> torch.Tensor:
            working = value.float()
            if not padding:
                return working
            leading = working.shape[:-2]
            working = working.reshape(-1, 1, *original_shape)
            working = torch_f.pad(
                working, (padding, padding, padding, padding), mode="reflect"
            )
            return working.reshape(*leading, *working.shape[-2:])

        planck_working = padded(backbone)
        texture_working = padded(residual)
        shape = tuple(planck_working.shape[-2:])
        alpha = fourier_highpass_mask(
            shape,
            transition_pixels,
            width_fraction=transition_width_fraction,
            transition_range_pixels=transition_range_pixels,
            device=backbone.device,
            dtype=torch.float32,
        )
        planck_transfer = torch.sqrt((1.0 - alpha).clamp_min(0.0))
        texture_transfer = torch.sqrt(alpha.clamp_min(0.0))
        if resolution_fwhm_pixels is not None:
            texture_transfer = texture_transfer * gaussian_resolution_transfer(
                shape,
                resolution_fwhm_pixels,
                device=backbone.device,
                dtype=torch.float32,
            )
        window = apodization_window(
            shape,
            apodization_fraction,
            device=backbone.device,
            dtype=torch.float32,
        )
        planck_fft = torch.fft.fftn(planck_working * window, dim=(-2, -1))
        texture_fft = torch.fft.fftn(texture_working * window, dim=(-2, -1))
        synthesis = torch.fft.ifftn(
            planck_fft * planck_transfer + texture_fft * texture_transfer,
            dim=(-2, -1),
        ).real
        if padding:
            synthesis = synthesis[..., padding:-padding, padding:-padding]
        delta = synthesis - backbone

    if boundary_taper_fraction > 0.0:
        envelope = apodization_window(
            tuple(delta.shape[-2:]),
            boundary_taper_fraction,
            device=delta.device,
            dtype=delta.dtype,
        )
        # Choose the pre-envelope offset so the tapered delta has exactly zero
        # mean while its outermost pixels meet the unchanged Planck backbone.
        offset = (delta * envelope).sum(dim=(-2, -1), keepdim=True) / envelope.sum()
        delta = (delta - offset) * envelope
    else:
        delta = delta - delta.mean(dim=(-2, -1), keepdim=True)
    return backbone + delta, delta


def low_frequency_residual(
    synthesis: torch.Tensor,
    backbone: torch.Tensor,
    transition_pixels: float,
    transition_range_pixels: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Measure leakage below the configured transition without beam convolution."""

    delta = synthesis - backbone
    shape = tuple(delta.shape[-2:])
    fy = torch.fft.fftfreq(shape[0], device=delta.device, dtype=torch.float32)[:, None]
    fx = torch.fft.fftfreq(shape[1], device=delta.device, dtype=torch.float32)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    if transition_range_pixels is not None:
        strict_cutoff = 1.0 / float(transition_range_pixels[1])
    else:
        strict_cutoff = (1.0 / float(transition_pixels)) * (1.0 - 0.0)
    mask = (radius < strict_cutoff).to(torch.float32)
    return torch.fft.ifftn(
        torch.fft.fftn(delta.float(), dim=(-2, -1)) * mask,
        dim=(-2, -1),
    ).real
