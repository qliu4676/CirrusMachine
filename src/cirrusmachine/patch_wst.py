"""Patch-wise WST optimization helpers.

These utilities keep the notebook close to the optimization style used by the
bundled ``scattering.synthesis_general`` function: optimize a learnable delta
around a fixed base image, build a deterministic closure for each optimizer
step, then call ``loss.backward()`` inside that closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PatchBatch:
    """Patch coordinates for one optimizer step."""

    coords: tuple[tuple[int, int], ...]


class OptimizablePatchImage(torch.nn.Module):
    """Image model that learns a delta around a fixed backbone."""

    def __init__(self, base_image: torch.Tensor, delta_init: torch.Tensor, delta_scale: float = 0.3):
        super().__init__()
        self.register_buffer("base_image", base_image)
        self.param = torch.nn.Parameter(delta_init)
        self.delta_scale = float(delta_scale)

    @property
    def image(self) -> torch.Tensor:
        return self.base_image + self.delta_scale * self.param


def make_patch_anchors(
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    stride_fraction: float = 0.5,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return regular overlapping patch anchors and their stride."""

    ny, nx = image_shape
    ph, pw = patch_shape
    stride_y = max(int(round(ph * stride_fraction)), 1)
    stride_x = max(int(round(pw * stride_fraction)), 1)
    ys = list(range(0, max(ny - ph + 1, 1), stride_y))
    xs = list(range(0, max(nx - pw + 1, 1), stride_x))
    if ys[-1] != max(ny - ph, 0):
        ys.append(max(ny - ph, 0))
    if xs[-1] != max(nx - pw, 0):
        xs.append(max(nx - pw, 0))
    return [(y, x) for y in ys for x in xs], (stride_y, stride_x)


def jitter_anchor(
    anchor: tuple[int, int],
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    stride: tuple[int, int],
    jitter_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Jitter one anchor while keeping the patch inside the image."""

    y, x = anchor
    ny, nx = image_shape
    ph, pw = patch_shape
    jy = int(round(stride[0] * jitter_fraction))
    jx = int(round(stride[1] * jitter_fraction))
    if jy > 0:
        y += int(rng.integers(-jy, jy + 1))
    if jx > 0:
        x += int(rng.integers(-jx, jx + 1))
    y = int(np.clip(y, 0, max(ny - ph, 0)))
    x = int(np.clip(x, 0, max(nx - pw, 0)))
    return y, x


def make_epoch_batches(
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    batch_size: int,
    stride_fraction: float,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[PatchBatch], int]:
    """Create one shuffled, jittered epoch of patch batches."""

    anchors, stride = make_patch_anchors(image_shape, patch_shape, stride_fraction)
    order = rng.permutation(len(anchors))
    batches: list[PatchBatch] = []
    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        coords = tuple(
            jitter_anchor(anchors[int(i)], image_shape, patch_shape, stride, jitter_fraction, rng)
            for i in ids
        )
        batches.append(PatchBatch(coords=coords))
    return batches, len(anchors)


def lowpass_torch(image_stack: torch.Tensor, cutoff_pixels: float) -> torch.Tensor:
    """Gaussian Fourier low-pass for tensors shaped ``(N, M, N)``."""

    m, n = image_stack.shape[-2:]
    fy = torch.fft.fftfreq(m, d=1.0, device=image_stack.device)[:, None]
    fx = torch.fft.fftfreq(n, d=1.0, device=image_stack.device)[None, :]
    radius = torch.sqrt(fy**2 + fx**2)
    cutoff_freq = 1.0 / float(cutoff_pixels)
    filt = torch.exp(-0.5 * (radius / cutoff_freq) ** 2)
    return torch.fft.ifftn(
        torch.fft.fftn(image_stack, dim=(-2, -1)) * filt[None, :, :],
        dim=(-2, -1),
    ).real


def extract_patches(image: torch.Tensor, coords: Iterable[tuple[int, int]], patch_shape: tuple[int, int]) -> torch.Tensor:
    """Extract a batch of patches from a tensor shaped ``(1, M, N)``."""

    ph, pw = patch_shape
    return torch.cat([image[:, y : y + ph, x : x + pw] for y, x in coords], dim=0)


def gaussian_smooth_torch(image_stack: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    """Gaussian Fourier smoothing for tensors shaped ``(..., M, N)``."""

    sigma_pixels = float(sigma_pixels)
    if sigma_pixels <= 0:
        return image_stack
    m, n = image_stack.shape[-2:]
    fy = torch.fft.fftfreq(m, d=1.0, device=image_stack.device)[:, None]
    fx = torch.fft.fftfreq(n, d=1.0, device=image_stack.device)[None, :]
    ksq = fx**2 + fy**2
    filt = torch.exp(-2.0 * (math.pi**2) * (sigma_pixels**2) * ksq)
    while filt.ndim < image_stack.ndim:
        filt = filt[None, :, :]
    return torch.fft.ifftn(
        torch.fft.fftn(image_stack, dim=(-2, -1)) * filt,
        dim=(-2, -1),
    ).real


def edge_weight_from_window(window: torch.Tensor) -> torch.Tensor:
    """Return a normalized edge weight from an apodization window."""

    edge = 1.0 - window
    denom = torch.mean(edge**2).clamp_min(1e-12)
    return edge / torch.sqrt(denom)


def optimize_patch_wst_delta(
    image_init_stack: np.ndarray,
    hybrid_patch_target: torch.Tensor,
    *,
    scattering_module: Any,
    J: int,
    L: int,
    coef_key: str,
    use_iso: bool,
    patch_shape: tuple[int, int],
    window: np.ndarray,
    n_epochs: int = 3,
    batch_size: int = 2,
    stride_fraction: float = 0.5,
    jitter_fraction: float = 0.25,
    learning_rate: float = 0.01,
    lowpass_weight: float = 0.1,
    close_weight: float = 1e-5,
    patch_mean_weight: float = 1.0,
    patch_smooth_weight: float = 0.0,
    patch_smooth_sigma_pixels: float = 1.0,
    patch_edge_weight: float = 0.0,
    lowpass_cutoff_pixels: float | None = None,
    delta_scale: float = 0.3,
    delta_init_std: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    progress_factory: Callable[[Iterable[int], str], Iterable[int]] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], list[tuple[int, int, np.ndarray]]]:
    """Optimize a delta around a fixed image backbone using patch WST losses."""

    image_np = np.asarray(image_init_stack, dtype=np.float32)
    if image_np.ndim != 3 or image_np.shape[0] != 1:
        raise ValueError(f"Expected image_init_stack with shape (1, M, N), got {image_np.shape}")
    _, ny, nx = image_np.shape
    ph, pw = patch_shape
    if ph > ny or pw > nx:
        raise ValueError(f"Patch shape {patch_shape} is larger than image shape {(ny, nx)}")
    if lowpass_cutoff_pixels is None:
        lowpass_cutoff_pixels = 0.8 * min(patch_shape)

    torch_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    base = torch.from_numpy(image_np).float().to(torch_device)
    if delta_init_std > 0:
        generator = torch.Generator(device=torch_device)
        generator.manual_seed(seed)
        delta_init = delta_init_std * base.std() * torch.randn(base.shape, generator=generator, device=torch_device)
    else:
        delta_init = torch.zeros_like(base)

    image_model = OptimizablePatchImage(base, delta_init, delta_scale=delta_scale).to(torch_device)
    optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)

    target = hybrid_patch_target.float().to(torch_device)
    window_t = torch.from_numpy(np.asarray(window, dtype=np.float32))[None, :, :].to(torch_device)
    edge_weight_t = edge_weight_from_window(window_t)
    init_low = lowpass_torch(base, lowpass_cutoff_pixels).detach()
    st_calc_patch = scattering_module.Scattering2d(ph, pw, J, L, device=device)
    st_calc_patch.add_synthesis_P00(s_cov=target, if_iso=use_iso)

    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    snapshots: list[tuple[int, int, np.ndarray]] = []

    for epoch in range(n_epochs):
        batches, n_anchors = make_epoch_batches(
            (ny, nx), patch_shape, batch_size, stride_fraction, jitter_fraction, rng
        )
        iterator: Iterable[int] = range(len(batches))
        if progress_factory is not None:
            iterator = progress_factory(iterator, f"epoch {epoch + 1}/{n_epochs}")

        for step_in_epoch in iterator:
            batch = batches[step_in_epoch]

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                image = image_model.image
                patches_raw = extract_patches(image, batch.coords, patch_shape)
                base_patches = extract_patches(base, batch.coords, patch_shape)
                delta_patches = patches_raw - base_patches
                patches = patches_raw * window_t
                coef = st_calc_patch.scattering_cov(
                    patches,
                    use_ref=True,
                    if_large_batch=False,
                    normalization="P00",
                )[coef_key]
                wst_loss = ((coef - target) ** 2).mean() * 1e8
                low_loss = lowpass_weight * ((lowpass_torch(image, lowpass_cutoff_pixels) - init_low) ** 2).mean()
                close_loss = close_weight * ((image - base) ** 2).mean()
                mean_loss = patch_mean_weight * ((patches_raw.mean(dim=(-2, -1)) - base_patches.mean(dim=(-2, -1))) ** 2).mean()
                if patch_smooth_weight > 0:
                    smooth_delta = gaussian_smooth_torch(delta_patches, patch_smooth_sigma_pixels)
                    smooth_loss = patch_smooth_weight * ((delta_patches - smooth_delta) ** 2).mean()
                else:
                    smooth_loss = torch.zeros((), device=torch_device)
                if patch_edge_weight > 0:
                    edge_loss = patch_edge_weight * ((delta_patches * edge_weight_t) ** 2).mean()
                else:
                    edge_loss = torch.zeros((), device=torch_device)
                loss = wst_loss + low_loss + close_loss + mean_loss + smooth_loss + edge_loss
                loss.backward()
                closure.last = {
                    "epoch": float(epoch),
                    "step_in_epoch": float(step_in_epoch),
                    "n_anchors": float(n_anchors),
                    "loss": float(loss.detach().cpu()),
                    "wst_loss": float(wst_loss.detach().cpu()),
                    "lowpass_loss": float(low_loss.detach().cpu()),
                    "close_loss": float(close_loss.detach().cpu()),
                    "patch_mean_loss": float(mean_loss.detach().cpu()),
                    "patch_smooth_loss": float(smooth_loss.detach().cpu()),
                    "patch_edge_loss": float(edge_loss.detach().cpu()),
                }
                return loss

            closure.last = {}
            optimizer.step(closure)
            row = dict(closure.last)
            history.append(row)

            if callback is not None:
                callback(
                    {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "num_steps_in_epoch": len(batches),
                        "history_row": row,
                        "image": image_model.image.detach().cpu().numpy()[0],
                    }
                )

        snapshots.append((epoch, len(batches) - 1, image_model.image.detach().cpu().numpy()[0].copy()))

    return image_model.image.detach().cpu().numpy()[0], history, snapshots
