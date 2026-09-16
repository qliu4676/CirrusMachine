"""Diagnostics for the generic single-image Morlet-SWD experiment."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from cirrusmachine.power_spectrum import power_spectrum_2d, radial_power_spectrum
from cirrusmachine.scattering_mean import MeanFeature, ScatteringMean2D

from .diagnostics import boundary_metrics, nearest_patch_metrics
from .features import (
    OrientedMorletFeatures,
    RobustNormalizer,
    correlation_floor_weights,
    extract_patch_vectors,
    log_aligned_cross_scale_coupling,
    log_cross_scale_envelope_coupling,
    log_wavelet_energy_by_scale,
    subsample_rows,
)
from .losses import sliced_wasserstein_loss
from .single_image import SingleImageConfig, morlet_margin


def _morlet_response_maps(
    image: np.ndarray | torch.Tensor,
    calculator,
    scales: tuple[int, ...],
) -> np.ndarray:
    """Return log-modulus maps as ``(scale, orientation, y, x)``."""

    data = calculator._prepare(image)
    transformed = torch.fft.fftn(data, dim=(-2, -1))
    p = calculator.pad_pixels
    ny, nx = calculator.shape
    rows = []
    tiny = torch.finfo(data.dtype).tiny
    for scale in scales:
        response = torch.fft.ifftn(
            transformed[:, None] * calculator.filters[scale][None],
            dim=(-2, -1),
        ).abs()
        response = response[0, :, p : p + ny, p : p + nx]
        rows.append(torch.log(response.clamp_min(tiny)))
    return torch.stack(rows).detach().float().cpu().numpy()


def _plot_canvas(
    path: Path,
    input_image: np.ndarray,
    synthesis: np.ndarray,
    restored_canvas: np.ndarray,
    crop: tuple[slice, slice],
    label: str,
) -> None:
    limits = np.percentile(
        np.concatenate((input_image.ravel(), synthesis.ravel())), (0.1, 99.9)
    )
    canvas_limits = np.percentile(restored_canvas, (0.1, 99.9))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (input_image, label, limits),
        (restored_canvas, "Full neural-field canvas", canvas_limits),
        (synthesis, "Delivered central crop", limits),
    )
    for axis, (image, title, bounds) in zip(axes, panels, strict=True):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="viridis",
            vmin=bounds[0],
            vmax=bounds[1],
        )
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, shrink=0.76)
    yslice, xslice = crop
    axes[1].add_patch(
        Rectangle(
            (xslice.start - 0.5, yslice.start - 0.5),
            xslice.stop - xslice.start,
            yslice.stop - yslice.start,
            fill=False,
            edgecolor="white",
            linewidth=2.0,
        )
    )
    axes[1].text(
        xslice.start + 3,
        yslice.stop - max(32, int(0.08 * (yslice.stop - yslice.start))),
        f"delivered {xslice.stop - xslice.start}×{yslice.stop - yslice.start} crop",
        color="white",
        fontsize=9,
        weight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_history(path: Path, history: list[dict[str, float]], scales) -> None:
    step = np.asarray([row["step"] for row in history]) + 1
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].semilogy(step, [row["loss"] for row in history], color="black")
    axes[0, 0].set(title="Joint Morlet-SWD", ylabel="Loss")
    for scale in scales:
        axes[0, 1].semilogy(
            step,
            [row[f"morlet_j{scale}"] for row in history],
            label=f"j={scale}",
        )
    axes[0, 1].set(title="Diagnostic per-scale SWD", ylabel="Loss")
    axes[0, 1].legend()
    axes[1, 0].semilogy(step, [row["gradient_norm"] for row in history])
    axes[1, 0].set(title="Unclipped gradient norm", xlabel="Step", ylabel="Norm")
    axes[1, 1].plot(step, [row["crop_rms"] for row in history])
    axes[1, 1].set(
        title="Centered crop amplitude", xlabel="Step", ylabel="RMS"
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        if axis in axes[0]:
            axis.set_xlabel("Step")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_constraint_components(
    path: Path,
    history: list[dict[str, float]],
    config: SingleImageConfig,
    target_features: torch.Tensor,
    output_features: torch.Tensor,
) -> None:
    """Show each physical/statistical contribution and its resolved scales."""

    scales = config.morlet.training_scales
    step = np.asarray([row["step"] for row in history]) + 1
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    contributions = (
        ("Morlet-SWD", "morlet_loss", 1.0),
        (
            "Absolute wavelet energy",
            "wavelet_power_loss",
            config.wavelet_power.weight,
        ),
        ("Cross-scale concentration", "cross_scale_loss", config.cross_scale.weight),
        ("Radial power", "power_loss", config.power.weight),
    )
    for label, key, weight in contributions:
        axes[0, 0].semilogy(
            step,
            np.maximum(weight * np.asarray([row[key] for row in history]), 1.0e-12),
            label=f"{label} (weighted)",
        )
    axes[0, 0].set(
        title="Components of total training constraint",
        xlabel="Step",
        ylabel="Weighted loss",
    )
    axes[0, 0].legend(fontsize=8)

    for scale in scales:
        axes[0, 1].semilogy(
            step,
            np.maximum(
                [row[f"wavelet_power_j{scale}"] for row in history], 1.0e-12
            ),
            label=f"j={scale}",
        )
    axes[0, 1].set(
        title="Absolute Morlet-energy error by scale",
        xlabel="Step",
        ylabel="Huber loss",
    )
    axes[0, 1].legend(ncol=2, fontsize=8)

    with torch.no_grad():
        target_energy = log_wavelet_energy_by_scale(
            target_features,
            scales=len(scales),
            orientations=config.morlet.orientations,
        ).cpu()
        output_energy = log_wavelet_energy_by_scale(
            output_features,
            scales=len(scales),
            orientations=config.morlet.orientations,
        ).cpu()
    axes[1, 0].plot(scales, target_energy, "o-", label="Input")
    axes[1, 0].plot(scales, output_energy, "s-", label="Synthesis")
    axes[1, 0].set(
        title="Absolute log Morlet energy",
        xlabel="Morlet scale j",
        ylabel=r"$\log\langle|I\star\psi_j|^2\rangle$",
    )
    axes[1, 0].legend()

    labels = config.cross_scale.pairs or tuple(
        (fine, coarse)
        for fine in config.cross_scale.fine_scales
        for coarse in config.cross_scale.coarse_scales
    )
    if labels:
        index = {scale: position for position, scale in enumerate(scales)}
        pairs = tuple((index[fine], index[coarse]) for fine, coarse in labels)
        with torch.no_grad():
            coupling_function = (
                log_aligned_cross_scale_coupling
                if config.cross_scale.aligned_orientations
                else log_cross_scale_envelope_coupling
            )
            target_coupling = torch.exp(
                coupling_function(
                    target_features,
                    scales=len(scales),
                    orientations=config.morlet.orientations,
                    pairs=pairs,
                )
            ).cpu()
            output_coupling = torch.exp(
                coupling_function(
                    output_features,
                    scales=len(scales),
                    orientations=config.morlet.orientations,
                    pairs=pairs,
                )
            ).cpu()
        position = np.arange(len(labels))
        axes[1, 1].plot(position, target_coupling, "o-", label="Input")
        axes[1, 1].plot(position, output_coupling, "s-", label="Synthesis")
        axes[1, 1].set_xticks(
            position, [f"{fine}→{coarse}" for fine, coarse in labels], rotation=60
        )
        axes[1, 1].set(
            title="Fine/coarse energy concentration",
            xlabel="fine j → coarse j",
            ylabel="Normalized coupling",
        )
        axes[1, 1].legend()
    else:
        axes[1, 1].axis("off")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_fbm_extrapolation(path: Path, info: dict) -> None:
    """Visualize trusted-scale fitting and the hybrid radial target."""

    scales = np.asarray(info["scales"])
    trusted = set(np.asarray(info["trusted_scales"]).tolist())
    replaced = set(np.asarray(info["replaced_scales"]).tolist())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].errorbar(
        scales,
        info["observed_log_energy"],
        yerr=info["observed_log_energy_std"],
        marker="o",
        label="Dragonfly measured",
    )
    axes[0].errorbar(
        scales,
        info["fbm_log_energy"],
        yerr=info["fbm_log_energy_std"],
        marker="s",
        label="Fitted anisotropic fBm",
    )
    for scale in scales:
        if scale in trusted:
            axes[0].axvspan(scale - 0.18, scale + 0.18, color="green", alpha=0.12)
        if scale in replaced:
            axes[0].axvspan(scale - 0.18, scale + 0.18, color="red", alpha=0.10)
    axes[0].set(
        title=(
            f"fBm slope={info['slope']:g}; spectral PA="
            f"{info['spectral_angle_degrees']:.1f}°; q₂={info['quadrupole']:.3f}"
        ),
        xlabel="Morlet scale j",
        ylabel="Absolute log Morlet energy",
    )
    axes[0].legend()

    frequency = np.asarray(info["power_frequency_arcmin_inverse"])
    axes[1].plot(frequency, info["observed_log_power"], label="Observed target")
    axes[1].plot(frequency, info["fbm_log_power"], label="Fitted fBm")
    axes[1].plot(
        frequency, info["hybrid_log_power"], linewidth=2.5, label="Hybrid target"
    )
    axes[1].set_xscale("log")
    axes[1].set(
        title="Smooth observed/fBm radial-power blend",
        xlabel=r"Spatial frequency [arcmin$^{-1}$]",
        ylabel="Log power",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_response_maps(
    path: Path,
    maps: np.ndarray,
    scales: tuple[int, ...],
    title: str,
    *,
    accepted_mask: np.ndarray | None = None,
    crop: tuple[slice, slice] | None = None,
    excluded_scales: tuple[int, ...] = (),
) -> None:
    orientations = maps.shape[1]
    figure, axes = plt.subplots(
        len(scales),
        orientations,
        figsize=(3.1 * orientations, 2.8 * len(scales)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(scales), orientations)
    for row, scale in enumerate(scales):
        for orientation in range(orientations):
            image = maps[row, orientation]
            if crop is not None:
                image = image[crop]
            bounds = np.percentile(image[np.isfinite(image)], (1.0, 99.0))
            shown = axes[row, orientation].imshow(
                image,
                origin="lower",
                cmap="magma",
                vmin=bounds[0],
                vmax=bounds[1],
            )
            state = "excluded" if scale in excluded_scales else "trained"
            axes[row, orientation].set_title(
                f"j={scale}, orientation={orientation} ({state})", fontsize=8
            )
            axes[row, orientation].set_xticks([])
            axes[row, orientation].set_yticks([])
            if accepted_mask is not None:
                ys, xs = np.where(accepted_mask)
                axes[row, orientation].add_patch(
                    Rectangle(
                        (xs.min() - 0.5, ys.min() - 0.5),
                        xs.max() - xs.min() + 1,
                        ys.max() - ys.min() + 1,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.2,
                    )
                )
            figure.colorbar(shown, ax=axes[row, orientation], shrink=0.65)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_response_distributions(
    path: Path,
    input_maps: np.ndarray,
    output_maps: np.ndarray,
    target_mask: np.ndarray,
    crop: tuple[slice, slice],
    scales: tuple[int, ...],
    excluded_scales: tuple[int, ...],
) -> None:
    columns = min(3, len(scales))
    rows = math.ceil(len(scales) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    quantiles = np.linspace(0.01, 0.99, 99)
    for row, scale in enumerate(scales):
        axis = axes.ravel()[row]
        for orientation in range(input_maps.shape[1]):
            input_values = input_maps[row, orientation][target_mask]
            output_values = output_maps[row, orientation][crop].ravel()
            axis.plot(
                np.quantile(input_values, quantiles),
                np.quantile(output_values, quantiles),
                label=f"orientation {orientation}",
                alpha=0.8,
            )
        all_values = np.concatenate(
            (
                input_maps[row, :, target_mask].ravel(),
                output_maps[row, :, crop[0], crop[1]].ravel(),
            )
        )
        limits = np.percentile(all_values, (1.0, 99.0))
        axis.plot(limits, limits, color="black", ls="--", lw=1)
        state = "excluded from loss" if scale in excluded_scales else "trained"
        axis.set(
            title=f"j={scale}: {state}",
            xlabel="Input log-response quantile",
            ylabel="Synthesis log-response quantile",
        )
        axis.grid(alpha=0.2)
        if row == 0:
            axis.legend(fontsize=7)
    for axis in axes.ravel()[len(scales) :]:
        axis.set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _spectrum(image: np.ndarray, pixel_scale_arcsec: float):
    scale_arcmin = pixel_scale_arcsec / 60.0
    return radial_power_spectrum(
        power_spectrum_2d(image), scale_arcmin, nbins=48
    )[:2]


def _plot_power(
    path: Path,
    input_image: np.ndarray,
    synthesis: np.ndarray,
    centered_canvas: np.ndarray,
    pixel_scale_arcsec: float,
    reference_scale_arcsec: float | None,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for image, label in (
        (input_image, "Input"),
        (synthesis, "Central synthesis"),
        (centered_canvas, "Full canvas"),
    ):
        frequency, power = _spectrum(image, pixel_scale_arcsec)
        valid = np.isfinite(frequency) & np.isfinite(power) & (power > 0)
        axes[0].loglog(frequency[valid], power[valid], label=label)
    if reference_scale_arcsec is not None:
        axes[0].axvline(
            60.0 / reference_scale_arcsec,
            color="0.35",
            ls="--",
            label=f"{reference_scale_arcsec:g}″ diagnostic scale",
        )
    axes[0].set(
        title="Radial power (diagnostic only)",
        xlabel=r"Spatial frequency [arcmin$^{-1}$]",
        ylabel="Power",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, which="both")
    for axis, image, title in (
        (axes[1], input_image, "Input 2D log power"),
        (axes[2], synthesis, "Synthesis 2D log power"),
    ):
        power = np.fft.fftshift(
            np.abs(np.fft.fft2(image - np.mean(image))) ** 2
        )
        shown = axis.imshow(
            np.log10(np.maximum(power, np.finfo(float).tiny)),
            origin="lower",
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(shown, ax=axis, shrink=0.75)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _axis_leakage_ratio(power: np.ndarray) -> float:
    """Estimate the FFT-axis excess relative to nearby parallel slices."""

    size = power.shape[0]
    center = size // 2
    coordinate = np.arange(size) - center
    selected = (np.abs(coordinate) >= 8) & (np.abs(coordinate) <= size // 3)
    axis = np.concatenate(
        (power[center, selected], power[selected, center])
    )
    offset = max(3, min(6, size // 32))
    nearby = np.concatenate(
        (
            power[center + offset, selected],
            power[center - offset, selected],
            power[selected, center + offset],
            power[selected, center - offset],
        )
    )
    denominator = np.median(nearby)
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.median(axis) / denominator)


def _plot_windowed_power(
    path: Path,
    input_image: np.ndarray,
    synthesis: np.ndarray,
) -> None:
    """Separate internal Cartesian structure from non-periodic edge leakage."""

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for axis, image, title in (
        (axes[0], input_image, "Input"),
        (axes[1], synthesis, "Synthesis"),
    ):
        centered = image - np.mean(image)
        window_1d = np.hanning(image.shape[0])
        window = np.outer(window_1d, window_1d)
        raw_power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
        power = np.fft.fftshift(
            np.abs(np.fft.fft2(centered * window)) ** 2
        )
        shown = axis.imshow(
            np.log10(np.maximum(power, np.finfo(float).tiny)),
            origin="lower",
            cmap="magma",
        )
        axis.set_title(
            f"{title}: Hann-windowed 2D log power\n"
            f"axis/off-axis: raw={_axis_leakage_ratio(raw_power):.2f}, "
            f"windowed={_axis_leakage_ratio(power):.2f}"
        )
        axis.set_axis_off()
        figure.colorbar(shown, ax=axis, shrink=0.75)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pixel_pdf(
    path: Path, input_image: np.ndarray, synthesis: np.ndarray
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(input_image.ravel(), bins=50, density=True, histtype="step", label="Input")
    axes[0].hist(synthesis.ravel(), bins=50, density=True, histtype="step", label="Synthesis")
    axes[0].set(title="Pixel PDF", xlabel="Image value", ylabel="Density")
    centered_input = input_image - np.mean(input_image)
    centered_synthesis = synthesis - np.mean(synthesis)
    quantiles = np.linspace(0.01, 0.99, 99)
    axes[1].plot(
        np.quantile(centered_input, quantiles),
        np.quantile(centered_synthesis, quantiles),
    )
    limits = np.percentile(
        np.concatenate((centered_input.ravel(), centered_synthesis.ravel())),
        (1, 99),
    )
    axes[1].plot(limits, limits, color="black", ls="--")
    axes[1].set(
        title="Centered pixel quantiles",
        xlabel="Input",
        ylabel="Synthesis",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend() if axis is axes[0] else None
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_wst_lexicographic(
    path: Path,
    target: np.ndarray,
    synthesis: np.ndarray,
    *,
    maximum_scale: int,
    orientations: int,
    device: torch.device,
) -> None:
    """Compare isotropic ``s_mean`` coefficients in grouped lexicographic order.

    Each group starts with ``S1(j1)`` followed by all strict second-order
    ``S2(j1, j2, delta_l)`` paths, ordered first by ``j2`` and then by relative
    orientation.  This makes the vertical separators genuine ``j1`` boundaries,
    unlike a flattened order that first lists every S1 coefficient.
    """

    scale_count = int(maximum_scale) + 1
    target_calculator = ScatteringMean2D(
        target.shape,
        scale_count,
        int(orientations),
        pad_pixels=0,
        mask_boundary=False,
        device=str(device),
    )
    synthesis_calculator = ScatteringMean2D(
        synthesis.shape,
        scale_count,
        int(orientations),
        pad_pixels=0,
        mask_boundary=False,
        device=str(device),
    )
    features: list[MeanFeature] = []
    group_ends: list[int] = []
    for j1 in range(scale_count):
        features.append(MeanFeature(1, j1))
        for j2 in range(j1 + 1, scale_count):
            features.extend(
                MeanFeature(2, j1, j2, delta_l)
                for delta_l in range(orientations)
            )
        group_ends.append(len(features))

    with torch.no_grad():
        target_log = target_calculator.log_features(target, features)[0].cpu().numpy()
        synthesis_log = synthesis_calculator.log_features(synthesis, features)[0].cpu().numpy()
    coefficient = np.arange(len(features))
    fractional = np.expm1(synthesis_log - target_log)

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    axes[0].plot(coefficient, target_log, color="black", lw=1.1, label="Input target")
    axes[0].plot(coefficient, synthesis_log, color="tab:orange", lw=0.9, label="Synthesis")
    axes[0].set(
        title=(
            "WST s_mean lexicographic comparison: "
            "S1(j1), then S2(j1, j2, Δℓ)"
        ),
        ylabel="log s_mean",
    )
    axes[0].legend(loc="best")
    axes[1].axhspan(-0.05, 0.05, color="tab:green", alpha=0.14, label="±5%")
    axes[1].plot(coefficient, fractional, color="tab:blue", lw=0.85)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set(
        xlabel="Lexicographic s_mean coefficient index",
        ylabel="Synthesis / input − 1",
    )
    axes[1].legend(loc="best")
    for j1, stop in enumerate(group_ends[:-1]):
        separator = stop - 0.5
        for axis in axes:
            axis.axvline(separator, color="0.35", ls="--", lw=0.8, alpha=0.8)
        midpoint = (0 if j1 == 0 else group_ends[j1 - 1]) + (stop - (0 if j1 == 0 else group_ends[j1 - 1])) / 2 - 0.5
        axes[0].text(
            midpoint,
            0.98,
            f"j1={j1}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    # The final group has no following separator but still receives its label.
    start = group_ends[-2] if len(group_ends) > 1 else 0
    axes[0].text(
        start + (group_ends[-1] - start) / 2 - 0.5,
        0.98,
        f"j1={scale_count - 1}",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
    )
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_latent_state(
    path: Path,
    centered_canvas: np.ndarray,
    crop: tuple[slice, slice],
    model,
) -> None:
    grids = [
        parameter.detach().float().cpu().numpy()[0].std(axis=0)
        for parameter in model.grids
    ]
    panel_count = 2 + len(grids)
    columns = 3
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    limit = np.percentile(np.abs(centered_canvas), 99.9)
    axes[0, 0].imshow(
        centered_canvas, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axes[0, 0].set_title("Centered full neural field")
    central = centered_canvas[crop]
    central_limit = np.percentile(np.abs(central), 99.9)
    axes[0, 1].imshow(
        central,
        origin="lower",
        cmap="coolwarm",
        vmin=-central_limit,
        vmax=central_limit,
    )
    axes[0, 1].set_title("Centered delivered crop")
    grid_axes = list(axes.ravel()[2 : 2 + len(grids)])
    for axis, grid, resolution in zip(
        grid_axes, grids, model.grid_resolutions, strict=True
    ):
        shown = axis.imshow(grid, origin="lower", cmap="magma")
        axis.set_title(f"Latent channel σ, {resolution}×{resolution}")
        figure.colorbar(shown, ax=axis, shrink=0.7)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[panel_count:]:
        axis.set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_initialization(
    path: Path,
    target: np.ndarray,
    fitted: np.ndarray,
    crop: tuple[slice, slice],
    history: list[dict[str, float]],
) -> None:
    """Show the zero-mean initialization target and neural prefit."""

    difference = fitted - target
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (target, "Initialization canvas (central mean = 0)"),
        (fitted, "Neural field after initialization prefit"),
        (difference, "Prefit minus initialization"),
        (target[crop], "Initialization central crop"),
        (fitted[crop], "Prefitted central crop"),
    )
    for axis, (image, title) in zip(axes.ravel()[:5], panels, strict=True):
        limit = np.percentile(np.abs(image), 99.5)
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(shown, ax=axis, shrink=0.72)
    step = np.asarray([row["step"] for row in history]) + 1
    axes[1, 2].semilogy(step, [row["mse"] for row in history], label="MSE")
    axes[1, 2].semilogy(
        step,
        [row["gradient_norm"] for row in history],
        label="gradient norm",
        alpha=0.75,
    )
    axes[1, 2].set(
        title="Initialization prefit",
        xlabel="Step",
        ylabel="Value",
    )
    axes[1, 2].grid(alpha=0.2)
    axes[1, 2].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_decomposition(
    path: Path,
    input_image: np.ndarray,
    backbone: np.ndarray,
    target_residual: np.ndarray,
    raw_canvas: np.ndarray,
    projected_canvas: np.ndarray,
    synthesis: np.ndarray,
    crop: tuple[slice, slice],
) -> None:
    """Show every component in the fixed-backbone decomposition."""

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = (
        (input_image, "Input", "viridis", False),
        (backbone, "Fixed Gaussian backbone", "viridis", False),
        (target_residual, "Zero-mean high-pass target", "coolwarm", True),
        (raw_canvas, "Raw neural-field canvas", "coolwarm", True),
        (projected_canvas, "Projected high-pass neural canvas", "coolwarm", True),
        (synthesis, "Backbone + synthesized residual", "viridis", False),
    )
    for axis, (image, title, cmap, symmetric) in zip(
        axes.ravel(), panels, strict=True
    ):
        if symmetric:
            limit = np.percentile(np.abs(image), 99.5)
            bounds = (-limit, limit)
        else:
            bounds = tuple(np.percentile(image, (0.1, 99.9)))
        shown = axis.imshow(
            image,
            origin="lower",
            cmap=cmap,
            vmin=bounds[0],
            vmax=bounds[1],
        )
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, shrink=0.72)
    yslice, xslice = crop
    for axis in (axes[1, 0], axes[1, 1]):
        axis.add_patch(
            Rectangle(
                (xslice.start - 0.5, yslice.start - 0.5),
                xslice.stop - xslice.start,
                yslice.stop - yslice.start,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
            )
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_decomposition_power(
    path: Path,
    input_image: np.ndarray,
    backbone: np.ndarray,
    target_residual: np.ndarray,
    synthesis: np.ndarray,
    synthesis_residual: np.ndarray,
    pixel_scale_arcsec: float,
    fwhm_pixels: float,
) -> None:
    """Compare component spectra and show the complementary Gaussian transfer."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for image, label in (
        (input_image, "Input"),
        (backbone, "Fixed backbone"),
        (target_residual, "Target high-pass"),
        (synthesis_residual, "Synthesized high-pass"),
        (synthesis, "Final synthesis"),
    ):
        frequency, power = _spectrum(image, pixel_scale_arcsec)
        valid = np.isfinite(frequency) & np.isfinite(power) & (power > 0)
        axes[0].loglog(frequency[valid], power[valid], label=label)
    axes[0].set(
        title="Backbone/residual radial power",
        xlabel=r"Spatial frequency [arcmin$^{-1}$]",
        ylabel="Power",
    )
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)

    sigma_pixels = fwhm_pixels / 2.354820045
    frequency_pixel = np.linspace(0.0, 0.5, 1024)
    lowpass = np.exp(
        -2.0 * np.pi**2 * sigma_pixels**2 * frequency_pixel**2
    )
    frequency_arcmin = frequency_pixel / (pixel_scale_arcsec / 60.0)
    axes[1].plot(frequency_arcmin, lowpass, label="Gaussian low-pass")
    axes[1].plot(frequency_arcmin, 1.0 - lowpass, label="Complementary high-pass")
    nominal_scale_arcmin = fwhm_pixels * pixel_scale_arcsec / 60.0
    axes[1].axvline(
        1.0 / nominal_scale_arcmin,
        color="0.35",
        ls="--",
        label=f"$2^{{j_{{max}}}}$ = {nominal_scale_arcmin:.3g}′",
    )
    axes[1].set(
        xlim=(0.0, min(5.0, frequency_arcmin[-1])),
        ylim=(-0.02, 1.02),
        title="Automatic complementary scale split",
        xlabel=r"Spatial frequency [arcmin$^{-1}$]",
        ylabel="Amplitude transfer",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _patch_copy_metrics(
    input_image: np.ndarray, synthesis: np.ndarray, seed: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    rng = np.random.default_rng(seed + 70000)
    for size in (16, 32):
        input_origins = np.argwhere(
            np.ones(
                (
                    input_image.shape[0] - size + 1,
                    input_image.shape[1] - size + 1,
                ),
                dtype=bool,
            )
        )
        synthesis_origins = input_origins.copy()
        count = min(256, len(input_origins))
        input_selected = input_origins[
            rng.choice(len(input_origins), count, replace=False)
        ]
        synthesis_selected = synthesis_origins[
            rng.choice(len(synthesis_origins), count, replace=False)
        ]
        input_vectors = extract_patch_vectors(
            torch.as_tensor(input_image), input_selected, size
        )
        synthesis_vectors = extract_patch_vectors(
            torch.as_tensor(synthesis), synthesis_selected, size
        )
        normalizer = RobustNormalizer.fit(input_vectors.numpy())
        values = nearest_patch_metrics(
            normalizer.apply(synthesis_vectors).numpy(),
            normalizer.apply(input_vectors).numpy(),
        )
        result.update({f"patch{size}_{key}": value for key, value in values.items()})
    return result


def create_single_image_diagnostics(
    *,
    directory: Path,
    input_image: np.ndarray,
    synthesis: np.ndarray,
    centered_canvas: np.ndarray,
    restored_canvas: np.ndarray,
    crop: tuple[slice, slice],
    target_mask: np.ndarray,
    input_mean: float,
    output_mean: float | None = None,
    pixel_scale_arcsec: float,
    config: SingleImageConfig,
    history: list[dict[str, float]],
    model,
    target_calculator,
    output_calculator,
    normalizer: RobustNormalizer,
    target_features: torch.Tensor,
    target_raw_features: torch.Tensor,
    device: torch.device,
    initialization_target: np.ndarray | None,
    initialization_model: np.ndarray | None,
    initialization_history: list[dict[str, float]],
    backbone: np.ndarray | None,
    target_residual: np.ndarray,
    raw_canvas: np.ndarray,
    decomposition_fwhm_pixels: float | None,
    fbm_prior_info: dict | None = None,
) -> dict[str, float]:
    """Generate all baseline figures and return machine-readable metrics."""

    scales = config.morlet.diagnostic_scales
    trained = config.morlet.training_scales
    excluded = tuple(scale for scale in scales if scale not in trained)
    input_maps = _morlet_response_maps(
        input_image - input_mean, target_calculator, scales
    )
    output_maps = _morlet_response_maps(centered_canvas, output_calculator, scales)

    _plot_canvas(
        directory / "canvas_and_crop.png",
        input_image,
        synthesis,
        restored_canvas,
        crop,
        config.input.label,
    )
    _plot_history(directory / "loss_history.png", history, trained)
    _plot_response_maps(
        directory / "morlet_response_input.png",
        input_maps,
        scales,
        "Input log-Morlet responses; cyan = accepted joint-target region",
        accepted_mask=target_mask,
        excluded_scales=excluded,
    )
    _plot_response_maps(
        directory / "morlet_response_synthesis.png",
        output_maps,
        scales,
        "Synthesis log-Morlet responses in delivered crop",
        crop=crop,
        excluded_scales=excluded,
    )
    _plot_response_distributions(
        directory / "morlet_response_quantiles.png",
        input_maps,
        output_maps,
        target_mask,
        crop,
        scales,
        excluded,
    )
    _plot_power(
        directory / "power_diagnostics.png",
        input_image,
        synthesis,
        centered_canvas,
        pixel_scale_arcsec,
        (
            config.power.minimum_scale_arcsec
            if config.power.enabled
            else 2.35 * pixel_scale_arcsec * 2 ** min(trained)
        ),
    )
    _plot_windowed_power(
        directory / "power_diagnostics_windowed.png",
        input_image,
        synthesis,
    )
    _plot_pixel_pdf(directory / "pixel_distribution.png", input_image, synthesis)
    if fbm_prior_info is not None:
        _plot_fbm_extrapolation(
            directory / "fbm_extrapolation_diagnostic.png", fbm_prior_info
        )
    _plot_wst_lexicographic(
        directory / "wst_lexicographic.png",
        input_image - input_mean,
        synthesis - input_mean,
        maximum_scale=max(scales),
        orientations=config.morlet.orientations,
        device=device,
    )
    _plot_latent_state(
        directory / "neural_field_diagnostics.png",
        centered_canvas,
        crop,
        model,
    )
    if initialization_target is not None and initialization_model is not None:
        _plot_initialization(
            directory / "initialization_diagnostics.png",
            initialization_target,
            initialization_model,
            crop,
            initialization_history,
        )
    if backbone is not None and decomposition_fwhm_pixels is not None:
        synthesized_residual = synthesis - backbone
        _plot_decomposition(
            directory / "decomposition_components.png",
            input_image,
            backbone,
            target_residual,
            raw_canvas,
            centered_canvas,
            synthesis,
            crop,
        )
        _plot_decomposition_power(
            directory / "decomposition_power.png",
            input_image,
            backbone,
            target_residual,
            synthesis,
            synthesized_residual,
            pixel_scale_arcsec,
            decomposition_fwhm_pixels,
        )

    output_descriptor = OrientedMorletFeatures(
        output_calculator,
        trained,
        spatial_stride=1,
        boundary_sigma=config.morlet.boundary_sigma,
        orientation_pool=False,
    )
    output_mask = np.zeros(centered_canvas.shape, dtype=bool)
    output_mask[crop] = True
    with torch.no_grad():
        output_features = output_descriptor.features(
            centered_canvas, center_mask=output_mask
        )
        generator = torch.Generator(device=device).manual_seed(
            config.runtime.seed + 80000
        )
        output_features = subsample_rows(
            output_features, config.morlet.output_samples, generator
        )
        standardized_output = normalizer.apply(output_features)
        joint = sliced_wasserstein_loss(
            standardized_output,
            target_features,
            projection_count=config.morlet.projections,
            quantile_count=config.morlet.quantiles,
            generator=torch.Generator(device=device).manual_seed(
                config.runtime.seed + 80001
            ),
            chunk_size=config.morlet.projection_chunk,
        )
    _plot_constraint_components(
        directory / "constraint_components.png",
        history,
        config,
        target_raw_features,
        output_features,
    )
    metrics: dict[str, float] = {
        "final_joint_morlet_swd": float(joint.cpu()),
        "input_std": float(np.std(input_image)),
        "synthesis_std": float(np.std(synthesis)),
        "input_mad_std": float(
            1.4826022185 * np.median(np.abs(input_image - np.median(input_image)))
        ),
        "synthesis_mad_std": float(
            1.4826022185 * np.median(np.abs(synthesis - np.median(synthesis)))
        ),
        "mean_error": float(
            np.mean(synthesis, dtype=np.float64)
            - (input_mean if output_mean is None else output_mean)
        ),
        "morlet_target_margin_pixels": float(
            morlet_margin(max(trained), config.morlet.boundary_sigma)
        ),
        "morlet_target_accepted_fraction": float(target_mask.mean()),
    }
    input_highpass = input_image - gaussian_filter(input_image, sigma=1.0)
    synthesis_highpass = synthesis - gaussian_filter(synthesis, sigma=1.0)
    metrics["input_highpass_sigma1_mad_std"] = float(
        1.4826022185
        * np.median(np.abs(input_highpass - np.median(input_highpass)))
    )
    metrics["synthesis_highpass_sigma1_mad_std"] = float(
        1.4826022185
        * np.median(np.abs(synthesis_highpass - np.median(synthesis_highpass)))
    )
    orientations = config.morlet.orientations
    with torch.no_grad():
        target_energy = log_wavelet_energy_by_scale(
            target_raw_features,
            scales=len(trained),
            orientations=orientations,
        )
        output_energy = log_wavelet_energy_by_scale(
            output_features,
            scales=len(trained),
            orientations=orientations,
        )
        for index, scale in enumerate(trained):
            section = slice(index * orientations, (index + 1) * orientations)
            loss = sliced_wasserstein_loss(
                standardized_output[:, section],
                target_features[:, section],
                projection_count=config.morlet.projections,
                quantile_count=config.morlet.quantiles,
                generator=torch.Generator(device=device).manual_seed(
                    config.runtime.seed + 81000 + scale
                ),
                chunk_size=config.morlet.projection_chunk,
            )
            metrics[f"final_morlet_j{scale}_swd"] = float(loss.cpu())
            metrics[f"input_log_wavelet_energy_j{scale}"] = float(
                target_energy[index].cpu()
            )
            metrics[f"synthesis_log_wavelet_energy_j{scale}"] = float(
                output_energy[index].cpu()
            )
            metrics[f"wavelet_energy_error_j{scale}"] = float(
                (output_energy[index] - target_energy[index]).cpu()
            )
        labels = config.cross_scale.pairs or tuple(
            (fine, coarse)
            for fine in config.cross_scale.fine_scales
            for coarse in config.cross_scale.coarse_scales
        )
        if labels:
            scale_index = {
                scale: position for position, scale in enumerate(trained)
            }
            pairs = tuple(
                (scale_index[fine], scale_index[coarse])
                for fine, coarse in labels
            )
            coupling_function = (
                log_aligned_cross_scale_coupling
                if config.cross_scale.aligned_orientations
                else log_cross_scale_envelope_coupling
            )
            target_coupling = coupling_function(
                target_raw_features,
                scales=len(trained),
                orientations=orientations,
                pairs=pairs,
            )
            output_coupling = coupling_function(
                output_features,
                scales=len(trained),
                orientations=orientations,
                pairs=pairs,
            )
            pair_weights = correlation_floor_weights(
                target_coupling,
                eta=config.cross_scale.correlation_weight_eta,
                maximum_uniform_multiple=config.cross_scale.maximum_uniform_multiple,
            )
            for index, (fine, coarse) in enumerate(labels):
                metrics[f"input_cross_scale_j{fine}_j{coarse}"] = float(
                    torch.exp(target_coupling[index]).cpu()
                )
                metrics[f"synthesis_cross_scale_j{fine}_j{coarse}"] = float(
                    torch.exp(output_coupling[index]).cpu()
                )
                metrics[f"cross_scale_weight_j{fine}_j{coarse}"] = float(
                    pair_weights[index].cpu()
                )
    metrics.update(
        {f"boundary_{key}": value for key, value in boundary_metrics(synthesis).items()}
    )
    metrics.update(_patch_copy_metrics(input_image, synthesis, config.runtime.seed))
    if backbone is not None:
        synthesized_residual = synthesis - backbone
        metrics.update(
            {
                "backbone_mean": float(np.mean(backbone, dtype=np.float64)),
                "target_residual_mean": float(
                    np.mean(target_residual, dtype=np.float64)
                ),
                "target_residual_std": float(np.std(target_residual)),
                "synthesis_residual_mean": float(
                    np.mean(synthesized_residual, dtype=np.float64)
                ),
                "synthesis_residual_std": float(np.std(synthesized_residual)),
            }
        )
        if input_image.shape == backbone.shape:
            metrics["decomposition_reconstruction_max_abs"] = float(
                np.max(np.abs(input_image - backbone - target_residual))
            )
        else:
            metrics["external_texture_target"] = 1.0
    if initialization_target is not None and initialization_model is not None:
        metrics["initialization_target_crop_mean"] = float(
            np.mean(initialization_target[crop], dtype=np.float64)
        )
        metrics["initialization_model_crop_mean"] = float(
            np.mean(initialization_model[crop], dtype=np.float64)
        )
        metrics["initialization_model_mse"] = float(
            np.mean((initialization_model - initialization_target) ** 2)
        )
    return metrics
