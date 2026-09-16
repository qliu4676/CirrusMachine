"""Numerical validation for neural-field synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel

from cirrusmachine.power_spectrum import power_spectrum_2d, radial_power_spectrum

from .operators import low_frequency_residual


def basic_metrics(
    synthesis: np.ndarray,
    backbone: np.ndarray,
    residual: np.ndarray,
    *,
    transition_pixels: float,
    transition_range_pixels: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Return mean preservation, low-mode leakage, and residual statistics."""

    synth = torch.as_tensor(synthesis, dtype=torch.float32)
    base = torch.as_tensor(backbone, dtype=torch.float32)
    leakage = low_frequency_residual(
        synth, base, transition_pixels, transition_range_pixels
    ).numpy()
    return {
        "synthesis_mean_mjy_sr": float(np.mean(synthesis)),
        "backbone_mean_mjy_sr": float(np.mean(backbone)),
        "mean_residual_mjy_sr": float(np.mean(synthesis) - np.mean(backbone)),
        "low_mode_leakage_rms_mjy_sr": float(np.sqrt(np.mean(leakage**2))),
        "residual_rms_mjy_sr": float(np.sqrt(np.mean(np.asarray(residual) ** 2))),
        "residual_min_mjy_sr": float(np.min(residual)),
        "residual_max_mjy_sr": float(np.max(residual)),
    }


def power_metrics(
    synthesis: np.ndarray,
    backbone: np.ndarray,
    exemplar: np.ndarray,
    pixel_scale_arcsec: float,
) -> dict[str, np.ndarray]:
    """Measure diagnostic-only 1D spectra; no power loss is implied."""

    result: dict[str, np.ndarray] = {}
    for name, image in {
        "synthesis": synthesis,
        "backbone": backbone,
        "exemplar": exemplar,
    }.items():
        power = power_spectrum_2d(image)
        frequency, radial, scatter = radial_power_spectrum(
            power, pixel_scale_arcsec / 60.0, nbins=48
        )
        result[f"{name}_frequency_arcmin_inv"] = frequency
        result[f"{name}_radial_power"] = radial
        result[f"{name}_radial_scatter"] = scatter
    return result


def plot_texture_reference_power_spectra(
    output: str | Path,
    cutouts: tuple[np.ndarray, ...],
    full_herschel: np.ndarray,
    full_planck: np.ndarray,
    pixel_scale_arcsec: float,
) -> None:
    """Compare each texture exemplar with the common full-map reference."""

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    scale = pixel_scale_arcsec / 60.0
    for index, image in enumerate(cutouts, start=1):
        frequency, power, _ = radial_power_spectrum(
            power_spectrum_2d(image), scale, nbins=48
        )
        axis.loglog(frequency, power, alpha=0.7, label=f"Herschel cutout {index}")
    for image, label, color in (
        (full_herschel, "Herschel central 0.5°", "black"),
        (full_planck, "Affine Planck central 0.5°", "tab:blue"),
    ):
        valid = np.isfinite(image)
        frequency, power, _ = radial_power_spectrum(
            power_spectrum_2d(image, valid=valid), scale, nbins=48
        )
        axis.loglog(frequency, power, color=color, lw=2.2, label=label)
    axis.set(
        xlabel="Spatial frequency [arcmin$^{-1}$]",
        ylabel="Azimuthally averaged power",
        title="Herschel texture references and central full-map power",
    )
    axis.axvspan(1.0 / 10.0, 1.0 / 3.0, color="0.7", alpha=0.25, label="3–10 arcmin")
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def distribution_metrics(image: np.ndarray, prefix: str) -> dict[str, float]:
    """Return robust pixel and high-pass distribution summaries."""

    values = np.asarray(image, dtype=float)
    highpass = values - gaussian_filter(values, 6.0, mode="reflect")
    result: dict[str, float] = {}
    for name, array in (("pixel", values), ("highpass", highpass)):
        quantiles = np.percentile(array[np.isfinite(array)], (1, 16, 50, 84, 99))
        for label, value in zip(("q01", "q16", "q50", "q84", "q99"), quantiles, strict=True):
            result[f"{prefix}_{name}_{label}"] = float(value)
    return result


def structure_tensor_metrics(image: np.ndarray, prefix: str, sigma: float = 2.0) -> dict[str, float]:
    """Summarize filament orientation and anisotropy from a structure tensor."""

    value = np.asarray(image, dtype=float)
    gx = sobel(value, axis=1, mode="reflect") / 8.0
    gy = sobel(value, axis=0, mode="reflect") / 8.0
    jxx = gaussian_filter(gx * gx, sigma, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma, mode="reflect")
    discriminant = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    trace = jxx + jyy
    anisotropy = discriminant / np.maximum(trace, np.finfo(float).eps)
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    weight = trace / np.maximum(np.sum(trace), np.finfo(float).eps)
    doubled = np.sum(weight * np.exp(2j * angle))
    return {
        f"{prefix}_structure_anisotropy_median": float(np.median(anisotropy)),
        f"{prefix}_structure_anisotropy_mean": float(np.mean(anisotropy)),
        f"{prefix}_orientation_resultant": float(np.abs(doubled)),
        f"{prefix}_orientation_deg": float(np.rad2deg(0.5 * np.angle(doubled))),
    }


def boundary_metrics(image: np.ndarray) -> dict[str, float]:
    """Measure both periodic seams and local normal gradients at the border."""

    value = np.asarray(image, dtype=float)
    horizontal = np.diff(value, axis=1)
    vertical = np.diff(value, axis=0)
    reference = np.sqrt(0.5 * (np.mean(horizontal**2) + np.mean(vertical**2)))
    left_right = np.sqrt(np.mean((value[:, 0] - value[:, -1]) ** 2))
    bottom_top = np.sqrt(np.mean((value[0] - value[-1]) ** 2))
    normal = np.concatenate(
        (
            (value[0] - value[1]).ravel(),
            (value[-1] - value[-2]).ravel(),
            (value[:, 0] - value[:, 1]).ravel(),
            (value[:, -1] - value[:, -2]).ravel(),
        )
    )
    normal_rms = np.sqrt(np.mean(normal**2))
    denominator = max(reference, np.finfo(float).eps)
    return {
        "edge_left_right_rms": float(left_right),
        "edge_bottom_top_rms": float(bottom_top),
        "edge_left_right_over_gradient": float(left_right / denominator),
        "edge_bottom_top_over_gradient": float(bottom_top / denominator),
        "edge_normal_gradient_rms": float(normal_rms),
        "edge_normal_over_gradient": float(normal_rms / denominator),
    }


def nearest_patch_metrics(
    synthesis_vectors: np.ndarray,
    exemplar_vectors: np.ndarray,
    *,
    chunk_size: int = 256,
) -> dict[str, float]:
    """Return standardized nearest-exemplar distances as a copying diagnostic."""

    synthesis = np.asarray(synthesis_vectors, dtype=np.float32)
    exemplar = np.asarray(exemplar_vectors, dtype=np.float32)
    nearest = []
    exemplar_norm = np.sum(exemplar**2, axis=1)[None]
    for start in range(0, len(synthesis), chunk_size):
        rows = synthesis[start : start + chunk_size]
        distance2 = (
            np.sum(rows**2, axis=1)[:, None]
            + exemplar_norm
            - 2.0 * rows @ exemplar.T
        )
        nearest.append(np.sqrt(np.maximum(np.min(distance2, axis=1), 0.0)))
    distances = np.concatenate(nearest)
    return {
        "nearest_patch_distance_min": float(np.min(distances)),
        "nearest_patch_distance_median": float(np.median(distances)),
        "nearest_patch_distance_q05": float(np.percentile(distances, 5.0)),
    }


def plot_core_diagnostics(
    output: str | Path,
    synthesis: np.ndarray,
    backbone: np.ndarray,
    exemplar: np.ndarray,
    residual: np.ndarray,
    *,
    transition_pixels: float,
) -> None:
    """Save the approved input/exemplar/output/residual/common-filter panels."""

    common_sigma = transition_pixels / 2.354820045
    common_synth = gaussian_filter(synthesis, common_sigma, mode="reflect")
    common_planck = gaussian_filter(backbone, common_sigma, mode="reflect")
    difference = common_synth - common_planck
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (backbone, "Affine Planck backbone", "viridis"),
        (exemplar, "Herschel cutout 4", "viridis"),
        (synthesis, "Neural-field synthesis", "viridis"),
        (residual, "Projected residual", "coolwarm"),
        (common_synth, "Common-filtered synthesis", "viridis"),
        (difference, "Common-filtered difference", "coolwarm"),
    )
    for ax, (image, title, cmap) in zip(axes.ravel(), panels, strict=True):
        if cmap == "coolwarm":
            limit = np.nanpercentile(np.abs(image), 99.9)
            shown = ax.imshow(image, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            limits = np.nanpercentile(image, (0.1, 99.9))
            shown = ax.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(shown, ax=ax, shrink=0.75)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_neural_field_state(
    output: str | Path,
    raw_field: np.ndarray,
    projected_residual: np.ndarray,
    model: torch.nn.Module,
    *,
    transition_pixels: float,
    transition_range_pixels: tuple[float, float] | None = None,
    apodization_fraction: float = 0.0,
) -> None:
    """Visualize the implicit field, projection, cutoff, and latent grids."""

    shape = raw_field.shape
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1]))[None, :]
    radius = np.hypot(fy, fx)
    if transition_range_pixels is None:
        transfer = (radius >= 1.0 / transition_pixels).astype(float)
    else:
        fine, coarse = transition_range_pixels
        lower, upper = 1.0 / coarse, 1.0 / fine
        phase = np.clip((radius - lower) / (upper - lower), 0.0, 1.0)
        transfer = 0.5 - 0.5 * np.cos(np.pi * phase)
    grids = [
        parameter.detach().float().cpu().numpy()[0].std(axis=0)
        for parameter in model.grids
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    projection_label = "Projected residual"
    if transition_range_pixels is not None:
        fine_pixels, coarse_pixels = transition_range_pixels
        projection_label = (
            f"Smooth-band projected residual "
            f"({fine_pixels:g}–{coarse_pixels:g} px)"
        )
    panels = [
        (raw_field, "Raw coordinate neural field", "coolwarm"),
        (projected_residual, projection_label, "coolwarm"),
        (transfer, "High-pass transfer (centered FFT)", "gray"),
    ]
    panels.extend(
        (grid, f"Latent-grid channel σ: {resolution}²", "magma")
        for grid, resolution in zip(grids, model.grid_resolutions, strict=True)
    )
    for ax, panel in zip(axes.ravel(), panels[:6], strict=True):
        image, title, cmap = panel
        if cmap == "coolwarm":
            limit = np.nanpercentile(np.abs(image), 99.9)
            shown = ax.imshow(image, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            shown = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(shown, ax=ax, shrink=0.75)
    # Four latent levels produce seven panels total; show the finest level as
    # an inset so the fixed 2x3 layout remains compact.
    if len(panels) > 6:
        inset = axes[1, 2].inset_axes([0.54, 0.04, 0.43, 0.43])
        inset.imshow(panels[6][0], origin="lower", cmap="magma")
        inset.set_title(panels[6][1], fontsize=7)
        inset.set_xticks([])
        inset.set_yticks([])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_power_comparison(
    output: str | Path,
    spectra: dict[str, np.ndarray],
    *,
    transition_arcmin: float,
    resolution_arcmin: float,
    transition_range_arcmin: tuple[float, float] | None = None,
) -> None:
    """Plot the diagnostic-only radial spectra on their native footprints."""

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for name, label in (
        ("backbone", "Affine Planck backbone"),
        ("exemplar", "Herschel cutout 4"),
        ("synthesis", "Neural-field synthesis"),
    ):
        frequency = spectra[f"{name}_frequency_arcmin_inv"]
        power = spectra[f"{name}_radial_power"]
        finite = np.isfinite(frequency) & np.isfinite(power) & (power > 0)
        ax.loglog(frequency[finite], power[finite], label=label)
    if transition_range_arcmin is None:
        ax.axvline(1.0 / transition_arcmin, color="0.35", ls="--", label="j=6 transition")
    else:
        fine, coarse = transition_range_arcmin
        ax.axvspan(
            1.0 / coarse,
            1.0 / fine,
            color="0.5",
            alpha=0.16,
            label=f"{fine:g}–{coarse:g} arcmin transition",
        )
    ax.axvline(1.0 / resolution_arcmin, color="0.35", ls=":", label="36 arcsec")
    ax.set(
        xlabel=r"Spatial frequency [arcmin$^{-1}$]",
        ylabel="Azimuthally averaged power",
        title="Power spectra (diagnostic only; different footprints)",
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_history(output: str | Path, history: list[dict[str, float]]) -> None:
    """Plot total and component losses plus residual RMS."""

    steps = np.asarray([row["step"] for row in history]) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    components = [
        ("loss", "total"),
        ("patch_swd", "patch SWD"),
        ("morlet_swd", "Morlet SWD"),
        ("md07", "MD07"),
    ]
    if "power" in history[0]:
        components.append(("power", "radial power"))
    if "isotropy" in history[0]:
        components.append(("isotropy", "power anisotropy"))
    if "global_power" in history[0]:
        components.append(("global_power", "global power"))
    if "boundary" in history[0]:
        components.append(("boundary", "boundary gradient"))
    for key, label in components:
        axes[0].semilogy(steps, [row[key] for row in history], label=label, alpha=0.9)
    axes[0].set(xlabel="Optimization step", ylabel="Loss", title="Training losses")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [row["residual_rms"] for row in history])
    axes[1].set(
        xlabel="Optimization step",
        ylabel=r"Projected residual RMS [MJy sr$^{-1}$]",
        title="Texture-amplitude evolution",
    )
    axes[1].grid(alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }
    output.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
