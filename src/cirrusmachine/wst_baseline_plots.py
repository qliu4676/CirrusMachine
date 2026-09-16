"""Diagnostic plots for the guarded single-image WST baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .morphology import morphology_curves
from .power_spectrum import power_spectrum_2d, radial_power_spectrum
from .scattering_mean import MeanFeature


def _save(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_image_comparison(
    target: np.ndarray,
    initial: np.ndarray,
    output: np.ndarray,
    path: Path,
    *,
    output_title: str = "Baseline A: reduced WST",
) -> None:
    """Compare target, independent power seed, and optimized central crop."""

    limits = np.percentile(np.concatenate((target.ravel(), output.ravel())), (0.5, 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = (
        (target, "Dragonfly target"),
        (initial, "Independent ensemble-power seed"),
        (output, output_title),
    )
    for ax, (image, title) in zip(axes, panels, strict=True):
        shown = ax.imshow(
            image, origin="lower", cmap="viridis", vmin=limits[0], vmax=limits[1]
        )
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(shown, ax=axes, label="kJy sr$^{-1}$", shrink=0.8)
    _save(
        fig,
        path,
        "Aim: visually test whether optimization restores non-Gaussian dynamic range "
        "and filamentary morphology without copying the target's phases. All panels "
        "share one intensity stretch.",
    )


def plot_power_comparison(
    target: np.ndarray,
    initial: np.ndarray,
    output: np.ndarray,
    *,
    model_frequency: np.ndarray,
    model_power: np.ndarray,
    pixel_scale_arcmin: float,
    path: Path,
) -> None:
    """Compare radial and two-dimensional spectra."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for image, label, color in (
        (target, "target", "black"),
        (initial, "initial", "tab:blue"),
        (output, "output", "tab:orange"),
    ):
        power = power_spectrum_2d(image)
        frequency, radial, _ = radial_power_spectrum(
            power, pixel_scale_arcmin, nbins=50
        )
        axes[0, 0].loglog(frequency, radial, color=color, label=label)
    axes[0, 0].loglog(
        model_frequency, model_power, color="tab:red", ls="--", label="latent model"
    )
    axes[0, 0].axvline(1.0, color="0.5", ls=":", label="fully reliable limit")
    axes[0, 0].axvline(2.0, color="0.5", ls="-.", label="downweighted limit")
    axes[0, 0].set(
        xlabel="spatial frequency [arcmin$^{-1}$]",
        ylabel="power",
        title="Radial power spectrum",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    for ax, image, title in (
        (axes[0, 1], target, "Target 2-D log power"),
        (axes[1, 0], initial, "Initial 2-D log power"),
        (axes[1, 1], output, "Output 2-D log power"),
    ):
        shown = ax.imshow(
            np.log10(np.maximum(power_spectrum_2d(image), np.finfo(float).tiny)),
            origin="lower",
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(shown, ax=ax, shrink=0.75)
    _save(
        fig,
        path,
        "Aim: verify ensemble-spectrum agreement while revealing crosses, rings, "
        "stripes, or axis-aligned excess. Frequencies above 1 arcmin^-1 are "
        "downweighted and above 2 arcmin^-1 are not claimed as recovered cirrus.",
    )


def plot_cdf_comparison(
    target: np.ndarray,
    initial: np.ndarray,
    output: np.ndarray,
    *,
    sigma_pixels: float,
    levels: np.ndarray,
    target_quantiles: np.ndarray,
    uncertainty: np.ndarray,
    path: Path,
) -> None:
    """Compare robust CDF and native pixel distributions."""

    initial_smooth = gaussian_filter(initial, sigma_pixels, mode="reflect")
    output_smooth = gaussian_filter(output, sigma_pixels, mode="reflect")
    initial_q = np.quantile(initial_smooth, levels)
    output_q = np.quantile(output_smooth, levels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].fill_between(
        levels,
        target_quantiles - uncertainty,
        target_quantiles + uncertainty,
        color="0.85",
        label="target ±1σ",
    )
    axes[0].plot(levels, target_quantiles, color="black", label="target")
    axes[0].plot(levels, initial_q, label="initial")
    axes[0].plot(levels, output_q, label="output")
    axes[0].set(
        xlabel="quantile level",
        ylabel="smoothed intensity [kJy sr$^{-1}$]",
        title=f"Robust CDF after Gaussian σ={sigma_pixels:g} px",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    bins = np.linspace(
        np.percentile(np.concatenate((target, output)), 0.2),
        np.percentile(np.concatenate((target, output)), 99.8),
        80,
    )
    for image, label in ((target, "target"), (initial, "initial"), (output, "output")):
        axes[1].hist(
            image.ravel(), bins=bins, density=True, histtype="step", label=label
        )
    axes[1].set(
        xlabel="native-grid intensity [kJy sr$^{-1}$]",
        ylabel="density",
        title="Native pixel PDF (diagnostic only)",
    )
    axes[1].legend(fontsize=8)
    _save(
        fig,
        path,
        "Aim: check recovery of the defensible surface-brightness distribution. Only "
        "the smoothed 0.5--99.5% quantiles enter the loss; native tails are shown to "
        "expose spikes or noise amplification.",
    )


def plot_wst_residuals(
    features: Sequence[MeanFeature],
    initial_z: np.ndarray,
    output_z: np.ndarray,
    path: Path,
) -> None:
    """Show uncertainty-standardized WST residuals."""

    x = np.arange(len(features))
    order = np.asarray([feature.order for feature in features])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for ax, values, title in (
        (axes[0], initial_z, "Initial WST residuals"),
        (axes[1], output_z, "Optimized WST residuals"),
    ):
        ax.axhspan(-1, 1, color="tab:green", alpha=0.15, label="accepted ±1σ band")
        for value, color, label in ((1, "tab:blue", "S1"), (2, "tab:orange", "S2")):
            use = order == value
            ax.scatter(x[use], values[use], s=24, color=color, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(ylabel="(measured − target) / uncertainty", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("canonical WST coefficient index")
    _save(
        fig,
        path,
        "Aim: verify uncertainty-band rather than exact coefficient matching. Points "
        "inside the green region are statistically adequate; j=2-related coefficients "
        "already carry enlarged observational uncertainty.",
    )


def plot_phase_residuals(
    labels: Sequence[str],
    initial_z: np.ndarray,
    output_z: np.ndarray,
    path: Path,
) -> None:
    """Compare adjacent-scale phase-harmonic residuals."""

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.axhspan(-1, 1, color="tab:green", alpha=0.15)
    ax.bar(x - width / 2, initial_z, width, label="phase-ablation input")
    ax.bar(x + width / 2, output_z, width, label="phase-ablation output")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="(measured − target) / uncertainty",
        title="Adjacent-scale phase-harmonic residuals",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(
        fig,
        path,
        "Compare phase alignment across adjacent reliable scales before and after "
        "the controlled phase-harmonic ablation.",
    )


def plot_scale_residuals(
    features: Sequence[MeanFeature],
    initial_z: np.ndarray,
    output_z: np.ndarray,
    *,
    scale: int,
    path: Path,
) -> None:
    """Compare residuals for every coefficient involving one scale."""

    use = np.asarray(
        [
            scale
            in (
                (feature.j1,)
                if feature.order == 1
                else (feature.j1, feature.j2)
            )
            for feature in features
        ]
    )
    labels = [feature.label for feature, keep in zip(features, use) if keep]
    x = np.arange(np.count_nonzero(use))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    ax.axhspan(-1, 1, color="tab:green", alpha=0.15)
    ax.bar(x - width / 2, initial_z[use], width, label="ablation input")
    ax.bar(x + width / 2, output_z[use], width, label="ablation output")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="(measured − target) / uncertainty",
        title=f"WST residuals involving j={scale}",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path, "")


def plot_loss_history(history: Sequence[Mapping[str, float]], path: Path) -> None:
    """Show staged loss, family violations, and negative-value monitoring."""

    steps = np.asarray([row["step"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].semilogy(steps, [row["loss"] for row in history])
    axes[0, 0].axvline(60, color="tab:red", ls="--", label="S2 activated")
    axes[0, 0].set(xlabel="Adam step", ylabel="loss", title="Uncertainty-band objective")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    families = ["power", "cdf", "s1", "s2"]
    if "phase_violation_rms" in history[0]:
        families.append("phase")
    for key in history[0]:
        if key.endswith("_violation_rms"):
            name = key.removesuffix("_violation_rms")
            if name not in families:
                families.append(name)
    for name in families:
        axes[0, 1].plot(
            steps,
            [row[f"{name}_violation_rms"] for row in history],
            label=name,
        )
    axes[0, 1].axhline(0.15, color="black", ls="--")
    axes[0, 1].set(
        xlabel="Adam step",
        ylabel="RMS excess beyond ±1σ",
        title="Family-level target violations",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    for name in families:
        axes[1, 0].plot(
            steps,
            [row[f"{name}_within_fraction"] for row in history],
            label=name,
        )
    axes[1, 0].axhline(0.9, color="black", ls="--")
    axes[1, 0].set(
        xlabel="Adam step",
        ylabel="fraction within ±1σ",
        ylim=(0, 1.03),
        title="Statistically adequate coefficients",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(
        steps, [row["minimum"] for row in history], label="minimum intensity"
    )
    negative = [row["negative_pixels"] for row in history]
    twin = axes[1, 1].twinx()
    twin.plot(steps, negative, color="tab:red", label="negative pixels")
    axes[1, 1].axhline(0, color="black", ls=":")
    axes[1, 1].set(
        xlabel="Adam step",
        ylabel="minimum [kJy sr$^{-1}$]",
        title="Required negative-value monitoring",
    )
    twin.set_ylabel("negative pixel count", color="tab:red")
    _save(
        fig,
        path,
        "Aim: expose optimizer stability, staged S2 activation, uncertainty-band "
        "convergence, and all negative reconstructed values. Positivity is never "
        "enforced.",
    )


def plot_artifact_and_morphology(
    target: np.ndarray,
    output: np.ndarray,
    path: Path,
    analysis_sigma_pixels: float = 4.0,
) -> None:
    """Compare high-pass texture and topology without adding repair losses."""

    target_reliable = gaussian_filter(
        target, analysis_sigma_pixels, mode="reflect"
    )
    output_reliable = gaussian_filter(
        output, analysis_sigma_pixels, mode="reflect"
    )
    target_hp = target_reliable - gaussian_filter(
        target_reliable, 4.0 * analysis_sigma_pixels, mode="reflect"
    )
    output_hp = output_reliable - gaussian_filter(
        output_reliable, 4.0 * analysis_sigma_pixels, mode="reflect"
    )
    limit = np.percentile(np.abs(np.concatenate((target_hp.ravel(), output_hp.ravel()))), 99)
    thresholds, target_euler, target_perimeter = morphology_curves(target_reliable)
    _, output_euler, output_perimeter = morphology_curves(output_reliable, thresholds)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], target_hp, "Target reliable-scale band-pass"),
        (axes[0, 1], output_hp, "Output reliable-scale band-pass"),
    ):
        shown = ax.imshow(
            image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(shown, ax=ax, shrink=0.75, label="kJy sr$^{-1}$")
    axes[1, 0].plot(thresholds, target_euler[0], label="target")
    axes[1, 0].plot(thresholds, output_euler[0], label="output")
    axes[1, 0].set(
        xlabel="threshold [robust σ]",
        ylabel="Euler characteristic",
        title=f"Excursion-set topology after σ={analysis_sigma_pixels:g} px smoothing",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].plot(thresholds, target_perimeter[0], label="target")
    axes[1, 1].plot(thresholds, output_perimeter[0], label="output")
    axes[1, 1].set(
        xlabel="threshold [robust σ]",
        ylabel="perimeter / area",
        title=f"Boundary complexity after σ={analysis_sigma_pixels:g} px smoothing",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    _save(
        fig,
        path,
        "Aim: reveal spikes, excessive smoothing, loops, or fragmented structures. "
        "High-pass and topology metrics are validation diagnostics only and never "
        "act as artifact-repair losses.",
    )
