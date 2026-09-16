"""Explanatory plots for the WST synthesis preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def _save(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_input_and_mask(
    image: np.ndarray,
    mask_fraction: np.ndarray,
    path: Path,
    *,
    target_name: str = "Dragonfly",
    unit: str = "kJy/sr",
) -> None:
    """Show the analyzed image, inpainting footprint, and fine-scale response."""

    highpass = image - gaussian_filter(image, 1.0, mode="reflect")
    limits = np.percentile(image, (0.5, 99.5))
    hp_limit = np.percentile(np.abs(highpass), 99.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    shown = axes[0].imshow(
        image, origin="lower", cmap="viridis", vmin=limits[0], vmax=limits[1]
    )
    axes[0].set_title(f"3×3-binned {target_name} target")
    fig.colorbar(shown, ax=axes[0], label=unit, shrink=0.8)
    shown = axes[1].imshow(mask_fraction, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Fraction inpainted per binned pixel")
    fig.colorbar(shown, ax=axes[1], label="mask fraction", shrink=0.8)
    shown = axes[2].imshow(
        highpass,
        origin="lower",
        cmap="coolwarm",
        vmin=-hp_limit,
        vmax=hp_limit,
    )
    axes[2].set_title("One-pixel high-pass diagnostic")
    fig.colorbar(shown, ax=axes[2], label=unit, shrink=0.8)
    for ax in axes:
        ax.set_axis_off()
    _save(
        fig,
        path,
        "Aim: locate inpainted regions and visually test whether their fine-scale "
        "texture differs from unmasked cirrus. The mask is used only for target "
        "uncertainty; it will not be supplied to synthesis.",
    )


def plot_power_model(power: Mapping[str, np.ndarray], path: Path) -> None:
    """Plot the hybrid spectrum fit, local slope, and ensemble draws."""

    frequency = power["frequency"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    shown = axes[0, 0].imshow(
        np.log10(np.maximum(power["power_2d"], np.finfo(float).tiny)),
        origin="lower",
        cmap="magma",
    )
    axes[0, 0].set_title("Tapered 2-D log power")
    axes[0, 0].set_axis_off()
    fig.colorbar(shown, ax=axes[0, 0], label="log$_{10}$ power", shrink=0.8)

    axes[0, 1].loglog(frequency, power["radial_power"], color="black", label="target")
    axes[0, 1].loglog(
        frequency, power["fitted_power"], color="tab:red", lw=2, label="hybrid fit"
    )
    for index, sample in enumerate(power["sampled_radial"][:5]):
        axes[0, 1].loglog(
            frequency,
            sample,
            color="tab:blue",
            alpha=0.25,
            label="ensemble draws" if index == 0 else None,
        )
    axes[0, 1].set(
        xlabel="spatial frequency [arcmin$^{-1}$]",
        ylabel="power",
        title="Target, smooth model, and sampled spectra",
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].semilogx(frequency, power["local_slope"], marker="o", ms=3)
    axes[1, 0].axhline(
        float(power["large_scale_slope"]),
        color="tab:red",
        ls="--",
        label=f"4–30′ slope = {float(power['large_scale_slope']):.2f}",
    )
    axes[1, 0].set(
        xlabel="spatial frequency [arcmin$^{-1}$]",
        ylabel=r"$d\log P/d\log k$",
        title="Scale-dependent spectral index",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].semilogx(
        frequency, power["quadrupole"], marker="o", ms=3, label="quadrupole amplitude"
    )
    axes[1, 1].semilogx(
        frequency,
        power["axial_ratio"],
        marker=".",
        alpha=0.7,
        label="axis/diagonal power",
    )
    axes[1, 1].axhline(1.0, color="0.4", ls=":")
    axes[1, 1].set(
        xlabel="spatial frequency [arcmin$^{-1}$]",
        ylabel="anisotropy diagnostic",
        title="Physical anisotropy versus FFT-axis excess",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    _save(
        fig,
        path,
        "Aim: verify that the hybrid power-law model captures the ensemble spectrum "
        "without copying individual Fourier amplitudes. Blue curves show natural "
        "realization variance; slope and axial panels reveal breaks, filtering, or "
        "boundary-aligned artifacts.",
    )


def plot_cdf_stability(cdf: Mapping[str, np.ndarray], path: Path) -> None:
    """Show smoothing, patch, and mask sensitivity of the robust CDF."""

    levels = cdf["quantile_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for sigma, values in zip(
        cdf["smoothing_sigma_pixels"], cdf["quantiles"], strict=True
    ):
        axes[0].plot(levels, values, label=f"Gaussian σ={sigma:g} px")
    axes[0].axvspan(0, 0.005, color="0.8")
    axes[0].axvspan(0.995, 1, color="0.8")
    axes[0].set(
        xlabel="quantile level",
        ylabel="intensity [kJy sr$^{-1}$]",
        title="Robust one-point CDF versus analysis resolution",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    for sigma, shifts in zip(
        cdf["smoothing_sigma_pixels"], cdf["mask_shift_sigma"], strict=True
    ):
        axes[1].plot(levels, shifts[0], label=f"real mask, σ={sigma:g}px")
        axes[1].plot(levels, shifts[1], ls=":", alpha=0.6)
    axes[1].axhline(0.5, color="black", ls="--", label="0.5σ gate")
    axes[1].set(
        xlabel="quantile level",
        ylabel="target shift / patch MAD",
        title="Mask sensitivity; dotted curves are null-mask 95%",
    )
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.2)
    _save(
        fig,
        path,
        "Aim: select a CDF resolution above the noise/PSF reliability cutoff and "
        "confirm that central quantiles are stable. The extreme 0.5% tails are "
        "shown only as excluded grey regions and are not synthesis targets.",
    )


def plot_scale_reliability(
    reliability: Mapping[str, np.ndarray], path: Path, L: int
) -> None:
    """Show the evidence and explicit decision for every dyadic scale."""

    j = reliability["j"]
    labels = [f"j={value}\n{scale:g}′" for value, scale in zip(
        j, reliability["scale_arcmin"], strict=True
    )]
    colors = {
        "constrain": "tab:green",
        "downweight": "tab:orange",
        "validation-only": "tab:blue",
        "exclude": "tab:red",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(j, reliability["mask_sensitivity_sigma"], "o-", label="mask")
    axes[0].plot(j, reliability["boundary_sensitivity_sigma"], "s-", label="boundary")
    axes[0].axhline(0.5, color="0.4", ls="--", label="mask gate")
    axes[0].axhline(1.0, color="0.4", ls=":", label="boundary gate")
    axes[0].set(
        ylabel="median target shift / patch MAD",
        title=f"L={L}: measurement sensitivity by outer wavelet scale",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    status = reliability["status"]
    axes[1].bar(j, np.ones_like(j), color=[colors[value] for value in status])
    for x, state in zip(j, status, strict=True):
        axes[1].text(x, 0.5, state.replace("-", "\n"), ha="center", va="center", fontsize=8)
    axes[1].set(xticks=j, xticklabels=labels, yticks=[], ylim=(0, 1))
    axes[1].set_title("Preflight action stated explicitly for every scale")
    _save(
        fig,
        path,
        "Aim: prevent silent scale removal. Green scales are candidate constraints; "
        "orange scales receive larger uncertainty; blue scales are diagnostics only; "
        "red scales are below the defensible latent-morphology resolution.",
    )


def plot_wst_sensitivity(
    wst: Mapping[str, np.ndarray], path: Path, L: int
) -> None:
    """Show coefficient-level mask and boundary sensitivity."""

    order = wst["feature_order"]
    j2 = np.where(order == 1, wst["feature_j1"], wst["feature_j2"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for value, color, label in ((1, "tab:blue", "S1"), (2, "tab:orange", "S2")):
        use = (order == value) & np.isfinite(wst["mask_shift_sigma"])
        axes[0].scatter(
            wst["mask_shift_sigma"][use],
            wst["boundary_shift_sigma"][use],
            c=j2[use],
            cmap="viridis",
            s=20,
            alpha=0.7,
            label=label,
        )
    axes[0].axvline(0.5, color="black", ls="--")
    axes[0].axhline(1.0, color="black", ls=":")
    axes[0].set(
        xlabel="real-mask shift / patch MAD",
        ylabel="boundary shift / patch MAD",
        title=f"L={L}: coefficient reliability",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    finite = np.isfinite(wst["mask_shift_sigma"])
    axes[1].hist(
        wst["mask_shift_sigma"][finite],
        bins=25,
        alpha=0.6,
        label="real mask",
    )
    axes[1].hist(
        wst["null_mask_p95_sigma"][finite],
        bins=25,
        histtype="step",
        lw=2,
        label="translated/rotated null masks, 95%",
    )
    axes[1].axvline(0.5, color="black", ls="--")
    axes[1].set(
        xlabel="target shift / patch MAD",
        ylabel="coefficient count",
        title="Does the supplied mask have exceptional influence?",
    )
    axes[1].legend(fontsize=8)
    _save(
        fig,
        path,
        "Aim: identify WST coefficients whose target values depend on inpainting or "
        "unknown boundaries. Mask geometry is never matched spatially; sensitivity "
        "only enlarges target uncertainty or triggers rollback to mask-blind targets.",
    )


def plot_orientation_comparison(
    wst4: Mapping[str, np.ndarray], wst8: Mapping[str, np.ndarray], path: Path
) -> None:
    """Compare common relative angles between L=4 and L=8."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for wst, L, ax in ((wst4, 4, axes[0]), (wst8, 8, axes[1])):
        order = wst["feature_order"]
        values = wst["target"]
        dl = wst["feature_delta_l"]
        pairs = list(
            dict.fromkeys(
                zip(
                    wst["feature_j1"][order == 2],
                    wst["feature_j2"][order == 2],
                    strict=True,
                )
            )
        )
        curves = []
        for j1, j2 in pairs:
            use = (
                (order == 2)
                & (wst["feature_j1"] == j1)
                & (wst["feature_j2"] == j2)
            )
            sequence = values[use][np.argsort(dl[use])]
            curves.append(sequence - np.mean(sequence))
        curves = np.asarray(curves)
        angle = np.arange(L) * 180.0 / L
        for curve in curves:
            ax.plot(angle, curve, color="0.7", alpha=0.25)
        ax.plot(angle, np.nanmedian(curves, axis=0), "o-", color="tab:red", lw=2)
        ax.set(
            xlabel="relative orientation [deg]",
            ylabel="log S2 minus pair mean",
            title=f"L={L} relative-angle structure",
        )
        ax.grid(alpha=0.2)
    _save(
        fig,
        path,
        "Aim: test whether L=8 resolves repeatable relative-angle structure absent "
        "at L=4. Grey lines are scale pairs; red is their median. Only mirror-unique "
        "and uncertainty-stable channels will be considered for optimization.",
    )


def plot_self_modulation(
    modulation: Mapping[str, np.ndarray], path: Path
) -> None:
    """Show empirical local RMS versus large-scale intensity."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for scale, x, y, accepted in zip(
        modulation["scale_arcmin"],
        modulation["intensity_bin_center"],
        modulation["local_rms_bin_median"],
        modulation["accepted"],
        strict=True,
    ):
        axes[0].plot(
            x,
            y,
            "o-",
            label=f"{scale:g}′ {'relation stable' if accepted else 'not stable'}",
        )
    axes[0].set(
        xlabel="local low-pass intensity [kJy sr$^{-1}$]",
        ylabel="median local detail RMS [kJy sr$^{-1}$]",
        title="Empirical self-modulation relation",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    x = np.arange(len(modulation["scale_j"]))
    axes[1].bar(x, modulation["spearman_rho"], alpha=0.5, label="full field")
    for quadrant in modulation["quadrant_rho"].T:
        axes[1].plot(x, quadrant, "o--", alpha=0.5)
    axes[1].axhline(0.3, color="black", ls="--")
    axes[1].axhline(-0.3, color="black", ls="--")
    axes[1].set(
        xticks=x,
        xticklabels=[f"j={j}\n{s:g}′" for j, s in zip(
            modulation["scale_j"], modulation["scale_arcmin"], strict=True
        )],
        ylabel="Spearman correlation",
        title="Spatial stability across four quadrants",
    )
    axes[1].grid(alpha=0.2)
    _save(
        fig,
        path,
        "Aim: decide whether brighter large-scale cirrus robustly carries stronger "
        "small-scale fluctuations. Self-modulation is enabled only when the sign and "
        "strength remain stable across spatial quadrants; MD07 is omitted here.",
    )


def plot_timing(
    timing: Mapping[int, Mapping[str, np.ndarray]], path: Path
) -> None:
    """Plot projected CPU runtime for candidate angular resolutions."""

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    width = 0.35
    for offset, (L, result) in zip((-width / 2, width / 2), timing.items(), strict=True):
        x = np.arange(len(result["output_sizes"]))
        ax.bar(
            x + offset,
            result["projected_300_step_minutes"],
            width,
            label=f"L={L}, {int(result['feature_count'])} features",
        )
    ax.axhspan(0, 15, color="tab:green", alpha=0.10, label="preferred ≤15 min")
    ax.axhspan(20, 30, color="tab:orange", alpha=0.10, label="warning 20–30 min")
    ax.axhline(20, color="tab:orange", ls="--")
    ax.axhline(30, color="tab:red", ls="--")
    first = next(iter(timing.values()))
    ax.set(
        xticks=np.arange(len(first["output_sizes"])),
        xticklabels=[
            f"{int(size)}² output\n{int(canvas)}² guarded"
            for size, canvas in zip(
                first["output_sizes"], first["guarded_sizes"], strict=True
            )
        ],
        ylabel="projected 300-step wall time [min]",
        title="CPU projection from one differentiable WST step",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _save(
        fig,
        path,
        "Aim: determine whether L=8 and the evidence-selected scale set fit the soft "
        "CPU budget before any optimization. Projection includes a 25% allowance for "
        "Adam, CDF/power losses, validation, and checkpointing.",
    )
