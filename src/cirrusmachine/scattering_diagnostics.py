"""Diagnostic plots for multiscale scattering-mean synthesis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .power_spectrum import power_spectrum_2d, radial_power_spectrum
from .scattering_mean import FullMeanFeature, MeanFeature


def unpack_isotropic_features(
    features: Sequence[MeanFeature], values: np.ndarray, J: int, L: int
) -> tuple[np.ndarray, np.ndarray]:
    """Place a sparse feature vector into ``S1[J]`` and ``S2[J,J,L]`` arrays."""

    s1 = np.full(J, np.nan)
    s2 = np.full((J, J, L), np.nan)
    for feature, value in zip(features, np.asarray(values), strict=True):
        if feature.order == 1:
            s1[feature.j1] = value
        else:
            s2[feature.j1, feature.j2, feature.delta_l] = value
    return s1, s2


def plot_image_triptych(
    herschel: np.ndarray,
    planck: np.ndarray,
    output: np.ndarray,
    path: str,
    output_title: str = "s_mean_iso synthesis",
    planck_title: str = "Converted PLANCK_RAD crop (5′)",
) -> None:
    """Save the required input-input-output image comparison.

    The shared stretch is derived from the high-resolution reference and
    synthesis, rather than the smooth Planck backbone, so bright and faint
    Herschel-scale structure remains visible without clipping their tails.
    """

    limits = np.nanpercentile(
        np.concatenate((herschel.ravel(), output.ravel())), (0.1, 99.9)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    panels = (
        (herschel, "Herschel cutout 4 (36″)"),
        (planck, planck_title),
        (output, output_title),
    )
    for ax, (image, title) in zip(axes, panels, strict=True):
        shown = ax.imshow(image, origin="lower", cmap="viridis", vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(shown, ax=axes, label="MJy sr$^{-1}$", shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_conditional_patch_validation(
    measured_log: np.ndarray,
    predicted_log: np.ndarray,
    radiance: np.ndarray,
    features: Sequence[MeanFeature | FullMeanFeature],
    path: str,
) -> None:
    """Compare each fixed output patch with its radiance-conditioned target.

    Each panel contains every fine scattering coefficient in one output patch.
    The target on the horizontal axis is evaluated from the patch-centre,
    5-arcmin-smoothed Planck radiance; it is therefore distinct for every
    panel even though the coefficient set is shared.
    """

    measured = np.asarray(measured_log, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    if measured.shape != predicted.shape or measured.ndim != 2:
        raise ValueError("measured_log and predicted_log must be matching 2-D arrays")
    if measured.shape != (radiance.size, len(features)):
        raise ValueError("radiance and features must match the coefficient arrays")

    count = measured.shape[0]
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    values = np.concatenate((measured.ravel(), predicted.ravel()))
    low, high = np.nanpercentile(values, (1, 99))
    margin = max((high - low) * 0.04, 1e-3)
    limits = (low - margin, high + margin)
    orders = np.asarray([feature.order for feature in features])

    fig, axes = plt.subplots(
        rows, columns, figsize=(4.1 * columns, 3.8 * rows), constrained_layout=True,
        squeeze=False,
    )
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.set_axis_off()
            continue
        for order, color, label in ((1, "tab:blue", "S1"), (2, "tab:orange", "S2")):
            use = orders == order
            ax.scatter(
                predicted[index, use], measured[index, use], s=12, alpha=0.7,
                color=color, label=label if index == 0 else None,
            )
        ax.plot(limits, limits, color="0.25", lw=1, ls="--")
        ax.set(xlim=limits, ylim=limits, aspect="equal")
        ax.set_title(f"Patch {index + 1}: R = {radiance[index]:.3g}")
        ax.grid(alpha=0.2)
        if index % columns == 0:
            ax.set_ylabel("measured log fine WST")
        if index // columns == rows - 1:
            ax.set_xlabel("radiance-predicted log fine WST")
    axes.flat[0].legend(loc="best", fontsize="small")
    fig.suptitle("Fixed-grid output patches: conditional fine-WST validation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scattering_statistics(
    datasets: Mapping[str, tuple[Sequence[MeanFeature], np.ndarray]],
    *,
    J: int,
    L: int,
    constrained_features: Sequence[MeanFeature],
    path: str,
) -> None:
    """Plot interpretable S1 curves and relative-angle S2 heatmaps."""

    unpacked = {
        name: unpack_isotropic_features(features, values, J, L)
        for name, (features, values) in datasets.items()
    }
    constrained = set(constrained_features)
    constrained_mask = np.zeros((J, J), dtype=bool)
    for feature in constrained:
        if feature.order == 2:
            constrained_mask[feature.j1, feature.j2] = True

    names = list(unpacked)
    fig = plt.figure(figsize=(4.0 * len(names), 3.0 * (L + 1)), constrained_layout=True)
    grid = fig.add_gridspec(L + 1, len(names))
    for column, name in enumerate(names):
        s1, s2 = unpacked[name]
        ax = fig.add_subplot(grid[0, column])
        ax.plot(np.arange(J), s1, marker="o")
        ax.set_title(name)
        ax.set_xlabel("j")
        ax.set_ylabel("log S1")
        ax.grid(alpha=0.25)
        for dl in range(L):
            ax = fig.add_subplot(grid[dl + 1, column])
            matrix = s2[:, :, dl]
            finite = np.isfinite(matrix)
            if finite.any():
                vmin, vmax = np.nanpercentile(matrix, (5, 95))
                if vmin == vmax:
                    vmax = vmin + 1e-6
                shown = ax.imshow(matrix, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
                fig.colorbar(shown, ax=ax, shrink=0.72)
            else:
                ax.imshow(np.zeros((J, J)), origin="lower", cmap="Greys", vmin=0, vmax=1)
            unsupported = np.isfinite(matrix) & ~constrained_mask
            yy, xx = np.where(unsupported)
            ax.scatter(xx, yy, marker="x", s=18, color="cyan", linewidths=0.7)
            ax.set_title(f"Δℓ={dl}; cyan × unconstrained")
            ax.set_xlabel("j2")
            ax.set_ylabel("j1")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_lexicographic_fit(
    groups: Sequence[tuple[str, Sequence[MeanFeature], np.ndarray]],
    output_log: np.ndarray,
    path: str,
) -> None:
    """Plot selected targets and output in their explicit lexicographic order."""

    target = np.concatenate([values for _, _, values in groups])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(target, color="black", lw=1.2, label="target")
    axes[0].plot(output_log, color="tab:orange", lw=0.9, label="output")
    axes[0].set_ylabel("log coefficient")
    axes[0].legend()
    fractional = np.expm1(output_log - target)
    axes[1].axhspan(-0.05, 0.05, color="tab:green", alpha=0.15, label="±5%")
    axes[1].plot(fractional, color="tab:blue", lw=0.9)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("fractional residual")
    axes[1].set_xlabel("constrained coefficient index")
    start = 0
    for name, features, _ in groups:
        stop = start + len(features)
        axes[0].axvspan(start, stop - 1, alpha=0.06)
        axes[0].text((start + stop - 1) / 2, 0.98, name, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
        axes[1].axvline(stop - 0.5, color="0.7", lw=0.6)
        start = stop
    axes[1].legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_power_and_artifacts(
    herschel: np.ndarray,
    planck: np.ndarray,
    output: np.ndarray,
    pixel_scale_arcmin: float,
    path: str,
) -> None:
    """Save independent power-spectrum, high-pass, and histogram checks."""

    from scipy.ndimage import gaussian_filter

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for image, label in ((herschel, "Herschel"), (planck, "Planck"), (output, "Output")):
        power = power_spectrum_2d(image)
        frequency, radial, _ = radial_power_spectrum(power, pixel_scale_arcmin, nbins=45)
        axes[0, 0].loglog(frequency, radial, label=label)
    axes[0, 0].axvline(1 / 0.6, color="0.5", ls="--", label="36″")
    axes[0, 0].axvline(1 / 5.0, color="0.5", ls=":", label="5′")
    axes[0, 0].set(xlabel="spatial frequency [arcmin$^{-1}$]", ylabel="power")
    axes[0, 0].legend(fontsize=8)

    highpass = output - gaussian_filter(output, 300.0 / 6.0 / 2.355, mode="reflect")
    limit = np.nanpercentile(np.abs(highpass), 99)
    shown = axes[0, 1].imshow(highpass, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0, 1].set_title("Output high-pass above 5′")
    axes[0, 1].set_axis_off()
    fig.colorbar(shown, ax=axes[0, 1], label="MJy sr$^{-1}$")

    for image, label in ((herschel, "Herschel"), (planck, "Planck"), (output, "Output")):
        axes[1, 0].hist(image.ravel(), bins=80, density=True, histtype="step", label=label)
    axes[1, 0].set(xlabel="MJy sr$^{-1}$", ylabel="density")
    axes[1, 0].legend(fontsize=8)

    delta = output - planck
    limit = np.nanpercentile(np.abs(delta), 99)
    shown = axes[1, 1].imshow(delta, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, 1].set_title("Output − converted Planck")
    axes[1, 1].set_axis_off()
    fig.colorbar(shown, ax=axes[1, 1], label="MJy sr$^{-1}$")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loss_history(history: Sequence[Mapping[str, float]], path: str) -> None:
    """Plot total loss and per-group fractional RMS histories."""

    steps = np.asarray([row["step"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].semilogy(steps, [row["loss"] for row in history])
    axes[0].set(xlabel="Adam step", ylabel="mean log-coefficient loss")
    keys = [key for key in history[0] if key.startswith("rms_")]
    for key in keys:
        axes[1].semilogy(steps, [row[key] for row in history], label=key.removeprefix("rms_"))
    axes[1].axhline(0.05, color="black", ls="--", label="5% target")
    axes[1].set(xlabel="Adam step", ylabel="fractional RMS")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ensemble_cross_validation(
    ensemble_log: np.ndarray,
    output_patch_log: np.ndarray,
    target_log: np.ndarray,
    uncertainty_log: np.ndarray,
    path: str,
) -> None:
    """Compare input-ensemble and fixed-grid output-patch cross coefficients."""

    target_log = np.asarray(target_log)
    input_fraction = np.expm1(np.asarray(ensemble_log) - target_log)
    output_fraction = np.expm1(np.asarray(output_patch_log) - target_log)
    sigma_fraction = np.expm1(np.asarray(uncertainty_log))
    x = np.arange(target_log.size)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].fill_between(x, -sigma_fraction, sigma_fraction, color="0.8", label="± mad_std")
    for row in input_fraction:
        axes[0].plot(x, row, color="tab:blue", alpha=0.35, lw=0.8)
    axes[0].plot(x, np.median(input_fraction, axis=0), color="black", lw=1.5, label="input median")
    axes[0].set_ylabel("Herschel fractional offset")
    axes[0].legend(fontsize=8)
    axes[1].fill_between(x, -sigma_fraction, sigma_fraction, color="0.8", label="± mad_std")
    for row in output_fraction:
        axes[1].plot(x, row, color="tab:orange", alpha=0.35, lw=0.8)
    axes[1].plot(x, np.median(output_fraction, axis=0), color="black", lw=1.5,
                 label="output-patch median")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set(xlabel="cross-scale coefficient index", ylabel="Output fractional offset")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_full_mean_fit(
    groups: Sequence[tuple[str, Sequence[FullMeanFeature], np.ndarray]],
    output_log: np.ndarray,
    *,
    J: int,
    L: int,
    path: str,
) -> None:
    """Visualize full-orientation S1 and S2 targets without flattening away meaning."""

    target = np.concatenate([values for _, _, values in groups])
    features = tuple(feature for _, group_features, _ in groups for feature in group_features)
    target_s1 = np.full((L, J), np.nan)
    output_s1 = np.full((L, J), np.nan)
    pairs = sorted({(f.j1, f.j2) for f in features if f.order == 2})
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    target_s2 = np.full((len(pairs), L * L), np.nan)
    output_s2 = np.full_like(target_s2, np.nan)
    for feature, target_value, output_value in zip(features, target, output_log, strict=True):
        if feature.order == 1:
            target_s1[feature.l1, feature.j1] = target_value
            output_s1[feature.l1, feature.j1] = output_value
        else:
            row = pair_index[(feature.j1, feature.j2)]
            column = feature.l1 * L + feature.l2
            target_s2[row, column] = target_value
            output_s2[row, column] = output_value
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = (
        (target_s1, "S1 target", "j", "l"),
        (output_s1, "S1 output", "j", "l"),
        (np.expm1(output_s1 - target_s1), "S1 fractional residual", "j", "l"),
        (target_s2, "S2 target", "(l1,l2)", "(j1,j2)"),
        (output_s2, "S2 output", "(l1,l2)", "(j1,j2)"),
        (np.expm1(output_s2 - target_s2), "S2 fractional residual", "(l1,l2)", "(j1,j2)"),
    )
    for ax, (array, title, xlabel, ylabel) in zip(axes.ravel(), panels, strict=True):
        if "residual" in title:
            limit = max(float(np.nanpercentile(np.abs(array), 95)), 0.05)
            shown = ax.imshow(array, origin="lower", aspect="auto", cmap="coolwarm",
                              vmin=-limit, vmax=limit)
        else:
            shown = ax.imshow(array, origin="lower", aspect="auto", cmap="magma")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if ylabel == "(j1,j2)":
            ax.set_yticks(np.arange(len(pairs)), [f"{a},{b}" for a, b in pairs], fontsize=7)
        fig.colorbar(shown, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)
