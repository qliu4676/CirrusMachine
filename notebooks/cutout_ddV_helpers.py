"""Input preparation and small plotting helpers for the Cutout example."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _portable_header(header):
    """Keep scientific metadata while omitting machine-specific provenance."""
    return fits.Header(
        [
            card
            for card in header.cards
            if not any(
                prefix in str(card.value)
                for prefix in ("/Users/", "/home/", "/content/drive/", "\\Users\\")
            )
        ]
    )


def _same_grid(first, second, shape):
    """Check celestial alignment at the corners and center of a shared grid."""
    ny, nx = shape
    points = np.array(
        [
            [0, 0],
            [nx - 1, 0],
            [0, ny - 1],
            [nx - 1, ny - 1],
            [(nx - 1) / 2, (ny - 1) / 2],
        ]
    )
    a = WCS(first).celestial.all_pix2world(points, 0)
    b = WCS(second).celestial.all_pix2world(points, 0)
    if not np.allclose(a, b, atol=1e-7, rtol=0):
        raise ValueError("Input maps and masks have different celestial grids")


def export_inputs(
    directory, *, optical, mask_bits, FIR, FIR_masks, PSFs, effective_FWHM=36.0
):
    """Write a portable subset of the existing Cutout inputs, without resampling.

    The optical mask's bit 256 marks the target; all other bits mark sources.
    FIR samples/error/coverage remain native; MASK_RAD supplies excluded sources.
    effective_FWHM records the adopted FIR resolution, not a smoothing request.
    Existing directories are rejected. Original inputs are never changed.
    """
    paths = dict(
        optical=optical, mask_bits=mask_bits, FIR=FIR, FIR_masks=FIR_masks, PSFs=PSFs
    )
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    image, header = fits.getdata(optical, header=True)
    bits, mask_header = fits.getdata(mask_bits, header=True)
    if image.ndim != 2 or bits.shape != image.shape:
        raise ValueError("Optical image and mask shapes must agree")
    _same_grid(header, mask_header, image.shape)
    header = _portable_header(header)
    bits = bits.astype(np.int32)
    source, target = (bits & ~256) != 0, (bits & 256) != 0
    with fits.open(FIR) as hdus:
        FIR_hdus = [hdus[name].copy() for name in ("image", "error", "coverage")]
    holes, holes_header = fits.getdata(FIR_masks, "MASK_RAD", header=True)
    shape = FIR_hdus[0].data.shape
    if any(h.data.shape != shape for h in FIR_hdus) or holes.shape != shape:
        raise ValueError("FIR image, error, coverage and mask shapes must agree")
    _same_grid(FIR_hdus[0].header, holes_header, shape)
    for hdu in FIR_hdus:
        hdu.header = _portable_header(hdu.header)
    holes_header = _portable_header(holes_header)
    with np.load(PSFs, allow_pickle=False) as archive:
        kernels = {k: archive[k] for k in ("psf_native", "psf_coarse")}
    for kernel in kernels.values():
        if kernel.ndim != 2 or not np.isfinite(kernel).all() or kernel.sum() <= 0:
            raise ValueError("PSFs must be finite two-dimensional kernels")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    fits.writeto(directory / "optical.fits", image, header)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(source.astype("u1"), header, name="SOURCE"),
            fits.ImageHDU(target.astype("u1"), header, name="TARGET"),
        ]
    ).writeto(directory / "masks.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            *FIR_hdus,
            fits.ImageHDU(holes.astype("u1"), holes_header, name="MASK"),
        ]
    ).writeto(directory / "fir.fits")
    np.savez_compressed(directory / "psfs.npz", **kernels)
    metadata = dict(
        native_pixel_scale=0.3,
        working_pixel_scale=0.9,
        detector_psf_pixel_scale=0.1,
        effective_FWHM=effective_FWHM,
        beam_note="Adopted effective FIR resolution; original native samples retained.",
        sources={key: Path(path).name for key, path in paths.items()},
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())
        },
    )
    (directory / "inputs.json").write_text(json.dumps(metadata, indent=2))
    return directory


def load_inputs(directory):
    """Read the portable Cutout inputs; masks are True for excluded pixels."""
    directory = Path(directory)
    required = ("optical.fits", "masks.fits", "fir.fits", "psfs.npz", "inputs.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Cutout inputs in {directory}: {', '.join(missing)}"
        )
    metadata = json.loads((directory / "inputs.json").read_text())
    if "effective_fwhm" in metadata:
        metadata["effective_FWHM"] = metadata.pop("effective_fwhm")
    # Also sanitize manifests exported by older versions of this helper.
    metadata["sources"] = {
        key: Path(value.replace("\\", "/")).name
        for key, value in metadata.get("sources", {}).items()
    }
    for name in required[:-1]:
        if (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            != metadata["sha256"][name]
        ):
            raise ValueError(f"Input checksum differs from inputs.json: {name}")
    optical, header = fits.getdata(directory / "optical.fits", header=True)
    with fits.open(directory / "masks.fits") as hdus:
        source, target = (hdus[k].data.astype(bool) for k in ("SOURCE", "TARGET"))
    with fits.open(directory / "fir.fits") as hdus:
        arrays = {
            k: hdus[k].data.copy() for k in ("IMAGE", "ERROR", "COVERAGE", "MASK")
        }
        FIR_WCS = WCS(hdus["IMAGE"].header).celestial
    with np.load(directory / "psfs.npz", allow_pickle=False) as archive:
        PSFs = {k: archive[k] for k in archive.files}
    return dict(
        optical=optical,
        optical_WCS=WCS(header).celestial,
        source_mask=source,
        target_mask=target,
        PSF_working=PSFs["psf_coarse"],
        PSF_detector=PSFs["psf_native"],
        FIR_image=arrays["IMAGE"],
        FIR_error=arrays["ERROR"],
        FIR_coverage=arrays["COVERAGE"],
        FIR_mask=arrays["MASK"].astype(bool),
        FIR_WCS=FIR_WCS,
        metadata=metadata,
    )


def show_image(image, *, ax=None, valid=None, title=None, limits=None, cmap="gray"):
    """Display an ADU image, optionally hiding excluded pixels."""
    ax = plt.gca() if ax is None else ax
    shown = np.asarray(image, dtype=float)
    if valid is not None:
        shown = np.where(valid, shown, np.nan)
    if limits is None:
        limits = np.nanpercentile(shown, [2, 98])
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("0.75")
    artist = ax.imshow(shown, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set(xlabel="Working x (pixel)", ylabel="Working y (pixel)", title=title)
    ax.figure.colorbar(artist, ax=ax, label="ADU", shrink=0.8)
    return ax


def plot_surface_brightness(profiles, *, ax=None):
    """Plot measured profiles; nonpositive intensities have undefined magnitudes."""
    ax = plt.gca() if ax is None else ax
    for name, profile in profiles.items():
        ax.plot(profile["radius"], profile["surface_brightness"], label=name)
    ax.set(
        xlabel="Semimajor radius (arcsec)",
        ylabel="Surface brightness (mag arcsec$^{-2}$)",
    )
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.legend()
    return ax


def plot_cumulative_flux(measurements, *, ax=None):
    """Plot signed sums on the same source-clean aperture support."""
    ax = plt.gca() if ax is None else ax
    for name, light in measurements.items():
        ax.plot(light["radius"], light["flux"], label=name)
    ax.set(
        xlabel="Semimajor radius (arcsec)", ylabel="Cumulative flux (working-pixel ADU)"
    )
    ax.legend()
    return ax
