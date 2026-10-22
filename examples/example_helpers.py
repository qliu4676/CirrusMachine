"""Input preparation and small plotting helpers for the Cutout example."""

import hashlib
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from cirrusmachine import modeling as cm

logger = logging.getLogger("cirrusmachine.examples")


def setup_example(
    *, data_dir=None, output_dir=None, config_path=None, log_level="DEBUG"
):
    """Load this example's config and inputs, and create one output directory.

    Default paths are beside this helper: modeling.yaml, inputs/, and output/.
    Installation is a separate pip command; inputs are never downloaded here.
    """
    directory = Path(__file__).resolve().parent
    cm.configure_logging(log_level)
    config_path = Path(config_path or directory / "modeling.yaml").expanduser()
    config = cm.load_config(config_path, cpu_threads=4)
    data_dir = Path(
        data_dir or os.environ.get("CIRRUSMACHINE_DATA") or directory / "inputs"
    ).expanduser()
    inputs = load_inputs(data_dir)
    meta = inputs["metadata"]
    if not np.isclose(config.beam_FWHM, meta["effective_FWHM"]):
        raise ValueError(
            "The configured coarse resolution differs from the input manifest."
        )
    if "working_pixel_scale" in meta and not np.isclose(
        config.pixel_scale, meta["working_pixel_scale"]
    ):
        raise ValueError(
            "The configured working pixel scale differs from the input manifest."
        )
    output_dir = Path(output_dir or directory / "output").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return config, inputs, output_dir


def load_or_calibrate(field, config, path):
    """Reuse a matching cache; a stale cache raises rather than being overwritten."""
    path = Path(path)
    if path.is_file():
        return cm.load_calibration(path, field, config)
    calibration = cm.calibrate_statistics(field, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm.save_calibration(calibration, path)
    return calibration


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
    directory, *, optical, mask_bits, coarse, coarse_mask, PSFs, effective_FWHM=36.0
):
    """Write a portable subset of the existing Cutout inputs, without resampling.

    The optical mask's bit 256 marks the target; all other bits mark sources.
    Coarse samples/error/coverage remain native; MASK_RAD supplies excluded sources.
    effective_FWHM records the adopted coarse resolution.
    """
    paths = dict(
        optical=optical,
        mask_bits=mask_bits,
        coarse=coarse,
        coarse_mask=coarse_mask,
        PSFs=PSFs,
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
    with fits.open(coarse) as hdus:
        coarse_hdus = [hdus[name].copy() for name in ("image", "error", "coverage")]
    holes, holes_header = fits.getdata(coarse_mask, "MASK_RAD", header=True)
    shape = coarse_hdus[0].data.shape
    if any(h.data.shape != shape for h in coarse_hdus) or holes.shape != shape:
        raise ValueError("Coarse image, error, coverage and mask shapes must agree")
    _same_grid(coarse_hdus[0].header, holes_header, shape)
    for hdu in coarse_hdus:
        hdu.header = _portable_header(hdu.header)
    holes_header = _portable_header(holes_header)
    with np.load(PSFs, allow_pickle=False) as archive:
        kernels = {k: archive[k] for k in ("psf_native", "psf_coarse")}
    for kernel in kernels.values():
        if kernel.ndim != 2 or not np.isfinite(kernel).all() or kernel.sum() <= 0:
            raise ValueError("PSFs must be finite two-dimensional kernels")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    fits.writeto(directory / "Optical.fits", image, header)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(source.astype("u1"), header, name="SOURCE"),
            fits.ImageHDU(target.astype("u1"), header, name="TARGET"),
        ]
    ).writeto(directory / "Mask.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            *coarse_hdus,
            fits.ImageHDU(holes.astype("u1"), holes_header, name="MASK"),
        ]
    ).writeto(directory / "FIR.fits")
    np.savez_compressed(directory / "PSF.npz", **kernels)
    metadata = dict(
        native_pixel_scale=0.3,
        working_pixel_scale=0.9,
        detector_psf_pixel_scale=0.1,
        effective_FWHM=effective_FWHM,
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
    required = ("Optical.fits", "Mask.fits", "FIR.fits", "PSF.npz", "inputs.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Cutout inputs in {directory}: {', '.join(missing)}"
        )
    metadata = json.loads((directory / "inputs.json").read_text())
    if "effective_fwhm" in metadata:
        metadata["effective_FWHM"] = metadata.pop("effective_fwhm")
    # Omit optional provenance fields from older input bundles.
    metadata.pop("beam_note", None)
    metadata.pop("sources", None)
    for name in required[:-1]:
        if (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            != metadata["sha256"][name]
        ):
            raise ValueError(f"Input checksum differs from inputs.json: {name}")
    optical, header = fits.getdata(directory / "Optical.fits", header=True)
    with fits.open(directory / "Mask.fits") as hdus:
        source, target = (hdus[k].data.astype(bool) for k in ("SOURCE", "TARGET"))
    with fits.open(directory / "FIR.fits") as hdus:
        arrays = {
            k: hdus[k].data.copy() for k in ("IMAGE", "ERROR", "COVERAGE", "MASK")
        }
        coarse_WCS = WCS(hdus["IMAGE"].header).celestial
    with np.load(directory / "PSF.npz", allow_pickle=False) as archive:
        PSFs = {k: archive[k] for k in archive.files}
    logger.info(
        "Loaded optical %s and coarse guide %s; input verified",
        optical.shape,
        arrays["IMAGE"].shape,
    )
    return dict(
        optical=optical,
        optical_WCS=WCS(header).celestial,
        source_mask=source,
        target_mask=target,
        PSF_working=PSFs["psf_coarse"],
        PSF_detector=PSFs["psf_native"],
        coarse_image=arrays["IMAGE"],
        coarse_error=arrays["ERROR"],
        coarse_coverage=arrays["COVERAGE"],
        coarse_mask=arrays["MASK"].astype(bool),
        coarse_WCS=coarse_WCS,
        metadata=metadata,
    )


def measure_comparison(
    images, field, config, *, corrected_sky, SB_ZP, radial_bin_edges
):
    """Compare truth, injected and corrected light on the same aperture support.

    ``Injected`` is the observed field plus the injected model. Radial bin
    edges are elliptical semimajor radii in arcsec.
    """
    crop = field.crop
    valid, radius = field.source_clean[crop], field.radius[crop]
    injected_sky = cm.estimate_sky(
        images["Injected"],
        valid,
        radius,
        config.sky_annulus,
        device=config.device,
    )
    skies = {
        "Truth": 0.0,
        "Injected": injected_sky["value_ADU"],
        "Corrected": corrected_sky,
    }
    profiles, cumulative = {}, {}
    for name, image in images.items():
        profiles[name] = cm.measure_profile(
            image,
            valid,
            radius,
            radial_bin_edges,
            sky=skies[name],
            SB_ZP=SB_ZP,
            device=config.device,
        )
        cumulative[name] = cm.measure_cumulative_light(
            image,
            valid,
            radius,
            radial_bin_edges[1:],
            sky=skies[name],
            pixel_scale=field.pixel_scale,
            device=config.device,
        )
    logger.info(
        "Sky (ADU): truth %.4g, injected %.4g, corrected %.4g",
        skies["Truth"],
        skies["Injected"],
        skies["Corrected"],
    )
    return profiles, cumulative


def show_image(
    image,
    *,
    ax=None,
    valid=None,
    title=None,
    limits=None,
    cmap="gray",
    mask_color="#283544",
):
    """Display an image, optionally hiding excluded pixels."""
    ax = plt.gca() if ax is None else ax
    shown = np.asarray(image, dtype=float)
    if valid is not None:
        shown = np.where(valid, shown, np.nan)
    if limits is None:
        limits = np.nanpercentile(shown, [2, 99])
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(mask_color)
    artist = ax.imshow(shown, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set(xlabel="X [pixel]", ylabel="Y [pixel]", title=title)
    ax.figure.colorbar(artist, ax=ax, label="ADU", shrink=0.8)
    return ax


def plot_surface_brightness(profiles, *, ax=None):
    """Plot measured profiles; nonpositive intensities have undefined magnitudes."""
    ax = plt.gca() if ax is None else ax
    for name, profile in profiles.items():
        ax.plot(profile["radius"], profile["surface_brightness"], label=name)
    ax.set(
        xlabel="Semi-major radius (arcsec)",
        ylabel="Surface brightness (mag arcsec$^{-2}$)",
    )
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.legend()
    ax.set_ylim(31, 19)
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
