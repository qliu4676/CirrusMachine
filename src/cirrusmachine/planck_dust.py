"""Planck thermal-dust modified-blackbody predictions on image WCS grids."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
from urllib.request import Request, urlopen

import healpy as hp
import numpy as np
from astropy.constants import c, h, k_B
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


PLANCK_REFERENCE_FREQUENCY_HZ = 353.0e9
PLANCK_R1_THERMAL_DUST_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps/maps/"
    "HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
)


def modified_blackbody_mjy_sr(
    tau353: np.ndarray,
    temperature_k: np.ndarray,
    beta: np.ndarray,
    wavelength_um: float = 250.0,
) -> np.ndarray:
    """Evaluate the Planck dust MBB model at one monochromatic wavelength.

    The convention is ``I_nu = tau353 * (nu/353 GHz)**beta * B_nu(T)``.
    The returned surface brightness is in MJy/sr.
    """
    tau353 = np.asarray(tau353, dtype=float)
    temperature_k = np.asarray(temperature_k, dtype=float)
    beta = np.asarray(beta, dtype=float)
    frequency_hz = c.value / (float(wavelength_um) * 1.0e-6)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exponent = h.value * frequency_hz / (k_B.value * temperature_k)
        planck_si = 2.0 * h.value * frequency_hz**3 / c.value**2 / np.expm1(exponent)
        intensity = tau353 * (frequency_hz / PLANCK_REFERENCE_FREQUENCY_HZ) ** beta
        intensity *= planck_si * 1.0e20  # W m^-2 Hz^-1 sr^-1 -> MJy sr^-1
    valid = (tau353 >= 0.0) & (temperature_k > 0.0) & np.isfinite(beta)
    return np.where(valid, intensity, np.nan)


def modified_blackbody_band_to_I100(
    temperature_k: np.ndarray,
    beta: np.ndarray,
    band_wavelength_um: float,
) -> np.ndarray:
    """Return the physical multiplier converting one FIR band to ``I_100``.

    Optical depth cancels in the monochromatic modified-blackbody ratio. The
    returned map satisfies ``I100 = I_band * band_to_I100`` under the Planck
    single-MBB model. Optical bands should instead supply this factor from an
    appropriate diffuse-Galactic-light model.
    """

    temperature = np.asarray(temperature_k, dtype=float)
    emissivity_index = np.asarray(beta, dtype=float)
    unit_tau = np.ones(np.broadcast_shapes(temperature.shape, emissivity_index.shape))
    i100 = modified_blackbody_mjy_sr(unit_tau, temperature, emissivity_index, 100.0)
    iband = modified_blackbody_mjy_sr(
        unit_tau, temperature, emissivity_index, float(band_wavelength_um)
    )
    return np.divide(
        i100,
        iband,
        out=np.full(np.broadcast_shapes(i100.shape, iband.shape), np.nan, dtype=float),
        where=np.isfinite(iband) & (iband > 0.0),
    )


def sample_planck_dust_fields(
    planck_fits: str | Path,
    target_wcs: WCS,
    shape: tuple[int, int],
    fields: Iterable[str] = ("TAU353", "TEMP", "BETA"),
    hdu: int | str = 1,
) -> dict[str, np.ndarray]:
    """Bilinearly sample Planck HEALPix table fields on a celestial WCS grid.

    Sampling uses HEALPix interpolation weights and accesses only the needed
    table rows, avoiding full copies of the 1.6 GB NSIDE=2048 product.
    """
    celestial = target_wcs.celestial
    y, x = np.indices(shape, dtype=float)
    ra_deg, dec_deg = celestial.pixel_to_world_values(x, y)
    galactic = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs").galactic

    with fits.open(planck_fits, memmap=True) as hdul:
        table_hdu = hdul[hdu]
        header = table_hdu.header
        if header.get("PIXTYPE", "").strip().upper() != "HEALPIX":
            raise ValueError(f"HDU {hdu!r} is not a HEALPix table")
        nside = int(header["NSIDE"])
        nest = header.get("ORDERING", "RING").strip().upper() == "NESTED"
        pixels, weights = hp.get_interp_weights(
            nside,
            galactic.l.to_value("deg"),
            galactic.b.to_value("deg"),
            nest=nest,
            lonlat=True,
        )
        result: dict[str, np.ndarray] = {}
        names = {name.upper(): name for name in table_hdu.columns.names}
        for requested in fields:
            key = requested.upper()
            if key not in names:
                raise KeyError(f"HEALPix field {requested!r} is not present")
            values = np.asarray(table_hdu.data[names[key]][pixels], dtype=float)
            bad = (~np.isfinite(values)) | (values <= hp.UNSEEN / 2.0)
            values[bad] = np.nan
            good_weights = np.where(bad, 0.0, weights)
            weight_sum = good_weights.sum(axis=0)
            sampled = np.divide(
                np.nansum(values * good_weights, axis=0),
                weight_sum,
                out=np.full(shape, np.nan, dtype=float),
                where=weight_sum > 0.0,
            )
            result[key] = sampled
    return result


def _http_byte_range(url: str, start: int, stop: int) -> bytes:
    """Read one inclusive HTTP byte range and reject accidental full downloads."""

    request = Request(url, headers={"Range": f"bytes={int(start)}-{int(stop)}"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range", "")
    expected = int(stop) - int(start) + 1
    if len(payload) != expected or not content_range.lower().startswith("bytes "):
        raise OSError(
            "The remote server did not honor the FITS byte-range request; "
            "refusing to download the complete HEALPix product."
        )
    return payload


def _fits_header_from_prefix(prefix: bytes, offset: int) -> tuple[fits.Header, int]:
    """Parse one padded FITS header contained in a downloaded prefix."""

    for position in range(int(offset), len(prefix), 80):
        if prefix[position : position + 8] == b"END     ":
            raw_length = position + 80 - int(offset)
            padded_length = int(np.ceil(raw_length / 2880.0) * 2880)
            end = int(offset) + padded_length
            if end > len(prefix):
                raise ValueError("remote prefix does not contain the full FITS header")
            header = fits.Header.fromstring(
                prefix[int(offset) : end].decode("ascii"), sep=""
            )
            return header, end
    raise ValueError("FITS END card was not found in remote prefix")


def _tform_nbytes(value: str) -> int:
    """Return fixed-width bytes occupied by one scalar/vector TFORM column."""

    match = re.fullmatch(r"\s*(\d*)([LXBIJKAEDCM])\s*", str(value).upper())
    if match is None:
        raise ValueError(f"Unsupported remote FITS TFORM {value!r}")
    count = int(match.group(1) or 1)
    code = match.group(2).replace(" ", "")
    sizes = {
        "L": 1,
        "X": 1,  # not used by the Planck product
        "B": 1,
        "I": 2,
        "J": 4,
        "K": 8,
        "A": 1,
        "E": 4,
        "D": 8,
        "C": 8,
        "M": 16,
    }
    return count * sizes[code]


def _merged_pixel_runs(
    pixels: np.ndarray, *, maximum_gap_rows: int = 4096
) -> tuple[tuple[int, int], ...]:
    """Merge nearby HEALPix rows to keep remote range requests inexpensive."""

    values = np.unique(np.asarray(pixels, dtype=np.int64))
    if values.size == 0:
        return ()
    runs: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for item in values[1:]:
        current = int(item)
        if current - previous > int(maximum_gap_rows) + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return tuple(runs)


def sample_remote_planck_radiance(
    url: str,
    target_wcs: WCS,
    shape: tuple[int, int],
    *,
    maximum_gap_rows: int = 4096,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample remote HEALPix radiance using small HTTP byte-range requests.

    This is intended for the row-oriented Planck thermal-dust table.  It avoids
    downloading the 1.5 GB all-sky product and returns provenance describing
    the actual transferred ranges.
    """

    prefix = _http_byte_range(str(url), 0, 65535)
    primary, extension_offset = _fits_header_from_prefix(prefix, 0)
    if int(primary.get("NAXIS", 0)) != 0:
        raise ValueError("remote Planck FITS primary HDU unexpectedly contains data")
    header, data_offset = _fits_header_from_prefix(prefix, extension_offset)
    if header.get("PIXTYPE", "").strip().upper() != "HEALPIX":
        raise ValueError("remote FITS extension is not a HEALPix table")
    names = [
        str(header[f"TTYPE{index}"]).strip().upper()
        for index in range(1, int(header["TFIELDS"]) + 1)
    ]
    if "RADIANCE" not in names:
        raise KeyError("remote HEALPix table has no RADIANCE column")
    radiance_index = names.index("RADIANCE") + 1
    column_offset = sum(
        _tform_nbytes(header[f"TFORM{index}"])
        for index in range(1, radiance_index)
    )
    radiance_format = str(header[f"TFORM{radiance_index}"]).strip().upper()
    if radiance_format not in {"E", "1E"}:
        raise ValueError("remote RADIANCE column is not one float32 per row")
    row_bytes = int(header["NAXIS1"])
    nrows = int(header["NAXIS2"])
    nside = int(header["NSIDE"])
    nest = header.get("ORDERING", "RING").strip().upper() == "NESTED"

    y, x = np.indices(shape, dtype=float)
    ra_deg, dec_deg = target_wcs.celestial.pixel_to_world_values(x, y)
    galactic = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs").galactic
    pixels, weights = hp.get_interp_weights(
        nside,
        galactic.l.to_value("deg"),
        galactic.b.to_value("deg"),
        nest=nest,
        lonlat=True,
    )
    unique = np.unique(pixels)
    if unique.min() < 0 or unique.max() >= nrows:
        raise ValueError("computed HEALPix row lies outside the remote table")
    lookup: dict[int, float] = {}
    ranges = _merged_pixel_runs(unique, maximum_gap_rows=maximum_gap_rows)
    transferred = len(prefix)
    for first, last in ranges:
        start = data_offset + first * row_bytes
        stop = data_offset + (last + 1) * row_bytes - 1
        payload = _http_byte_range(str(url), start, stop)
        transferred += len(payload)
        rows = np.frombuffer(payload, dtype=np.uint8).reshape(last - first + 1, row_bytes)
        values = rows[:, column_offset : column_offset + 4].copy().view(">f4").reshape(-1)
        requested = unique[(unique >= first) & (unique <= last)]
        for pixel in requested:
            lookup[int(pixel)] = float(values[int(pixel) - first])
    sampled_values = np.asarray(
        [lookup[int(pixel)] for pixel in pixels.ravel()], dtype=float
    ).reshape(pixels.shape)
    bad = (~np.isfinite(sampled_values)) | (sampled_values <= hp.UNSEEN / 2.0)
    sampled_values[bad] = np.nan
    good_weights = np.where(bad, 0.0, weights)
    weight_sum = good_weights.sum(axis=0)
    result = np.divide(
        np.nansum(sampled_values * good_weights, axis=0),
        weight_sum,
        out=np.full(shape, np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    provenance = {
        "url": str(url),
        "nside": nside,
        "ordering": "NESTED" if nest else "RING",
        "unique_healpix_rows": int(unique.size),
        "http_range_requests": int(1 + len(ranges)),
        "bytes_transferred": int(transferred),
        "full_product_bytes": int(data_offset + nrows * row_bytes),
    }
    return result, provenance


def predict_planck_250_on_wcs(
    planck_fits: str | Path,
    target_header: fits.Header,
    shape: tuple[int, int],
    wavelength_um: float = 250.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Project Planck dust parameters and predict monochromatic 250 um emission."""
    fields = sample_planck_dust_fields(
        planck_fits,
        WCS(target_header),
        shape,
        fields=("TAU353", "TEMP", "BETA", "RADIANCE"),
    )
    prediction = modified_blackbody_mjy_sr(
        fields["TAU353"], fields["TEMP"], fields["BETA"], wavelength_um
    )
    return prediction, fields


def centered_square_wcs_header(
    reference_header: fits.Header,
    reference_shape: tuple[int, int],
    width_deg: float = 1.0,
) -> tuple[fits.Header, tuple[int, int]]:
    """Expand or crop a celestial WCS to a centered square of fixed width.

    The pixel grid, sky orientation, and central sky coordinate are preserved.
    The returned width is rounded to the nearest integer number of pixels.
    """

    if width_deg <= 0.0:
        raise ValueError("width_deg must be positive")
    wcs = WCS(reference_header).celestial
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float)
    pixel_scale_deg = float(np.sqrt(np.prod(scales)))
    size = int(round(float(width_deg) / pixel_scale_deg))
    if size < 1:
        raise ValueError("Requested width is smaller than one reference pixel")
    shape = (size, size)
    header = reference_header.copy()
    header["CRPIX1"] = float(header["CRPIX1"]) + (shape[1] - reference_shape[1]) / 2.0
    header["CRPIX2"] = float(header["CRPIX2"]) + (shape[0] - reference_shape[0]) / 2.0
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]
    return header, shape


def write_planck_dust_cutout(
    output_path: str | Path,
    planck_fits: str | Path,
    target_header: fits.Header,
    shape: tuple[int, int],
    *,
    wavelengths_um: tuple[float, ...] = (100.0, 250.0),
    overwrite: bool = False,
) -> Path:
    """Sample and save Planck radiance, tau353, temperature, beta, and MBB maps."""

    output = Path(output_path)
    fields = sample_planck_dust_fields(
        planck_fits,
        WCS(target_header),
        shape,
        fields=("RADIANCE", "TAU353", "TEMP", "BETA"),
    )
    primary = fits.PrimaryHDU()
    primary.header["PLNKSRC"] = (Path(planck_fits).name, "Planck thermal-dust HEALPix source")
    primary.header["WIDTHDEG"] = (
        float(shape[1]) * float(np.mean(proj_plane_pixel_scales(WCS(target_header).celestial))),
        "Approximate cutout width [deg]",
    )
    arrays: list[tuple[str, np.ndarray, str]] = [
        ("RADIANCE", fields["RADIANCE"], "W m-2 sr-1"),
        ("TAU353", fields["TAU353"], "1"),
        ("DUST_TEMP", fields["TEMP"], "K"),
        ("DUST_BETA", fields["BETA"], "1"),
    ]
    for wavelength in wavelengths_um:
        name = f"I{int(round(wavelength))}"
        intensity = modified_blackbody_mjy_sr(
            fields["TAU353"], fields["TEMP"], fields["BETA"], wavelength
        )
        arrays.append((name, intensity, "MJy/sr"))
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [primary]
    for name, data, unit in arrays:
        hdu = fits.ImageHDU(np.asarray(data, dtype=np.float32), header=target_header, name=name)
        hdu.header["BUNIT"] = unit
        if name.startswith("I"):
            hdu.header["WAVELEN"] = (float(name[1:]), "Nominal wavelength [um]")
            hdu.header["MODEL"] = "Planck single-MBB monochromatic"
        hdus.append(hdu)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(output, overwrite=overwrite)
    return output
