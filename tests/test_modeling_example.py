"""Portable Cutout inputs retain native values, masks and celestial alignment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

spec = importlib.util.spec_from_file_location(
    "cutout_example", Path(__file__).parents[1] / "notebooks/cutout_ddV_helpers.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def original_inputs(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [4, 4]
    wcs.wcs.crval = [328.7, 17.6]
    wcs.wcs.cdelt = [-0.3 / 3600, 0.3 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    image = np.arange(64, dtype="f4").reshape(8, 8)
    bits = np.zeros((8, 8), dtype="i2")
    bits[1, 1], bits[2, 2], bits[3, 3] = 1, 256, 257
    holes = np.zeros((8, 8), dtype="u1")
    holes[0, 0] = 1
    paths = {
        key: tmp_path / f"{key}.fits"
        for key in ("optical", "mask_bits", "FIR", "FIR_masks")
    }
    fits.writeto(paths["optical"], image, header)
    fits.writeto(paths["mask_bits"], bits, header)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, header, name="image"),
            fits.ImageHDU(np.ones_like(image), header, name="error"),
            fits.ImageHDU(np.ones_like(image), header, name="coverage"),
        ]
    ).writeto(paths["FIR"])
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(holes, header, name="MASK_RAD")]
    ).writeto(paths["FIR_masks"])
    paths["PSFs"] = tmp_path / "psfs.npz"
    np.savez(paths["PSFs"], psf_native=np.ones((1, 1)), psf_coarse=np.ones((1, 1)))
    return paths, image, bits


def test_portable_inputs_preserve_values_and_masks(tmp_path, original_inputs):
    paths, image, bits = original_inputs
    destination = tmp_path / "portable"
    example.export_inputs(destination, **paths)
    loaded = example.load_inputs(destination)
    np.testing.assert_array_equal(loaded["optical"], image)
    np.testing.assert_array_equal(loaded["FIR_image"], image)
    np.testing.assert_array_equal(loaded["source_mask"], (bits & ~256) != 0)
    np.testing.assert_array_equal(loaded["target_mask"], (bits & 256) != 0)
    assert loaded["FIR_mask"][0, 0]
    assert loaded["metadata"]["effective_FWHM"] == 36
    with pytest.raises(FileExistsError):
        example.export_inputs(destination, **paths)
    with fits.open(destination / "optical.fits", mode="update") as hdus:
        hdus[0].data[0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        example.load_inputs(destination)


def test_portable_inputs_reject_misaligned_mask(tmp_path, original_inputs):
    paths, _, _ = original_inputs
    with fits.open(paths["FIR_masks"], mode="update") as hdus:
        hdus["MASK_RAD"].header["CRPIX1"] += 1
    with pytest.raises(ValueError, match="celestial grids"):
        example.export_inputs(tmp_path / "portable", **paths)
    assert not (tmp_path / "portable").exists()


def test_injection_preserves_detector_grid_center():
    from scipy.special import gammaincinv

    from cirrusmachine.modeling import FieldData, prepare_injection

    n, sampling, margin = 32, 3, 2
    center = 15 + 1 / 3 + 1e-14
    y, x = np.indices((n, n))
    field = FieldData(
        x.astype(float),
        np.ones((n, n), bool),
        np.ones((n, n), bool),
        (slice(0, n), slice(0, n)),
        (center, center),
        np.hypot(x - center, y - center),
        np.ones((1, 1)),
        pixel_scale=0.9,
    )
    result = prepare_injection(
        field,
        [{"n": 4, "Re_arcsec": 8, "Ie": 1}],
        PSF_native=np.ones((1, 1)),
        oversample=sampling,
        margin_pixels=margin,
        axis_ratio=1,
        PA_deg=0,
        device="cpu",
    )
    side = (n + 2 * margin) * sampling
    yy, xx = np.indices((side, side))
    radius = np.hypot(xx - 53, yy - 53) * 0.3
    expected = np.exp(-gammaincinv(8, 0.5) * ((radius / 8) ** 0.25 - 1)).astype("f4")
    expected = expected.reshape(
        n + 2 * margin, sampling, n + 2 * margin, sampling
    ).mean((1, 3))
    expected = expected[margin:-margin, margin:-margin]
    # Single-precision intrinsic samples/FFT; the sharp central bin must not
    # inherit a spurious subpixel displacement from working-grid arithmetic.
    np.testing.assert_allclose(result.truth, expected, rtol=2e-6, atol=2e-5)
