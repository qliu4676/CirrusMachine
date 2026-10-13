"""Numerical contracts for the J=5 modeling workflow."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter as scipy_gaussian
from scipy.signal import convolve2d

from cirrusmachine.modeling import (
    DrawSeeds,
    FieldData,
    ModelingConfig,
    load_config,
    measure_cumulative_light,
    measure_profile,
    prepare_injection,
    save_config,
)
from cirrusmachine.modeling.operators import (
    MaskedCovariance,
    covariance_kernel,
    gaussian_filter,
    noise_bases,
)
from cirrusmachine.modeling.statistics import Statistics, independent_targets


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_config_roundtrip_and_validation(tmp_path):
    config = ModelingConfig()
    save_config(config, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == config
    assert config.draws == (DrawSeeds(2711003, 2721003),)
    for kwargs in (
        {"scales": 6},
        {"steps": 0},
        {"beta_grid": ()},
        {"weights": (1.0,)},
        {"pixel_scale": float("nan")},
    ):
        with pytest.raises(ValueError):
            ModelingConfig(**kwargs)
    (tmp_path / "bad.yaml").write_text("unknown: 1")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(tmp_path / "bad.yaml")


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_gaussian_boundaries_match_scipy(mode):
    x = np.random.default_rng(2).normal(size=(17, 21))
    for sigma in (0.6, 3.0, 20.0):
        np.testing.assert_allclose(
            gaussian_filter(torch.tensor(x), sigma, mode).numpy(),
            scipy_gaussian(x, sigma, mode=mode),
            atol=2e-15,
        )


def test_restricted_covariance_dense_solution_and_gradient():
    mask = np.ones((9, 10), bool)
    mask[2:4, 3] = False
    cov = covariance_kernel(noise_bases(), [0.7, 0.1, 0.05, 0.1, 0.05])
    extra = np.linspace(0, 0.3, mask.size).reshape(mask.shape)
    op = MaskedCovariance(mask, cov, extra, rtol=1e-10)
    ids = np.argwhere(mask)
    dense = np.zeros((len(ids), len(ids)))
    for k, (y, x) in enumerate(ids):
        impulse = np.zeros(mask.shape)
        impulse[y, x] = 1
        dense[:, k] = convolve2d(impulse, cov, mode="same")[mask]
        dense[k, k] += extra[y, x]
    x = torch.tensor(
        np.random.default_rng(13).normal(size=mask.shape), requires_grad=True
    )
    truth = np.linalg.solve(dense, x.detach().numpy()[mask])
    np.testing.assert_allclose(
        op.solve(x.detach()[mask]).numpy(), truth, rtol=2e-9, atol=1e-10
    )
    op.quadratic(x).backward()
    np.testing.assert_allclose(
        x.grad.numpy()[mask], 2 * truth / mask.sum(), rtol=2e-9, atol=1e-10
    )
    assert np.all(x.grad.numpy()[~mask] == 0)
    op.solve(torch.zeros(int(mask.sum())))
    assert op.last["iterations"] == 0


def test_targets_ignore_hidden_values_and_full_statistics_differentiate():
    n = 64
    mask = np.ones((n, n), bool)
    mask[25:34, 24:35] = False
    classes = np.tile(np.arange(n) % 3, (n, 1))
    m = Statistics(mask, np.ones_like(mask), classes, ModelingConfig(device="cpu"))
    x = torch.tensor(
        np.random.default_rng(5).normal(size=mask.shape),
        dtype=torch.float32,
        requires_grad=True,
    )
    values = m.all(x)
    assert [tuple(a.shape) for a in values[:3]] == [(5, 8), (10, 8, 8), (5, 8)]
    sum(a.square().mean() for a in values).backward()
    assert torch.isfinite(x.grad).all()
    ones = [torch.ones_like(a) for a in values]
    zeros = [torch.zeros_like(a) for a in values]
    args = (noise_bases(), np.array([0.1, 0, 0, 0, 0]), [3, 4], ones, zeros, zeros)
    first = independent_targets(m, x.detach(), x.detach(), *args)
    poisoned = x.detach().clone()
    poisoned[~torch.tensor(mask)] = float("nan")
    second = independent_targets(m, poisoned, poisoned, *args)
    for a, b in zip(first[0] + first[1], second[0] + second[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_photometry_uniform_signal_and_missing_bins():
    y, x = np.indices((64, 64))
    radius = np.hypot(x - 32, y - 32)
    image = np.full(radius.shape, 7.0)
    valid = np.ones(radius.shape, bool)
    valid[(radius >= 5) & (radius < 10)] = False
    profile = measure_profile(image, valid, radius, [0, 5, 10, 20], sky=2, device="cpu")
    np.testing.assert_allclose(profile["intensity"][[0, 2]], 5)
    assert np.isnan(profile["intensity"][1])
    light = measure_cumulative_light(
        image, valid, radius, [5, 10, 20], sky=2, pixel_scale=1.0, device="cpu"
    )
    np.testing.assert_allclose(
        light["flux"], [5 * np.sum(valid & (radius < hi)) for hi in [5, 10, 20]]
    )
    assert light["flux"][0] == light["flux"][1]


def test_injection_truth_separate_and_normalization_recomputed():
    n = 64
    y, x = np.indices((n, n))
    r = np.hypot(x - 32, y - 32)
    image = np.random.default_rng(12).normal(size=(n, n))
    donor = r > 12
    field = FieldData(
        image,
        donor,
        np.ones_like(donor),
        (slice(0, n), slice(0, n)),
        (32, 32),
        r,
        np.ones((1, 1)),
        1.0,
        12.0,
    )
    result = prepare_injection(
        field,
        [dict(n=1.0, re_arcsec=4.0, ie=1.0)],
        psf_native=np.ones((1, 1)),
        oversample=1,
        margin_pixels=4,
        device="cpu",
    )
    np.testing.assert_allclose(result.field.image, field.image + result.truth)
    np.testing.assert_array_equal(field.image, image)
    assert result.field.normalization != field.normalization
    config = ModelingConfig(pixel_scale=1.0, core_radius=3.0)
    assert np.all(result.field.normalized(config)["observed"][~donor] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_cpu_statistics_and_covariance():
    rng = np.random.default_rng(4)
    n = 64
    mask = np.ones((n, n), bool)
    mask[25:30, 25:30] = False
    classes = np.tile(np.arange(n) % 3, (n, 1))
    x = torch.tensor(rng.normal(size=(n, n)), dtype=torch.float32)
    cpu = Statistics(mask, np.ones_like(mask), classes, ModelingConfig(device="cpu"))
    gpu = Statistics(mask, np.ones_like(mask), classes, ModelingConfig(device="cuda"))
    for masked in (False, True):
        for a, b in zip(cpu.all(x, masked), gpu.all(x.cuda(), masked)):
            torch.testing.assert_close(a, b.cpu(), rtol=2e-5, atol=1e-6)
    cov = covariance_kernel(noise_bases(), [0.7, 0.1, 0.05, 0.1, 0.05])
    a = MaskedCovariance(mask, cov).solve(x[mask])
    b = MaskedCovariance(mask, cov, device="cuda").solve(x.cuda()[mask]).cpu()
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_broad_conditioning_matches_dense_gaussian_mean():
    from cirrusmachine.modeling.priors import BroadModel

    n = 32
    rng = np.random.default_rng(54)
    mask = np.zeros((n, n), bool)
    mask[8:23:3, 9:24:3] = True
    observed = rng.normal(size=(n, n))
    guide = np.full((n, n), 0.4)
    config = ModelingConfig(device="cpu")
    model = BroadModel(observed, mask, guide, 0.3, 0.1, np.array([[0.2]]), config)
    ids = np.argwhere(mask)
    broad = model.apply(
        model.broad,
        torch.eye(n, dtype=torch.float64)[0, :, None]
        * torch.eye(n, dtype=torch.float64)[0, None, :],
    ).numpy()
    total = model.apply(
        model.total,
        torch.eye(n, dtype=torch.float64)[0, :, None]
        * torch.eye(n, dtype=torch.float64)[0, None, :],
    ).numpy()
    dy = (ids[:, 0, None] - ids[None, :, 0]) % n
    dx = (ids[:, 1, None] - ids[None, :, 1]) % n
    weights = np.linalg.solve(total[dy, dx], (observed - guide)[mask])
    yy, xx = np.indices((n, n))
    expected = guide + np.sum(
        broad[(yy[..., None] - ids[:, 0]) % n, (xx[..., None] - ids[:, 1]) % n]
        * weights,
        axis=-1,
    )
    np.testing.assert_allclose(model.mean().numpy(), expected, rtol=1e-5, atol=5e-6)
    torch.testing.assert_close(
        model.draw(model.mean(), 17), model.draw(model.mean(), 17), rtol=0, atol=0
    )


def test_calibration_cache_binds_masks_psf_and_settings(tmp_path):
    from cirrusmachine.modeling.calibration import (
        StatisticalCalibration,
        calibration_signature,
    )
    from cirrusmachine.modeling.io import load_calibration, save_calibration

    n = 64
    y, x = np.indices((n, n))
    r = np.hypot(x - 32, y - 32)
    field = FieldData(
        np.random.default_rng(5).normal(size=(n, n)),
        r > 15,
        np.ones((n, n), bool),
        (slice(0, n), slice(0, n)),
        (32, 32),
        r,
        np.ones((1, 1)),
        1.0,
        15.0,
    )
    config = ModelingConfig(pixel_scale=1.0, core_radius=5.0)
    calibration = StatisticalCalibration(
        {"beta": 2.5}, {"beta": 2.5}, {}, calibration_signature(field, config), {}
    )
    save_calibration(calibration, tmp_path / "cal.npz")
    assert (
        float(load_calibration(tmp_path / "cal.npz", field, config).noise["beta"])
        == 2.5
    )
    field.donor[0, 0] = False
    with pytest.raises(ValueError, match="Stale"):
        load_calibration(tmp_path / "cal.npz", field, config)


def test_native_binning_masks_and_wcs():
    from astropy.wcs import WCS

    from cirrusmachine.modeling import prepare_field_data

    n = 200
    x = np.arange(n * n).reshape(n, n).astype(float)
    sources = np.zeros((n, n), bool)
    sources[3, 3] = True
    target = np.zeros_like(sources)
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [328, 17]
    w.wcs.cdelt = [-0.3 / 3600, 0.3 / 3600]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    f = prepare_field_data(
        x,
        sources,
        target,
        np.ones((1, 1)),
        center_xy=(100, 100),
        axis_ratio=0.6,
        pa_deg=30,
        canvas_size=128,
        wcs=w,
    )
    np.testing.assert_array_equal(
        f.image[f.crop], x[1:199, 1:199].reshape(66, 3, 66, 3).mean((1, 3))
    )
    assert not f.source_clean[f.crop][0, 0]
    points = np.array([[0, 0], [30, 25], [65, 65]])
    original = w.all_pix2world(points * 3 + 2, 0)
    transformed = f.wcs.all_pix2world(points + 31, 0)
    np.testing.assert_allclose(original, transformed, rtol=0, atol=1e-10)


def test_fits_byte_order_and_reversed_inputs():
    from cirrusmachine.modeling.calibration import NumpyOperators
    from cirrusmachine.modeling.operators import to_tensor

    a = np.arange(64 * 64, dtype=">f4").reshape(64, 64)
    np.testing.assert_array_equal(to_tensor(a[::-1]).numpy(), a[::-1])
    result, _ = NumpyOperators(np.ones((1, 1)), ModelingConfig()).smooth_observed(
        a, np.ones(a.shape, bool), 2
    )
    assert np.isfinite(result).all()


def test_sky_matches_observed_halo_estimator():
    from cirrusmachine.modeling import estimate_sky
    from cirrusmachine.modeling.halo import sky_estimate

    y, x = np.indices((64, 64))
    radius = np.hypot(x - 32, y - 32)
    image = np.random.default_rng(431).normal(size=radius.shape) + 3
    image[5:7, 10:13] = 30
    valid = np.ones_like(image, bool)
    a = estimate_sky(image, valid, radius, (20, 29), device="cpu")
    b = sky_estimate(image, valid, radius, (20, 29))
    np.testing.assert_allclose(
        [a["value_ADU"], a["scatter_ADU"]],
        [b["value_ADU"], b["scatter_ADU"]],
        rtol=1e-14,
    )
    assert a["retained_pixels"] == b["retained_pixels"]
