"""Custom diffuse models participate in the complete alternating solver."""

import numpy as np
import pytest
import torch

from cirrusmachine import modeling as cm
from cirrusmachine.modeling import diffuse, solver
from cirrusmachine.modeling.calibration import calibration_signature
from cirrusmachine.modeling.operators import covariance_kernel, noise_bases
from cirrusmachine.modeling.statistics import Statistics


@pytest.fixture(scope="module")
def model_case():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        n = 64
        y, x = np.indices((n, n))
        radius = 0.9 * np.hypot(x - 32, y - 32)
        valid = np.ones((n, n), bool)
        valid[15:18, 20:23] = False
        field = cm.FieldData(
            5 + np.random.default_rng(21).normal(size=(n, n)),
            valid & (radius > 8), valid,
            (slice(8, 56), slice(10, 58)), (22, 24), radius, np.ones((1, 1)),
            exclusion_radius=8, normalization=(5, 2),
        )
        config = cm.ModelingConfig(
            device="cpu", steps=2, halo_refresh_steps=1, core_radius=2,
            sky_annulus=(15, 20), noise_seeds=(3, 4), inner_refit=True,
        )
        initial = cm.InitialRealization(torch.zeros((n, n)), cm.DrawSeeds())
        classes = x % 3
        measure = Statistics(field.donor, field.roi, classes, config)
        values = measure.all(torch.tensor(field.normalized(config)["science"]))
        targets = {"classes": classes}
        for j, value in enumerate(values):
            a = value.numpy()
            targets.update({
                f"observed{j}": a, f"M1_{j}": a,
                f"uncertainty{j}": np.full_like(a, 0.1),
                f"M1_std{j}": np.full_like(a, 0.1),
                f"factor{j}": np.ones_like(a),
                f"mask_offset{j}": np.zeros_like(a),
                f"mask_sigma{j}": np.full_like(a, 0.01),
            })
        kernels, variances = noise_bases(), np.array([0.1, 0, 0, 0, 0])
        calibration = cm.StatisticalCalibration(
            dict(beta=2.5, kernels=kernels, variances=variances,
                 covariance=covariance_kernel(kernels, variances)),
            dict(beta=2.5), targets, calibration_signature(field, config), {},
        )
        beam = dict(
            sigma=config.beam_FWHM / config.pixel_scale / 2.354820045,
            sample_grid=np.array([[[[0, 0], [0.25, 0.25]]]]),
            target=np.zeros(2), chol=np.eye(2), backbone=np.zeros((n, n)),
        )
        prior = cm.CirrusPrior(None, None, None, 0.2, beam, calibration)
        yield field, prior, initial, config
    finally:
        torch.set_num_threads(threads)


def test_custom_model_used_for_allowance_all_updates_and_each_draw(model_case, monkeypatch):
    field, prior, initial, config = model_case
    calls = []
    seeded = []

    def initialize(prior, config, *, seeds):
        seeded.append(seeds)
        return [initial]

    def forbidden(*args, **kwargs):
        pytest.fail("Custom model must not call isophote fitting or inner refitting")

    def custom(residual, valid, center, measurement_radius, annulus, previous=None,
               *, pixel_scale):
        assert residual.shape == field.image[field.crop].shape
        np.testing.assert_array_equal(valid, field.source_clean[field.crop])
        np.testing.assert_array_equal(measurement_radius, field.radius[field.crop])
        assert center == field.center_xy and annulus == config.sky_annulus
        assert pixel_scale == config.pixel_scale
        if previous is None:
            # The initial cirrus is the normalization mean (5), not zero ADU.
            np.testing.assert_allclose(residual[valid], field.image[field.crop][valid] - 5,
                                       atol=5e-7)
        iteration = 1 if previous is None else previous["iteration"] + 1
        calls.append(iteration)
        # No isophote-specific geometry metadata; tensors are detached by the solver.
        return torch.full(residual.shape, 0.25 * iteration, requires_grad=True), {
            "sky": {"value_ADU": torch.tensor(0.1)}, "iteration": iteration,
        }

    monkeypatch.setattr(solver, "initialize_cirrus", initialize)
    monkeypatch.setattr(solver, "fit_diffuse", forbidden)
    monkeypatch.setattr(diffuse, "refit_inner_diffuse", forbidden)
    results = cm.run_cirrus_halo_modeling(
        field, prior, [initial, initial], config, diffuse_model=custom, progress=False,
    )
    assert seeded == [(config.allowance_seeds,)]
    assert calls == [1, 1, 2, 3, 4, 1, 2, 3, 4]
    for result in results:
        np.testing.assert_allclose(result.halo, 1.0)
        assert result.sky == pytest.approx(0.1)
        assert result.diagnostics["inner_refit"] is None
        assert result.diagnostics["halo"]["iteration"] == 4
        assert np.isfinite(result.history).all()
        np.testing.assert_allclose(result.corrected, field.image[field.crop] - result.cirrus)


@pytest.mark.parametrize("explicit_default", [False, True])
def test_default_model_retains_inner_refit(model_case, monkeypatch, explicit_default):
    field, prior, initial, config = model_case
    refits = []

    def default(residual, *args, **kwargs):
        return np.full_like(residual, 0.5), {"sky": {"value_ADU": 0.1}}

    def refit(residual, original, *args):
        refits.append(original.copy())
        return original + 0.25, {"refined": True}

    monkeypatch.setattr(solver, "fit_diffuse", default)
    monkeypatch.setattr(diffuse, "refit_inner_diffuse", refit)
    result = cm.run_cirrus_halo_modeling(
        field, prior, [initial], config, progress=False,
        extra_variance=np.zeros_like(field.image),
        **({"diffuse_model": default} if explicit_default else {}),
    )[0]
    assert len(refits) == 1
    np.testing.assert_allclose(result.halo, 0.75)
    assert result.diagnostics["inner_refit"] == {"refined": True}


@pytest.mark.parametrize("bad", ["shape", "nan", "negative", "complex", "sky_missing", "sky_nan", "sky_array"])
def test_custom_model_rejects_invalid_outputs(model_case, monkeypatch, bad):
    field, prior, initial, config = model_case
    monkeypatch.setattr(solver, "initialize_cirrus", lambda *a, **kw: [initial])

    def custom(residual, *args, **kwargs):
        image = np.ones_like(residual)
        info = {"sky": {"value_ADU": 0.1}}
        if bad == "shape":
            image = image[:-1]
        elif bad == "nan":
            image[0, 0] = np.nan
        elif bad == "negative":
            image[0, 0] = -1
        elif bad == "complex":
            image = image + 1j
        elif bad == "sky_missing":
            info = {}
        elif bad == "sky_nan":
            info["sky"]["value_ADU"] = np.nan
        elif bad == "sky_array":
            info["sky"]["value_ADU"] = [0.1, 0.2]
        return image, info

    with pytest.raises(ValueError, match="Diffuse model"):
        cm.prepare_likelihood_variance(field, prior, config, diffuse_model=custom)


def test_noncallable_model_rejected_before_calibration():
    with pytest.raises(TypeError, match="diffuse_model must be callable"):
        cm.run_cirrus_halo_modeling(None, None, [], cm.ModelingConfig(), diffuse_model=3)
