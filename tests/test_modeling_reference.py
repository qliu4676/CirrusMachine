"""Optional read-only regressions using the external manuscript products.

Set CIRRUS_MODELING_CASE to the saved cutout1_ddv case directory and
CIRRUS_MODELING_PSF to the delivered-VIS PSF archive. No data are bundled.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cirrusmachine.modeling import (
    CirrusPrior,
    DrawSeeds,
    FieldData,
    InitialRealization,
    ModelingConfig,
    StatisticalCalibration,
    run_cirrus_halo_modeling,
)
from cirrusmachine.modeling.calibration import calibration_signature
from cirrusmachine.modeling.halo import refit_inner_halo
from cirrusmachine.modeling.io import load_arrays

pytestmark = pytest.mark.skipif(
    not os.environ.get("CIRRUS_MODELING_CASE")
    or not os.environ.get("CIRRUS_MODELING_PSF"),
    reason="External manuscript data paths not configured",
)


@pytest.fixture
def reference():
    case = Path(os.environ["CIRRUS_MODELING_CASE"])
    config = ModelingConfig(device="cpu", beam_fwhm=18.0, steps=1, inner_refit=False)
    data = load_arrays(case / "cache/input.npz")
    science = load_arrays(case / "cache/science_input.npz")
    metadata = json.loads((case / "config/experiment.json").read_text())
    psf = load_arrays(os.environ["CIRRUS_MODELING_PSF"])["psf_coarse"]
    mu, std = float(data["mu"]), float(data["std"])
    field = FieldData(
        mu + std * science["observed"].astype(float),
        data["donor"],
        science["source_clean"],
        (slice(179, 845), slice(179, 845)),
        tuple(metadata["center_crop_xy"]),
        science["radius"],
        psf,
        normalization=(mu, std),
    )
    calibration = StatisticalCalibration(
        load_arrays(case / "cache/noise.npz"),
        load_arrays(case / "cache/refined_model.npz"),
        load_arrays(case / "cache/targets.npz"),
        calibration_signature(field, config),
        {},
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield case, config, field, calibration, data, science, metadata
    torch.set_num_threads(threads)


def test_first_manuscript_step(reference):
    case, config, field, calibration, data, science, metadata = reference
    # These are explicitly imported historical products, not freshly calibrated
    # 36 arcsec inputs. Use the saved allowance and phase to isolate the solver.
    prior = CirrusPrior(
        None,
        None,
        None,
        float(load_arrays(case / "cache/initial_prior.npz")["amp"]),
        load_arrays(case / "cache/backbone.npz"),
        calibration,
    )
    initial = InitialRealization(
        torch.tensor(np.load(case / "products/initial_draw3.npy")), DrawSeeds()
    )
    result = run_cirrus_halo_modeling(
        field,
        prior,
        [initial],
        config,
        extra_variance=load_arrays(case / "cache/likelihood_extra.npz")["variance"],
    )[0]
    expected = json.loads((case / "runs/draw3/history.json").read_text())["history"][0]
    # CPU roundoff allowance; locally all nine entries were bitwise identical.
    np.testing.assert_allclose(result.history[0], expected, rtol=2e-6, atol=2e-7)
    assert (
        max(v["relative_residual"] for v in result.diagnostics["covariance_solves"])
        < 2e-6
    )


def test_saved_inner_halo(reference):
    case, config, field, calibration, data, science, metadata = reference
    output = case / "runs/draw3"
    saved = load_arrays(output / "checkpoint_150.npz")
    info = json.loads((output / "halo_150.json").read_text())
    obs = float(data["mu"]) + float(data["std"]) * science["observed"][field.crop]
    result, report = refit_inner_halo(
        obs - saved["image"].astype(float) - float(saved["sky_ADU"]),
        saved["halo"].astype(float),
        field.source_clean[field.crop],
        field.center_xy,
        info,
        calibration.noise["covariance"] * float(data["std"]) ** 2,
        config,
    )
    np.testing.assert_allclose(
        result, np.load(output / "inner_refit/halo.npy"), rtol=1e-10, atol=1e-8
    )
    assert report["solver"]["kkt"] < 1e-8
