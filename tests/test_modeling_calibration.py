"""Small, uncached calibration and conditional-initialization integration test."""

import numpy as np
import torch

from cirrusmachine.modeling import (
    FieldData,
    ModelingConfig,
    build_cirrus_prior,
    calibrate_statistics,
    initialize_cirrus,
)
from cirrusmachine.modeling.calibration import calibration_signature
from cirrusmachine.modeling.operators import power_field


def test_uncached_calibration_and_conditional_initialization():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        n = 128
        y, x = np.indices((n, n))
        radius = np.hypot(x - 64, y - 64)
        texture = power_field((n, n), 2.5, 324, np.ones((1, 1)), coherence=0).numpy()
        image = (
            2
            + texture
            + 0.2 * np.random.default_rng(4).normal(size=texture.shape)
            + x * 0.005
        )
        field = FieldData(
            image,
            radius > 10,
            np.ones((n, n), bool),
            (slice(0, n), slice(0, n)),
            (64, 64),
            radius,
            np.ones((1, 1)),
            0.9,
            10.0,
        )
        config = ModelingConfig(
            device="cpu", core_radius=3, sky_annulus=(35, 50), inner_refit=False
        )
        cal = calibrate_statistics(field, config)
        assert cal.noise["beta"] in config.beta_grid
        assert max(cal.diagnostics["targets"]["heldout_mask_transfer_RMSE"][:3]) < 0.2
        assert max(cal.diagnostics["targets"]["hidden_value_invariance"]) == 0
        beam = dict(
            sigma=config.beam_FWHM / config.pixel_scale / 2.354820045,
            backbone=np.zeros((n, n)),
            optical_signature=calibration_signature(field, config),
        )
        prior = build_cirrus_prior(field, beam, cal, config)
        first = initialize_cirrus(prior, config)
        second = initialize_cirrus(prior, config)
        assert len(first) == 1
        assert torch.isfinite(first[0].image).all()
        torch.testing.assert_close(first[0].image, second[0].image, rtol=0, atol=0)
        assert max(v["relative_residual"] for v in prior.broad.solves) < 3e-5
    finally:
        torch.set_num_threads(threads)
