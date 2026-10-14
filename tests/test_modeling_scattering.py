"""Exercise the upstream dependency through the maintained synthesis interface."""

import numpy as np
import pytest
import torch

from cirrusmachine import MockSynthesisConfig, synthesize_single_image
from cirrusmachine._scattering_synthesis import synthesis


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("estimator", ["s_mean_iso", "s_cov"])
def test_upstream_estimators_through_synthesis_adapter(estimator):
    image = np.random.default_rng(7).normal(size=(16, 16))
    config = MockSynthesisConfig(
        estimator_name=estimator,
        J=2,
        seed=12,
        device="cpu",
        optim_algorithm="Adam",
        steps=2,
        learning_rate=0.005,
    )
    first = synthesize_single_image(image, config=config)["image_out"]
    second = synthesize_single_image(image, config=config)["image_out"]
    assert first.shape[-2:] == image.shape and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_optional_seed_and_multiple_realizations():
    image = np.random.default_rng(8).normal(size=(1, 16, 16))
    result = synthesis(
        "s_mean_iso",
        image,
        J=2,
        L=4,
        seed=None,
        N_synthesis=2,
        device="cpu",
        optim_algorithm="Adam",
        steps=2,
        learning_rate=0.005,
    )
    assert result.shape == (2, 1, 16, 16)
    assert np.isfinite(result).all()
    assert not np.array_equal(result[0], result[1])
