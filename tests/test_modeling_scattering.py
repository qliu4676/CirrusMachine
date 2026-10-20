"""Exercise the bundled estimators through the maintained synthesis interface."""

import os
import subprocess
import sys

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


def test_backend_does_not_import_a_separate_scattering_installation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['scattering'] = None
from cirrusmachine.modeling.statistics import Scattering2d
from cirrusmachine._scattering_synthesis import synthesis
import numpy as np
import torch
torch.set_num_threads(2)
assert Scattering2d.__module__.startswith('cirrusmachine._vendor.')
image = np.random.default_rng(5).normal(size=(1, 16, 16))
result = synthesis('s_mean_iso', image, J=2, steps=1, device='cpu',
                   optim_algorithm='Adam', seed=3)
assert result.shape == image.shape and np.isfinite(result).all()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cpu_defaults_ignore_external_gpu_backend_settings():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.modules['skcuda'] = None
sys.modules['cupy'] = None
import torch
torch.set_num_threads(2)
torch.cuda.is_available = lambda: False
def no_cuda(*args, **kwargs):
    raise AssertionError('CPU fallback attempted a CUDA allocation')
torch.Tensor.cuda = no_cuda
import numpy as np
from cirrusmachine import MockSynthesisConfig, synthesize_single_image
from cirrusmachine.modeling import ModelingConfig
from cirrusmachine.modeling.statistics import Statistics
from cirrusmachine._vendor.scattering import backend
assert backend.BACKEND == 'torch'
mask = np.ones((64, 64), dtype=bool)
classes = np.tile(np.arange(64) % 3, (64, 1))
measure = Statistics(mask, mask, classes, ModelingConfig())
image = torch.randn(64, 64, requires_grad=True)
values = measure.all(image)
assert all(value.device.type == 'cpu' for value in values)
sum(value.square().mean() for value in values).backward()
assert torch.isfinite(image.grad).all()
result = synthesize_single_image(
    np.random.default_rng(4).normal(size=(16, 16)),
    config=MockSynthesisConfig(J=2, steps=1, seed=3, optim_algorithm='Adam'),
)['image_out']
assert np.isfinite(result).all()
"""],
        env={**os.environ, "KYMATIO_BACKEND": "skcuda"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
