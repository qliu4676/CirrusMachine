"""Intensity conventions, progress reporting and repeatable notebook logging."""

import io
import logging

import numpy as np
import pytest
import torch
from scipy.special import gammaincinv

from cirrusmachine.modeling import sersic_image, surface_brightness_zeropoint
from cirrusmachine.modeling.logging import ModelingProgress, configure_logging


def test_sersic_central_and_effective_intensities():
    n, Re, Ie = 2, 3, 1.7
    I0 = Ie * np.exp(gammaincinv(2 * n, 0.5))
    args = dict(
        shape=(11, 11), center_xy=(5, 5), pixel_scale=1, axis_ratio=1, device="cpu"
    )
    effective = sersic_image(components=[dict(n=n, Re_arcsec=Re, Ie=Ie)], **args)
    central = sersic_image(components=[dict(n=n, Re_arcsec=Re, I0=I0)], **args)
    legacy = sersic_image(components=[dict(n=n, re_arcsec=Re, ie=Ie)], **args)
    torch.testing.assert_close(central, effective, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(legacy, effective)
    assert float(central[5, 5]) == pytest.approx(I0)
    assert float(effective[5, 8]) == pytest.approx(Ie)
    with pytest.raises(ValueError, match="one of"):
        sersic_image(components=[dict(n=n, Re_arcsec=Re, Ie=Ie, I0=I0)], **args)
    with pytest.raises(ValueError, match="nonnegative"):
        sersic_image(components=[dict(n=n, Re_arcsec=Re, Ie=np.nan)], **args)


def test_surface_brightness_binning_conventions():
    from cirrusmachine.modeling import measure_profile

    # The same uniform sky: mean-bin intensity stays fixed; 3x3 sums grow by 9.
    mean_ZP = surface_brightness_zeropoint(30.13, 0.1)
    sum_ZP = surface_brightness_zeropoint(30.13, 0.3)
    assert mean_ZP == pytest.approx(25.13)
    assert sum_ZP - 2.5 * np.log10(9) == pytest.approx(mean_ZP)
    image = np.ones((4, 4))
    profile = measure_profile(
        image * 9, image.astype(bool), image, [0, 2], SB_ZP=sum_ZP, device="cpu"
    )
    assert profile["surface_brightness"][0] == pytest.approx(mean_ZP)


@pytest.mark.parametrize(
    "steps, expected",
    [
        (150, list(range(15, 151, 15))),
        (3, [1, 2, 3]),
        (11, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ],
)
def test_progress_throttles_callbacks(steps, expected):
    reports = []
    with ModelingProgress(steps, 0, 1, reports.append) as status:
        for step in range(1, steps + 1):
            status.update(step, 1 / step, 0.01)
    assert [r["step"] for r in reports] == expected
    assert reports[-1]["loss"] == 1 / steps


def test_progress_closes_on_failure(monkeypatch):
    import cirrusmachine.modeling.logging as module

    class Bar:
        n = 0
        closed = False

        def set_postfix(self, **kwargs):
            pass

        def update(self, count):
            self.n += count

        def close(self):
            self.closed = True

    bar = Bar()
    monkeypatch.setattr(module, "tqdm", lambda **kwargs: bar)
    with pytest.raises(RuntimeError):
        with ModelingProgress(10, 0, 1, True) as status:
            status.update(1, 2.0, 0.01)
            raise RuntimeError("Numerical failure")
    assert bar.closed and bar.n == 1


def test_logging_notebook_rerun_and_levels():
    stream = io.StringIO()
    logger = logging.getLogger("cirrusmachine")
    previous = logger.handlers[:], logger.level, logger.propagate
    try:
        configure_logging("DEBUG", color=True, stream=stream)
        configure_logging("INFO", color=False, stream=stream)
        logging.getLogger("cirrusmachine.modeling.solver").info("one message")
        logging.getLogger("cirrusmachine.modeling.solver").debug("hidden")
        assert stream.getvalue() == "[INFO] one message\n"
    finally:
        logger.handlers, logger.level, logger.propagate = previous
