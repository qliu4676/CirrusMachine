"""Reusable helpers for the CirrusMachine research workflow.

CirrusMachine is currently notebook-first.  This package is a gradual landing
place for utility code extracted from the Colab notebooks without changing the
scientific behavior of the included Cheng et al. scattering package.
"""

from .synthesis import (
    MockSynthesisConfig,
    crop_reflect_padding,
    crop_to_even,
    dyadic_size,
    ensure_image_stack,
    reflect_pad,
    synthesize_batch,
    synthesize_from_estimator,
    synthesize_padded_mock,
    synthesize_single_image,
)

__all__ = [
    "MockSynthesisConfig",
    "crop_reflect_padding",
    "crop_to_even",
    "dyadic_size",
    "ensure_image_stack",
    "reflect_pad",
    "synthesize_batch",
    "synthesize_from_estimator",
    "synthesize_padded_mock",
    "synthesize_single_image",
]
