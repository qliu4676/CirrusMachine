"""Self-supervised coordinate neural fields for cirrus texture synthesis.

This subpackage is intentionally independent of the existing scattering
synthesis experiments.  It reuses their calibrated inputs and diagnostics
without changing their behavior.
"""

from .config import ExperimentConfig, load_config
from .model import MultiResolutionLatentField
from .operators import compose_with_backbone, project_residual

__all__ = [
    "ExperimentConfig",
    "MultiResolutionLatentField",
    "compose_with_backbone",
    "load_config",
    "project_residual",
]
