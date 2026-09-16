"""Wrappers around the bundled Cheng et al. scattering package.

TODO:
- Provide a lazy import path for ``scattering_transform/scattering``.
- Wrap ``Scattering2d.scattering_coef`` and ``scattering_cov`` with documented
  output shapes for ``for_synthesis`` and ``for_synthesis_iso``.
- Add masks for valid second-order scale pairs, especially ``j1 < j2`` where
  appropriate for isotropic losses.
- Add safe log-feature transforms with epsilons and finite-value checks.
"""
