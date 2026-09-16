"""Fourier stitching and scale-separation utilities.

TODO:
- Add smooth low-pass and high-pass masks for combining a large-scale cirrus
  backbone with a high-resolution texture field.
- Implement Fourier stitching without manual Hermitian symmetry enforcement;
  real-input FFT routines already preserve the needed conjugate symmetry.
- Add power-spectrum diagnostics and tests for finite real-valued outputs.
- Document pixel-scale and Fourier-normalization conventions at every public
  entry point.
"""
