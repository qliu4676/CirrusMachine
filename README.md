# CirrusMachine

Galactic cirrus synthesis and FIR-conditioned cirrus/halo modeling for
low-surface-brightness imaging. The modeling workflow uses J=5, L=8 wavelet
scattering statistics and calibrates the power-spectrum slope from the data.
The Cheng `scattering` backend is bundled; do not install it separately.

## Install

Python 3.10+; run from the repository checkout:

```bash
python -m pip install -e .              # Package and runtime dependencies
python -m pip install -e ".[tests]"    # Also install pytest (optional)
python -m pytest -q                    # Run the package tests
```

Test dependencies are optional. Tests live in the checkout and are not installed
in the wheel. Data-dependent reference tests skip when their inputs are absent.
Use `.[notebooks]` for Jupyter. Install a CUDA-enabled PyTorch build for GPU use;
modeling with `device="auto"` selects CUDA when available and CPU otherwise.
WST, optimization, covariance operations and photometry use PyTorch; WCS and
halo geometry use CPU routines. MPS is not supported.

## Cirrus and halo modeling

Supply an optical image, exclusion masks, a unit-sum PSF on the working grid,
and native FIR image/error/coverage/mask arrays with their WCS. Optical values
are ADU per delivered pixel; FIR values and 1-sigma errors are MJy/sr. Masks
passed below are `True` for excluded pixels. Inputs are not bundled.

```python
from cirrusmachine import modeling as cm

config = cm.load_config("configs/cutout_ddV.yaml")  # Or cm.ModelingConfig()
field = cm.prepare_field_data(
    optical, source_mask, target_mask, psf_working,
    center_xy=center_native, axis_ratio=0.6, pa_deg=30,
    native_pixel_scale=0.3, bin_factor=3, canvas_size=1024,
    exclusion_radius=150, wcs=optical_wcs,
)
statistics = cm.calibrate_statistics(field, config, progress=print)
beam = cm.prepare_fir_constraint(
    field, fir_image, fir_error, fir_coverage, fir_mask, fir_wcs, config,
    effective_fwhm=36.0,
)
prior = cm.build_cirrus_prior(field, beam, statistics, config)
initial = cm.initialize_cirrus(prior, config)
results = cm.run_cirrus_halo_modeling(
    field, prior, initial, config, progress=print,
)
corrected = results[0].corrected
cm.save_modeling_results(results, "results/cutout_ddV", config, field=field)
```

The supplied FIR map must already have the stated 36 arcsec effective resolution.
The config defaults to one realization using the existing representative seed
pair; add `draws` entries for more. All config keys may be omitted to use defaults.
`prepare_injection` supports synthetic-galaxy inputs; `measure_profile` and
`measure_cumulative_light` provide simple photometry. The example notebook is
planned for stage 2.

## Basic synthesis

```python
from cirrusmachine import MockSynthesisConfig, synthesize_single_image

# image: finite, nonconstant 2D NumPy array.
mock = synthesize_single_image(
    image, config=MockSynthesisConfig(device="cpu", seed=4703, steps=100),
)["image_out"]
```

For a small CPU installation check: `python -m cirrusmachine --output smoke.fits`.
This runs five optimization steps, not a converged science model.

Current code is in `src/cirrusmachine/modeling` and `synthesis.py`, with the
bundled backend in `src/scattering`. Historical variants and their local callers
use `cirrusmachine.legacy`; that subpackage is ignored by Git and excluded from
package builds. Local legacy tests are under `tests/legacy` and run only when
explicitly selected. Scientific inputs, generated outputs and notebooks remain
untracked.
