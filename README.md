# CirrusMachine

Galactic cirrus synthesis and FIR-conditioned cirrus/halo modeling for
low-surface-brightness imaging, using wavelet scattering statistics to constrain
cirrus structure while recovering diffuse galaxy light.
The Cheng `scattering` backend is bundled; do not install it separately.

## Install

Python 3.10+; run from the repository checkout:

```bash
python -m pip install -e .              # Package and runtime dependencies
python -m pip install -e ".[tests]"    # Also install pytest (optional)
python -m pip install -e ".[examples]" # Also install Jupyter/widgets (optional)
python -m pytest -q                    # Run the package tests
```

Test dependencies are optional. Tests live in the checkout and are not installed
in the wheel. Data-dependent reference tests skip when their inputs are absent.
Combine extras with `.[tests,examples]`. Install a CUDA-enabled PyTorch build for GPU use;
modeling with `device="auto"` selects CUDA when available and CPU otherwise.
WST, optimization, covariance operations and photometry use PyTorch; WCS and
halo geometry use CPU routines. MPS is not supported.

## Cutout example

Open [the Cutout notebook](notebooks/cutout_ddV_modeling.ipynb) for galaxy
injection, cirrus modeling, and surface-brightness/cumulative-flux comparisons.
It uses the 36″ likelihood and one existing representative seed pair.
The notebook discovers the checkout and creates a fresh output path per run;
Colab uses the same notebook with a GPU runtime or CPU fallback.

Keep the external input folder at `product/examples/cutout_ddV`, or set
`CIRRUSMACHINE_DATA`. It contains `optical.fits`, `masks.fits`, `fir.fits`,
`psfs.npz`, and `inputs.json` (about 30 MB). Copy this folder separately to Colab.
Pip extras install dependencies; this optional data bundle stays outside Git
and package distributions. Provenance includes filenames without personal paths.
The adjacent [helper script](notebooks/cutout_ddV_helpers.py) provides
`export_inputs` to prepare these files from the original FITS maps and PSF
archive, plus the notebook's small plotting functions.

## Cirrus and halo modeling

Supply an optical image, exclusion masks, a unit-sum PSF on the working grid,
and native FIR image/error/coverage/mask arrays with their WCS. Optical values
are ADU per delivered pixel; FIR values and 1-sigma errors are MJy/sr. Masks
passed below are `True` for excluded pixels. Inputs are not bundled.

```python
from cirrusmachine import modeling as cm
from uuid import uuid4

cm.configure_logging("INFO")  # Optional: DEBUG; color="auto" by default.
config = cm.load_config("configs/cutout_ddV.yaml")  # Or cm.ModelingConfig()
field = cm.prepare_field_data(
    optical, source_mask, target_mask, PSF_working,
    center_xy=center_native, axis_ratio=0.6, PA_deg=30,
    native_pixel_scale=0.3, bin_factor=3, canvas_size=1024,
    exclusion_radius=150, WCS=optical_WCS,
)
statistics = cm.calibrate_statistics(field, config)
beam = cm.prepare_FIR_constraint(
    field, FIR_image, FIR_error, FIR_coverage, FIR_mask, FIR_WCS, config,
    effective_FWHM=36.0,
)
prior = cm.build_cirrus_prior(field, beam, statistics, config)
initial = cm.initialize_cirrus(prior, config)
results = cm.run_cirrus_halo_modeling(field, prior, initial, config)
corrected = results[0].corrected
cm.save_modeling_results(results, f"results/cutout_ddV/{uuid4().hex}", config, field=field)
```

The supplied FIR map must already have the stated 36 arcsec effective resolution.
The config defaults to one realization using the existing representative seed
pair; add `draws` entries for more. All config keys may be omitted to use defaults.
The progress bar updates every 10%; pass `progress=False` to hide it.
Injection components use `n`, `Re_arcsec` and either `Ie` (at Re) or `I0`
(central intensity), with `I0 = Ie exp(b_n)`; legacy `ie` remains readable.
`measure_profile` requires an explicit `SB_ZP`; `surface_brightness_zeropoint`
converts a flux ZP and intensity pixel scale. The notebook explains the ERO
VIS conversion and its inherited mean-binning assumption.

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
explicitly selected. Scientific inputs, generated outputs and historical
notebooks remain untracked.
