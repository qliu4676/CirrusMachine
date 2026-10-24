# CirrusMachine

Galactic cirrus synthesis and diffuse emission recovery with a coarse guide map
and wavelet scattering statistics.

## Install

Python 3.10+:

```bash
python -m pip install cirrusmachine
# Optional notebook environment:
python -m pip install "cirrusmachine[examples]"
```

The `[examples]` extra adds notebook dependencies; download the notebook and
optional input files separately.

CirrusMachine includes a private copy of [Cheng's scattering package](https://github.com/SihaoCheng/scattering_transform).
The `cirrusmachine._scattering_synthesis` adapter
retains the local image-increment optimization separately.

The default `device="auto"` uses CUDA when available and otherwise CPU. Setting `device="cpu"` to use CPU only. 
GPU use is optional and requires a compatible driver and a CUDA-enabled PyTorch build. Apple Silicon Mac's MPS is unsupported.

## Example

The notebook [the modeling notebook](examples/cutout_example_modeling.ipynb) 
demonstrates an examplefor diffuse light injection, one representative cirrus 
realization with 100 optimization steps, and surface-brightness and cumulative-flux comparisons.

[example_helpers.py](examples/example_helpers.py) contains input and plotting helper function for the example. 

To run the example, keep the optional input bundle in
`examples/inputs/`: `Optical.fits`, `Mask.fits`, `FIR.fits`, `PSF.npz`, and
`inputs.json` (or set `CIRRUSMACHINE_DATA`). The notebook will create
`examples/output/` for generated files, including the caches (`statistics.npz`).


A Google Colab T4 GPU test of the 100-step example is about 8–9× faster in
preparation/calibration and 4–5× faster in modeling than the CPU backend on the
same Colab host (2 vCPUs).

## Main usage

```python
from cirrusmachine import modeling as cm

cm.configure_logging("INFO")
config = cm.load_config("examples/modeling.yaml")

# Calibrate image statistics and the FIR coarse-guide backbone constraint.
statistics = cm.calibrate_statistics(field, config)
beam = cm.prepare_backbone_constraint(
    field, inputs["coarse_image"], inputs["coarse_error"],
    inputs["coarse_coverage"], inputs["coarse_mask"], inputs["coarse_WCS"], config,
    effective_FWHM=inputs["metadata"]["effective_FWHM"],
)

# Build the cirrus prior and generate initial realizations.
prior = cm.build_cirrus_prior(field, beam, statistics, config)
initial = cm.initialize_cirrus(prior, config)

# Model cirrus and diffuse light; select the first realization.
result = cm.run_cirrus_halo_modeling(field, prior, initial, config)[0]
corrected = result.corrected  # Observed image with modeled cirrus subtracted.
```

* Passing `diffuse_model=MY_MODEL` to `run_cirrus_halo_modeling` allows one to replace 
the default isophote fit with a callable matching `fit_diffuse` inputs and 
`(model_image, info)` output (including `info["sky"]["value_ADU"]`).

** For injection test, both `Ie` (at `Re_arcsec`) and `I0` (central brightness) 
are in mag/arcsec².

For a de Vaucouleurs profile injection: `n=4`, `Re_arcsec=20`, and
`Ie=24.0`:

```python
injection = cm.prepare_injection(
    field, [{"n": 4, "Re_arcsec": 20.0, "Ie": 24.0}],
    SB_ZP=SB_ZP, PSF_native=inputs["PSF_detector"],
    oversample=9, device=config.device,
)
field = injection.field  
# Then calibrate and model this field.
```

In the example, the coarse guide is the source-clean FIR image from Herschel.
`modeling.coarse` handles coarse guide calibration. 
`modeling.diffuse` provides fit for smooth diffuse light.
`run_cirrus_halo_modeling` do joint modeling on cirrus and diffuse light.
