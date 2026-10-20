# CirrusMachine

Galactic cirrus synthesis and diffuse emission recovery with a coarse guide map
and wavelet scattering statistics.

## Install

Python 3.10+; from the repository checkout:

```bash
python -m pip install -e .
# Optional notebook environment:
python -m pip install -e ".[examples]"
```

CirrusMachine includes a private copy of [Cheng's scattering package](https://github.com/SihaoCheng/scattering_transform).
The [`cirrusmachine._scattering_synthesis`](src/cirrusmachine/_scattering_synthesis.py) adapter
retains the local image-increment optimization separately.
No GPU is required. The default `device="auto"` uses CUDA when available and
otherwise CPU; `device="cpu"` always uses CPU. GPU use is optional and requires
a compatible driver and a CUDA-enabled PyTorch build; MPS is unsupported.

## Example

Open [the modeling notebook](examples/cutout_example_modeling.ipynb) for diffuse
light injection, one representative cirrus realization with 100 optimization
steps, and surface-brightness and cumulative-flux comparisons.
[example_helpers.py](examples/example_helpers.py)
contains input and plotting helpers. Place the optional input bundle in
`examples/inputs/`: `Optical.fits`, `Mask.fits`, `FIR.fits`, `PSF.npz`, and
`inputs.json` (or set `CIRRUSMACHINE_DATA`). Notebook setup creates
`examples/output/` for generated files, including `statistics.npz`.
Pip installs dependencies; it does not create input or output directories.

With the observed `field` and `inputs` prepared as in the notebook:

```python
from cirrusmachine import modeling as cm

cm.configure_logging("INFO")
config = cm.load_config("examples/modeling.yaml")
statistics = cm.calibrate_statistics(field, config)
beam = cm.prepare_backbone_constraint(
    field, inputs["coarse_image"], inputs["coarse_error"],
    inputs["coarse_coverage"], inputs["coarse_mask"], inputs["coarse_WCS"], config,
    effective_FWHM=inputs["metadata"]["effective_FWHM"],
)
prior = cm.build_cirrus_prior(field, beam, statistics, config)
initial = cm.initialize_cirrus(prior, config)
result = cm.run_cirrus_halo_modeling(field, prior, initial, config)[0]
corrected = result.corrected
```

For injection test, both `Ie` (at `Re_arcsec`) and `I0` (central brightness) 
are in **mag/arcsec²**.

For a de Vaucouleurs profile injection: `n=4`, `Re_arcsec=20`, and
`Ie=24.0`<sup>†</sup>:

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
`run_cirrus_halo_modeling` provides joint modeling on cirrus and diffuse light.

†: Based on the Sérsic law: $I(R)=I_e\exp[-b_4((R/R_e)^{1/4}-1)]$ with
$b_4\simeq7.67$ ($I_e$ in linear intensity).