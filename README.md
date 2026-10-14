# CirrusMachine

Galactic cirrus synthesis and FIR-conditioned cirrus/halo modeling with wavelet
scattering statistics.

## Install

Python 3.10+; from the repository checkout:

```bash
git submodule update --init external/scattering_transform
python -m pip install -e .
# Optional notebook environment:
python -m pip install -e ".[examples]"
```

CirrusMachine uses [Cheng's scattering package](https://github.com/SihaoCheng/scattering_transform).
The [`cirrusmachine._scattering_synthesis`](src/cirrusmachine/_scattering_synthesis.py) adapter
retains the local image-increment optimization separately.
Install a CUDA-enabled PyTorch build for GPU use. `device="auto"` selects CUDA
when available and otherwise CPU; MPS is unsupported.

## Example

Open [the modeling notebook](notebooks/cutout_example_modeling.ipynb) for galaxy
injection, one representative cirrus realization, and surface-brightness and
cumulative-flux comparisons. [example_helpers.py](notebooks/example_helpers.py)
contains input and plotting helpers. Place the optional input bundle in
`product/examples/cutout_ddV` (or set `CIRRUSMACHINE_DATA`); copy it separately
to Colab. The notebook discovers the checkout and uses fresh output directories.

With the observed `field` and native FIR `inputs` prepared in the notebook:

```python
from cirrusmachine import modeling as cm

cm.configure_logging("INFO")
config = cm.load_config("configs/cutout_ddV.yaml")
statistics = cm.calibrate_statistics(field, config)
beam = cm.prepare_FIR_constraint(
    field, inputs["FIR_image"], inputs["FIR_error"],
    inputs["FIR_coverage"], inputs["FIR_mask"], inputs["FIR_WCS"], config,
    effective_FWHM=inputs["metadata"]["effective_FWHM"],
)
prior = cm.build_cirrus_prior(field, beam, statistics, config)
initial = cm.initialize_cirrus(prior, config)
result = cm.run_cirrus_halo_modeling(field, prior, initial, config)[0]
corrected = result.corrected
```

Injection components accept `n`, `Re_arcsec`, and either `Ie` (at Re) or `I0`
(central intensity). Photometry uses an explicit `SB_ZP`; the notebook explains
the pixel-scale and intensity conventions.
