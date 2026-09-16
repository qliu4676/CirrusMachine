# CirrusMachine

Generate independent mock Galactic cirrus images for low-surface-brightness
astronomy, using wavelet scattering statistics, power spectra, phase coupling,
and local amplitude constraints. Coarse Planck or Herschel maps can condition
the large-scale structure. Foreground removal is a longer-term research goal.

## Install and check locally

Python 3.10+ is required. Run from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m cirrusmachine --output result/smoke.fits
```

The smoke command runs five CPU optimization steps on a small synthetic input
and saves only a FITS image. It tests installation and synthesis, not scientific
convergence. Output files are never overwritten by this command.
For a quick check using an available Dragonfly image:

```bash
python -m cirrusmachine \
  --input product/data/dfuws/UWtile_04703_luminance_filtered_kJy_sr.fits \
  --output result/dragonfly_smoke.fits
```

The command block-averages the input to 32 x 32 pixels; do not use that smoke
output to assess full-resolution morphology. The Python API also supports
explicit CPU operation:

```python
from cirrusmachine import MockSynthesisConfig, synthesize_single_image

# image is a finite, nonconstant 2D NumPy array.
mock = synthesize_single_image(
    image,
    config=MockSynthesisConfig(device="cpu", seed=4703, steps=100),
)["image_out"]
```

`requirements.txt` installs the same core dependencies as `setup.py`.
Optional extras: `.[notebooks]` for Jupyter; `.[astronomy]` for HEALPix input,
TurbuStat controls and mask filling. The original workflow-specific requirement
lists are retained for reference. The preserved Cheng scattering backend is
bundled under `src/scattering`; it is installed with this package, with its
upstream README retained. Do not install an unrelated package named scattering.

## Layout and data

- `src/cirrusmachine/`: reusable synthesis, calibration and neural-field code.
- `scripts/` and `configs/`: research workflows and CPU-default configurations.
- `tests/`: regression tests.
- `notebooks/`: four principal historical Colab notebooks, temporarily ignored.
- `product/`: untracked input FITS, prepared maps, target caches and intermediates.
- `result/`: untracked generated results and figures.

Science inputs are not included in Git. The migrated local data live under
`product/data/{dfuws,A2390,Perseus}`; cached preflights live under
`product/dfuws` and `product/A2390_pipeline`. A2390 prepared Herschel inputs
are also under `product/A2390_pipeline`. The old Drive results and training
checkpoints remain in the original project archive; they are not all duplicated.
Caches hash arrays and configuration, sometimes including absolute paths, so
moving a dataset can legitimately trigger recomputation. Historical cached
metadata retains its original paths and units.

Run `python scripts/<script>.py --help` for individual workflow options.
Some legacy scripts automatically write text/JSON experiment reports: use those
report-producing runs only when explicitly requested. Do not save additional
text/Markdown audit reports or records unless explicitly requested.

## Colab after local validation

First check CPU tests and small local runs, then validate the intended full
science configuration. Only after those checks should a GPU run be started.
Clone/sync this repository to Colab, make the ignored inputs available separately,
then install that checkout with `python -m pip install -e .`. The compatibility
setup scripts install this checkout without fetching or switching backend branches.
For CUDA-enabled workflows, explicitly set the configuration's runtime device to
`cuda` (or `gpu` for the notebook synthesis API). Some current conditioned-WST
scripts still explicitly use CPU and need a separate GPU-portability check.

The copied notebooks preserve their original cells and Colab paths; update their
mount/project paths before reuse. Git will sync source and configurations, not
ignored inputs, caches, results or notebooks. No Git remote is configured in this
local checkout yet.
