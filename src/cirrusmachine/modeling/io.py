"""Portable numeric calibration archives and model outputs."""

import io
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from .calibration import StatisticalCalibration
from .config import save_config


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_arrays(path):
    """Read one archive snapshot without pickle or mutable file-backed arrays."""
    with np.load(io.BytesIO(Path(path).read_bytes()), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_calibration(calibration, path):
    """Save all statistical calibration arrays and their input signature."""
    arrays = {}
    for group in ("noise", "texture", "targets"):
        arrays.update(
            {f"{group}/{k}": v for k, v in getattr(calibration, group).items()}
        )
    arrays["metadata"] = np.array(
        json.dumps(
            dict(signature=calibration.signature, diagnostics=calibration.diagnostics),
            default=_json,
            allow_nan=False,
        )
    )
    np.savez_compressed(path, **arrays)


def load_calibration(path, field, config):
    """Load calibration, rejecting changes to its data or numerical conventions."""
    arrays = load_arrays(path)
    meta = json.loads(str(arrays.pop("metadata")))
    groups = [
        {k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(group + "/")}
        for group in ("noise", "texture", "targets")
    ]
    result = StatisticalCalibration(*groups, meta["signature"], meta["diagnostics"])
    result.validate(field, config)
    return result


def save_modeling_results(results, directory, config, *, field=None):
    """Write FITS maps, normalized canvases, histories, seeds and runtime metadata.

    Existing output directories are rejected to protect prior runs. FITS WCS
    is included when a field with a working-canvas WCS is supplied.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_config(config, directory / "config.yaml")
    metadata = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        draw_count=len(results),
    )
    (directory / "runtime.json").write_text(json.dumps(metadata, indent=2))
    for i, result in enumerate(results):
        output = directory / f"draw_{i:03d}"
        output.mkdir()
        header = (
            fits.Header()
            if field is None or field.WCS is None
            else field.WCS.slice(field.crop).to_header()
        )
        header["BUNIT"] = "ADU"
        header["SKYADU"] = result.sky
        for name in ("cirrus", "corrected", "halo"):
            fits.writeto(
                output / f"{name}.fits", getattr(result, name).astype("f4"), header
            )
        np.save(output / "normalized.npy", result.normalized, allow_pickle=False)
        report = dict(
            seeds=asdict(result.seeds),
            history=result.history,
            diagnostics=result.diagnostics,
        )
        (output / "diagnostics.json").write_text(
            json.dumps(report, default=_json, allow_nan=False)
        )
