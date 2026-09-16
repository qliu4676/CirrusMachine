"""Notebook-compatible mock cirrus synthesis helpers.

The current notebooks follow a repeated recipe:

1. Load a 2D cirrus image.
2. Optionally bin it with ``astropy.nddata.block_reduce`` in the notebook.
3. Crop to even dimensions.
4. Add an image axis so the array has shape ``(N_image, M, N)``.
5. Reflect-pad by ``M//2`` and ``N//2`` to soften boundary artifacts.
6. Call ``scattering.synthesis`` with the padded image.
7. Crop the synthesized image back to the original footprint.

This module preserves that behavior.  It does not change the bundled
``scattering`` loss definitions or coefficient selection.

Notebook example
----------------
```python
from astropy.io import fits
from astropy.nddata import block_reduce
from cirrusmachine.synthesis import MockSynthesisConfig, synthesize_single_image

raw = fits.getdata("/content/sample_data/WISE_cutout_MJy_ps_rm_lpf_6arcsec.fits")
image = block_reduce(raw, 4, func=np.mean)
result = synthesize_single_image(
    image,
    config=MockSynthesisConfig(J=None, L=4, seed=6946, device="gpu"),
    scattering_module=scattering,
)
image_out = result["image_out"]
```
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MockSynthesisConfig:
    """Configuration matching the current notebook synthesis calls."""

    estimator_name: str = "s_cov"
    J: Optional[int] = None
    L: int = 4
    seed: Optional[int] = None
    device: str = "cpu"
    image_init: Any = "random phase"
    pad_mode: str = "reflect"
    optim_algorithm: str = "LBFGS"
    steps: int = 300
    learning_rate: float = 0.2


def dyadic_size(shape: Sequence[int]) -> int:
    """Return the largest dyadic scale used by the notebooks."""

    return int(np.ceil(np.log2(min(shape)) - 1))


def crop_to_even(image: np.ndarray) -> np.ndarray:
    """Crop the last two dimensions to even sizes, preserving leading axes."""

    arr = np.asarray(image)
    m, n = arr.shape[-2:]
    return arr[..., : m // 2 * 2, : n // 2 * 2]


def ensure_image_stack(image: np.ndarray) -> np.ndarray:
    """Return an array with notebook-compatible shape ``(N_image, M, N)``."""

    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected a 2D image or 3D image stack, got shape {arr.shape}.")


def reflect_pad(image_stack: np.ndarray, pad_mode: str = "reflect") -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a 3D image stack by half the image size in each spatial dimension."""

    stack = ensure_image_stack(image_stack)
    m, n = stack.shape[-2:]
    pad = (m // 2, n // 2)
    padded = np.pad(stack, pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode=pad_mode)
    return padded, pad


def crop_reflect_padding(image_stack: np.ndarray, pad: tuple[int, int]) -> np.ndarray:
    """Crop the half-size reflect padding used by :func:`reflect_pad`."""

    pad_m, pad_n = pad
    end_m = -pad_m if pad_m else None
    end_n = -pad_n if pad_n else None
    return np.asarray(image_stack)[..., pad_m:end_m, pad_n:end_n]


def _resolve_scattering(scattering_module: Any = None) -> Any:
    if scattering_module is not None:
        return scattering_module
    try:
        import scattering  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import the bundled scattering package. In notebooks, run "
            "setup_scattering_colab.py or add scattering_transform to sys.path first."
        ) from exc
    return scattering


def synthesize_padded_mock(
    image_padded: np.ndarray,
    *,
    config: MockSynthesisConfig | None = None,
    scattering_module: Any = None,
    **synthesis_kwargs: Any,
) -> np.ndarray:
    """Call ``scattering.synthesis`` on an already padded image stack.

    This mirrors notebook cells that directly pass ``image_padded`` to
    ``scattering.synthesis``.  Extra keyword arguments are forwarded unchanged.
    """

    cfg = config or MockSynthesisConfig()
    scattering = _resolve_scattering(scattering_module)
    stack = ensure_image_stack(image_padded)
    j = dyadic_size(stack.shape[-2:]) if cfg.J is None else cfg.J
    kwargs = {
        "J": j,
        "L": cfg.L,
        "seed": cfg.seed,
        "device": cfg.device,
        "image_init": cfg.image_init,
        "optim_algorithm": cfg.optim_algorithm,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
    }
    kwargs.update(synthesis_kwargs)
    return scattering.synthesis(cfg.estimator_name, stack, **kwargs)


def synthesize_from_estimator(
    target_coefficients: np.ndarray,
    *,
    image_shape: tuple[int, int],
    image_init: np.ndarray,
    config: MockSynthesisConfig | None = None,
    scattering_module: Any = None,
    **synthesis_kwargs: Any,
) -> np.ndarray:
    """Run the estimator-mode recipe used by Bullseye and ICL notebooks."""

    cfg = config or MockSynthesisConfig()
    scattering = _resolve_scattering(scattering_module)
    m, n = image_shape
    j = dyadic_size(image_shape) if cfg.J is None else cfg.J
    kwargs = {
        "J": j,
        "L": cfg.L,
        "M": m,
        "N": n,
        "mode": "estimator",
        "seed": cfg.seed,
        "device": cfg.device,
        "image_init": image_init,
        "optim_algorithm": cfg.optim_algorithm,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
    }
    kwargs.update(synthesis_kwargs)
    return scattering.synthesis(cfg.estimator_name, target_coefficients, **kwargs)


def synthesize_single_image(
    image: np.ndarray,
    *,
    config: MockSynthesisConfig | None = None,
    scattering_module: Any = None,
    **synthesis_kwargs: Any,
) -> Dict[str, np.ndarray | tuple[int, int] | int]:
    """Run the current single-image notebook recipe end to end."""

    cfg = config or MockSynthesisConfig()
    image_stack = ensure_image_stack(crop_to_even(image))
    j = dyadic_size(image_stack.shape[-2:]) if cfg.J is None else cfg.J
    cfg_for_call = dataclasses.replace(cfg, J=j)
    image_padded, pad = reflect_pad(image_stack, pad_mode=cfg.pad_mode)
    image_syn = synthesize_padded_mock(
        image_padded,
        config=cfg_for_call,
        scattering_module=scattering_module,
        **synthesis_kwargs,
    )
    image_out = crop_reflect_padding(image_syn[:1], pad)
    return {
        "image": image_stack,
        "image_padded": image_padded,
        "image_syn": image_syn,
        "image_out": image_out,
        "pad": pad,
        "J": j,
    }


def synthesize_batch(
    image_padded: np.ndarray,
    *,
    seeds: Iterable[int],
    output_shape: tuple[int, int],
    config: MockSynthesisConfig | None = None,
    scattering_module: Any = None,
    **synthesis_kwargs: Any,
) -> np.ndarray:
    """Generate a batch by looping over seeds, matching WISE/HeLMS notebooks."""

    cfg = config or MockSynthesisConfig()
    m, n = output_shape
    pad = (m // 2, n // 2)
    outputs = []
    for seed in seeds:
        cfg_seed = MockSynthesisConfig(
            estimator_name=cfg.estimator_name,
            J=cfg.J,
            L=cfg.L,
            seed=seed,
            device=cfg.device,
            image_init=cfg.image_init,
            pad_mode=cfg.pad_mode,
            optim_algorithm=cfg.optim_algorithm,
            steps=cfg.steps,
            learning_rate=cfg.learning_rate,
        )
        image_syn = synthesize_padded_mock(
            image_padded,
            config=cfg_seed,
            scattering_module=scattering_module,
            **synthesis_kwargs,
        )
        outputs.append(crop_reflect_padding(image_syn[:1], pad)[0])
    return np.stack(outputs, axis=0)
