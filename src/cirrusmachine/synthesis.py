"""Mock cirrus synthesis with reflect padding and scattering statistics.

Images are cropped to even dimensions, reflect-padded to soften boundaries,
optimized through the synthesis adapter, and cropped to their original footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MockSynthesisConfig:
    """Configuration for scattering-constrained image synthesis."""

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
    """Return the dyadic scale count for the smaller image dimension."""

    return int(np.ceil(np.log2(min(shape)) - 1))


def crop_to_even(image: np.ndarray) -> np.ndarray:
    """Crop the last two dimensions to even sizes, preserving leading axes."""

    arr = np.asarray(image)
    m, n = arr.shape[-2:]
    return arr[..., : m // 2 * 2, : n // 2 * 2]


def ensure_image_stack(image: np.ndarray) -> np.ndarray:
    """Return an array with shape ``(N_image, M, N)``."""

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
    from . import _scattering_synthesis as scattering

    return scattering


def synthesize_padded_mock(
    image_padded: np.ndarray,
    *,
    config: MockSynthesisConfig | None = None,
    scattering_module: Any = None,
    **synthesis_kwargs: Any,
) -> np.ndarray:
    """Run synthesis on an already padded image stack.

    Extra keyword arguments are forwarded to the synthesis adapter.
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
    """Synthesize an image from target scattering coefficients."""

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
    """Pad, synthesize, and crop a single image."""

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
    """Generate a batch of realizations using the supplied seeds."""

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
