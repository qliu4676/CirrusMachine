"""Numerical settings for cirrus and diffuse light modeling."""

import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
import yaml

from .operators import resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrawSeeds:
    """Independent broad-field and texture phases."""

    broad: int = 2711003
    texture: int = 2721003


@dataclass(frozen=True)
class ModelingConfig:
    """Model controls; angular sizes are arcsec unless explicitly in pixels.

    ``auto`` selects CUDA when available, otherwise CPU. The covariance solves
    require float64 and complex FFTs, so MPS is not a supported backend.
    ``stage_steps`` gives the duration of each learning rate. Shorter runs
    stop within the schedule; longer runs retain its final learning rate.
    """

    device: str = "auto"
    pixel_scale: float = 0.9
    beam_FWHM: float = 36.0
    scales: int = 5
    orientations: int = 8
    minimum_kernel_coverage: float = 0.55
    orientation_pool_FWHM: float = 55
    texture_highpass_FWHM: float = 40
    CDF_FWHM: float = 40
    beta_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
    split_sigma_pixels: float = 12.0
    broad_length_pixels: float = 60.0
    prior_lowpass_pixels: float = 30.0
    prior_weight: float = 5.0
    weights: tuple[float, ...] = (1.0, 0.1, 1.0, 1.0, 0.01, 2.0)
    steps: int = 100
    learning_rates: tuple[float, ...] = (0.005, 0.003, 0.001)
    stage_steps: tuple[int, ...] = (40, 30, 30)
    halo_refresh_steps: int = 10
    target_refresh_steps: int = 25
    halo_damping: float = 0.5
    core_radius: float = 32.0
    core_taper: float = 5.0
    sky_annulus: tuple[float, float] = (180.0, 220.0)
    calibration_min_support: float = 0.5
    reference_count: int = 16
    validation_count: int = 8
    noise_seeds: tuple[int, ...] = tuple(range(990000, 990008))
    draws: tuple[DrawSeeds, ...] = field(default_factory=lambda: (DrawSeeds(),))
    allowance_seeds: DrawSeeds = field(
        default_factory=lambda: DrawSeeds(2711000, 2721000)
    )
    inner_refit: bool = True

    def __post_init__(self):
        if self.scales != 5 or self.orientations != 8:
            raise ValueError("This implementation preserves J=5 and L=8.")
        for name in (
            "pixel_scale",
            "beam_FWHM",
            "orientation_pool_FWHM",
            "texture_highpass_FWHM",
            "CDF_FWHM",
            "split_sigma_pixels",
            "broad_length_pixels",
            "prior_lowpass_pixels",
            "core_taper",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "steps",
            "halo_refresh_steps",
            "target_refresh_steps",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.weights) != 6 or any(
            not math.isfinite(v) or v < 0 for v in self.weights
        ):
            raise ValueError("weights must contain six finite nonnegative values")
        if not self.learning_rates or any(
            not math.isfinite(v) or v <= 0 for v in self.learning_rates
        ):
            raise ValueError("learning_rates must be positive")
        if (
            not isinstance(self.stage_steps, (tuple, list))
            or len(self.stage_steps) != len(self.learning_rates)
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in self.stage_steps
            )
        ):
            raise ValueError(
                "stage_steps must list one positive integer duration per learning rate"
            )
        if not self.beta_grid or any(
            not math.isfinite(v) or v <= 0 for v in self.beta_grid
        ):
            raise ValueError("beta_grid must contain positive slopes")
        if (
            len(self.sky_annulus) != 2
            or not 0 < self.sky_annulus[0] < self.sky_annulus[1]
        ):
            raise ValueError("sky_annulus must be increasing and positive")
        if not 0 <= self.halo_damping <= 1 or not 0 < self.minimum_kernel_coverage < 1:
            raise ValueError("Invalid halo damping or kernel support")
        if not 0 < self.calibration_min_support < 1 or self.core_radius < 0:
            raise ValueError("Invalid calibration support or core radius")
        if (
            self.reference_count < 2
            or self.validation_count < 1
            or len(self.noise_seeds) < 2
        ):
            raise ValueError(
                "Reference and noise ensembles need enough samples for variance"
            )
        if not math.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if not math.isfinite(self.core_radius):
            raise ValueError("core_radius must be finite")
        if not self.draws:
            raise ValueError("At least one draw is required")
        for seeds in (*self.draws, self.allowance_seeds):
            if not isinstance(seeds, DrawSeeds) or min(seeds.broad, seeds.texture) < 0:
                raise ValueError("draws must contain nonnegative DrawSeeds")


def load_config(path: str | Path, *, cpu_threads=None) -> ModelingConfig:
    """Read YAML and resolve ``auto`` to the available computation device.

    Unknown settings are rejected. ``cpu_threads`` optionally caps PyTorch CPU
    threads for this process; omitting it preserves the existing thread count.
    """
    if cpu_threads is not None and (
        isinstance(cpu_threads, bool)
        or not isinstance(cpu_threads, int)
        or cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be a positive integer or None")
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    for old, new in {
        "beam_fwhm": "beam_FWHM",
        "orientation_pool_fwhm": "orientation_pool_FWHM",
        "texture_highpass_fwhm": "texture_highpass_FWHM",
        "cdf_fwhm": "CDF_FWHM",
    }.items():
        if old in raw:
            if new in raw:
                raise ValueError(f"Specify only {new}, not both spellings")
            raw[new] = raw.pop(old)
    unknown = set(raw) - {f.name for f in fields(ModelingConfig)}
    if unknown:
        raise ValueError(f"Unknown modeling settings: {sorted(unknown)}")
    for key in ("beta_grid", "weights", "learning_rates", "sky_annulus", "noise_seeds"):
        if key in raw:
            raw[key] = tuple(raw[key])
    if isinstance(raw.get("stage_steps"), list):
        raw["stage_steps"] = tuple(raw["stage_steps"])
    if "draws" in raw:
        raw["draws"] = tuple(DrawSeeds(**v) for v in raw["draws"])
    if "allowance_seeds" in raw:
        raw["allowance_seeds"] = DrawSeeds(**raw["allowance_seeds"])
    config = ModelingConfig(**raw)
    device = resolve_device(config.device)
    config = replace(config, device=str(device))
    if device.type == "cpu" and cpu_threads is not None:
        torch.set_num_threads(
            min(cpu_threads, os.cpu_count() or 1, torch.get_num_threads())
        )
    logger.info(
        "Device: %s%s",
        config.device,
        f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "",
    )
    logger.info("Run: %d realization(s), %d steps", len(config.draws), config.steps)
    return config


class _ConfigDumper(yaml.SafeDumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        return super().represent_sequence(tag, sequence, flow_style=True)


def save_config(config: ModelingConfig, path: str | Path) -> None:
    """Save resolved settings as portable YAML with compact inline lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# All settings are optional: omitted keys use ModelingConfig defaults.\n"
        "# Seeds are optional reproducibility controls; usually keep the defaults.\n"
        + yaml.dump(
            asdict(config),
            Dumper=_ConfigDumper,
            sort_keys=False,
            default_flow_style=None,
            width=100,
        )
    )
