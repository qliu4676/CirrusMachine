"""Typed YAML configuration for neural-field experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass(frozen=True)
class InputConfig:
    products: str
    herschel_cutout: str
    herschel_cutouts: tuple[str, ...] = ()
    planck_hdu: str = "PLANCK_RAD"
    herschel_full_hdu: str = "CIBRM_RAD"
    temperature_hdu: str = "DUST_TEMP"
    beta_hdu: str = "DUST_BETA"
    radiance_slope: float | None = None
    radiance_offset: float | None = None
    target_square_pixels: int = 440
    pixel_scale_arcsec: float = 6.0
    effective_resolution_arcsec: float = 36.0


@dataclass(frozen=True)
class SplitConfig:
    holdout_size: int = 50
    holdout_origin: tuple[int, int] = (95, 95)
    buffer_pixels: int = 5
    minimum_valid_fraction: float = 1.0


@dataclass(frozen=True)
class TransitionConfig:
    dyadic_index: int = 6
    width_fraction: float = 0.0
    fine_scale_arcmin: float | None = None
    coarse_scale_arcmin: float | None = None
    apodization_fraction: float = 0.0
    boundary_padding_pixels: int = 0
    boundary_taper_fraction: float = 0.0
    complementary_blend: bool = False
    render_margin_pixels: int = 0
    final_split_cosine_alpha: float = 0.0
    final_split_cosine_beta: float = 0.9
    final_apodization_mode: str = "radial"


@dataclass(frozen=True)
class ArchitectureConfig:
    grid_resolutions: tuple[int, ...] = (16, 32, 64, 128)
    channels_per_level: int = 4
    hidden_features: int = 64
    hidden_layers: int = 2
    output_scale: float = 1.0
    include_coordinates: bool = True


@dataclass(frozen=True)
class SWDConfig:
    patch_sizes: tuple[int, ...] = (16, 32)
    patch_batch: int = 64
    target_samples: int = 2048
    projections: int = 64
    quantiles: int = 128
    projection_chunk: int = 4096
    morlet_scales: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    morlet_orientations: int = 4
    morlet_stride: int = 2
    morlet_samples: int = 4096
    orientation_mode: str = "observed"


@dataclass(frozen=True)
class PowerConfig:
    enabled: bool = False
    patch_size: int = 64
    patch_batch: int = 16
    target_samples: int = 128
    bins: int = 8
    minimum_scale_arcmin: float = 4.0
    maximum_scale_arcmin: float = 0.6
    huber_delta: float = 1.0
    warmup_steps: int = 50
    global_bins: int = 10
    global_minimum_scale_arcmin: float = 10.0
    projection_iterations: int = 0
    normalize_backbone_transition: bool = False
    normalization_crop_arcmin: float = 30.0
    normalization_anchor_scale_arcmin: float = 15.0
    normalization_max_gain: float = 5.0


@dataclass(frozen=True)
class MD07Config:
    enabled: bool = True
    scale_indices: tuple[int, ...] = (4, 5, 6)
    context_factor: float = 3.0
    spacing_factor: float = 0.5
    amplitude_correction: float | str = "fit"
    extrapolate: bool = True
    huber_delta: float = 0.5
    warmup_steps: int = 75


@dataclass(frozen=True)
class LossConfig:
    patch_swd: float = 0.5
    morlet_swd: float = 0.5
    md07: float = 0.1
    power: float = 0.0
    isotropy: float = 0.0
    global_power: float = 0.0
    boundary: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    steps: int = 300
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    mixed_precision: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    seeds: tuple[int, ...] = (4676, 4677, 4678)
    coordinate_chunk: int = 262144
    diagnostic_interval: int = 100
    checkpoint_interval: int = 100
    deterministic: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str = "a2390_latent_grid_j6"
    overwrite: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    inputs: InputConfig
    output: OutputConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    swd: SWDConfig = field(default_factory=SWDConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    md07: MD07Config = field(default_factory=MD07Config)
    losses: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.inputs.target_square_pixels < 32:
            raise ValueError("target_square_pixels must be at least 32")
        if self.inputs.herschel_cutouts and self.inputs.herschel_cutout not in self.inputs.herschel_cutouts:
            raise ValueError("herschel_cutout must be included in herschel_cutouts for validation")
        if self.inputs.pixel_scale_arcsec <= 0 or self.inputs.effective_resolution_arcsec <= 0:
            raise ValueError("Pixel scale and resolution must be positive")
        if self.transition.dyadic_index < 1:
            raise ValueError("transition.dyadic_index must be positive")
        if not 0.0 <= self.transition.width_fraction < 1.0:
            raise ValueError("transition.width_fraction must lie in [0, 1)")
        transition_pair = (
            self.transition.fine_scale_arcmin,
            self.transition.coarse_scale_arcmin,
        )
        if any(value is not None for value in transition_pair):
            if any(value is None for value in transition_pair):
                raise ValueError("Both transition scale limits must be provided")
            if not 0 < transition_pair[0] < transition_pair[1]:
                raise ValueError("Transition scales require 0 < fine < coarse")
        if not 0.0 <= self.transition.apodization_fraction <= 1.0:
            raise ValueError("transition.apodization_fraction must lie in [0, 1]")
        if self.transition.boundary_padding_pixels < 0:
            raise ValueError("transition.boundary_padding_pixels cannot be negative")
        if self.transition.render_margin_pixels < 0:
            raise ValueError("transition.render_margin_pixels cannot be negative")
        if not 0.0 <= self.transition.boundary_taper_fraction <= 1.0:
            raise ValueError("transition.boundary_taper_fraction must lie in [0, 1]")
        if not 0.0 <= self.transition.final_split_cosine_alpha <= 1.0:
            raise ValueError("transition.final_split_cosine_alpha must lie in [0, 1]")
        if not 0.0 <= self.transition.final_split_cosine_beta <= 1.0:
            raise ValueError("transition.final_split_cosine_beta must lie in [0, 1]")
        if self.transition.final_apodization_mode not in {"radial", "separable"}:
            raise ValueError("transition.final_apodization_mode must be 'radial' or 'separable'")
        if any(size < 2 for size in self.swd.patch_sizes):
            raise ValueError("All SWD patch sizes must be at least two")
        if self.swd.projections < 1 or self.swd.quantiles < 2:
            raise ValueError("SWD needs positive projections and at least two quantiles")
        if self.swd.orientation_mode not in {"observed", "isotropic"}:
            raise ValueError("swd.orientation_mode must be 'observed' or 'isotropic'")
        if self.power.enabled:
            if self.power.patch_size < 8 or self.power.bins < 2:
                raise ValueError("Power patches and bins are too small")
            if self.power.global_bins < 2:
                raise ValueError("Global power needs at least two bins")
            if self.power.projection_iterations < 0:
                raise ValueError("Power projection iterations cannot be negative")
            if self.power.normalization_crop_arcmin <= 0:
                raise ValueError("Power normalization crop must be positive")
            if self.power.normalization_anchor_scale_arcmin <= self.transition.coarse_scale_arcmin:
                raise ValueError("Power normalization anchor must exceed the coarse transition scale")
            if self.power.normalization_max_gain < 1.0:
                raise ValueError("Power normalization maximum gain must be at least one")
            if not 0 < self.power.maximum_scale_arcmin < self.power.minimum_scale_arcmin:
                raise ValueError("Power scales require 0 < maximum < minimum")
        if self.losses.isotropy > 0 and not self.power.enabled:
            raise ValueError("The isotropy loss reuses power patches; enable power")
        if not self.runtime.seeds:
            raise ValueError("At least one seed is required")
        if self.optimizer.steps < 1 or self.optimizer.learning_rate <= 0:
            raise ValueError("Optimizer steps and learning rate must be positive")
        if self.runtime.checkpoint_interval < 1 or self.runtime.diagnostic_interval < 1:
            raise ValueError("Checkpoint and diagnostic intervals must be positive")
        if self.runtime.coordinate_chunk < 1:
            raise ValueError("runtime.coordinate_chunk must be positive")
        if isinstance(self.md07.amplitude_correction, (int, float)):
            if float(self.md07.amplitude_correction) <= 0:
                raise ValueError("MD07 amplitude correction must be positive")
        elif self.md07.amplitude_correction != "fit":
            raise ValueError("MD07 amplitude correction must be positive or 'fit'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any], section: str) -> T:
    if not isinstance(values, dict):
        raise TypeError(f"Configuration section {section!r} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    converted: dict[str, Any] = {}
    for key, value in values.items():
        hinted = hints.get(key)
        if isinstance(value, list) and hinted is not None and "tuple" in str(hinted):
            value = tuple(value)
        converted[key] = value
    return cls(**converted)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate one YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping")
    section_types = {
        "inputs": InputConfig,
        "output": OutputConfig,
        "split": SplitConfig,
        "transition": TransitionConfig,
        "architecture": ArchitectureConfig,
        "swd": SWDConfig,
        "power": PowerConfig,
        "md07": MD07Config,
        "losses": LossConfig,
        "optimizer": OptimizerConfig,
        "runtime": RuntimeConfig,
    }
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    required = {"inputs", "output"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    kwargs = {
        name: _construct(section_types[name], values, name)
        for name, values in raw.items()
    }
    config = ExperimentConfig(**kwargs)
    config.validate()
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved configuration without Python-specific YAML tags."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(config.as_dict(), stream, sort_keys=False)
