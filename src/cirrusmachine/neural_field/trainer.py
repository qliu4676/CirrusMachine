"""Fast full-field training with patch-distribution losses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as torch_f
from astropy.io import fits

from cirrusmachine.m07_modulation import M07LocalRMSConstraint, M07ModulationConfig
from cirrusmachine.planck_dust import modified_blackbody_band_to_I100
from cirrusmachine.power_spectrum import split_cosine_bell_window
from cirrusmachine.scattering_mean import ScatteringMean2D

from .calibration import fit_md07_amplitude_correction
from .config import ExperimentConfig, save_config
from .data import NeuralFieldData, sample_origins, valid_patch_origins
from .diagnostics import (
    basic_metrics,
    boundary_metrics,
    distribution_metrics,
    nearest_patch_metrics,
    plot_neural_field_state,
    plot_core_diagnostics,
    power_metrics,
    plot_power_comparison,
    plot_texture_reference_power_spectra,
    plot_training_history,
    structure_tensor_metrics,
    write_metrics,
)
from .features import (
    OrientedMorletFeatures,
    RobustNormalizer,
    angular_power_anisotropy,
    boundary_gradient_excess,
    dihedral_augment_patches,
    extract_patch_vectors,
    radial_log_power_features,
    radial_log_power_physical,
    subsample_rows,
)
from .losses import sliced_wasserstein_loss
from .model import MultiResolutionLatentField
from .operators import separable_split_cosine_window
from .render import (
    normalize_backbone_transition_power,
    phase_preserving_power_projection,
    render_synthesis,
    write_synthesis_fits,
)


def resolve_device(requested: str) -> torch.device:
    name = requested.lower()
    if name in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"mps", "gpu"} and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(config: ExperimentConfig, device: torch.device) -> MultiResolutionLatentField:
    architecture = config.architecture
    return MultiResolutionLatentField(
        architecture.grid_resolutions,
        architecture.channels_per_level,
        architecture.hidden_features,
        architecture.hidden_layers,
        architecture.output_scale,
        architecture.include_coordinates,
    ).to(device)


class NeuralFieldTrainer:
    """Own target construction and one-seed optimization."""

    def __init__(self, config: ExperimentConfig, data: NeuralFieldData) -> None:
        self.config = config
        self.data = data
        self.device = resolve_device(config.runtime.device)
        self.transition_pixels = float(2**config.transition.dyadic_index)
        self.transition_range_pixels = None
        if config.transition.fine_scale_arcmin is not None:
            pixel_scale_arcmin = data.pixel_scale_arcsec / 60.0
            self.transition_range_pixels = (
                config.transition.fine_scale_arcmin / pixel_scale_arcmin,
                config.transition.coarse_scale_arcmin / pixel_scale_arcmin,
            )
        self.resolution_pixels = (
            config.inputs.effective_resolution_arcsec / data.pixel_scale_arcsec
        )
        self.backbone_numpy, self.power_normalization_metrics = self._normalize_backbone()
        self.backbone = torch.as_tensor(
            self.backbone_numpy, device=self.device, dtype=torch.float32
        )
        self.exemplar = torch.as_tensor(data.exemplar, device=self.device, dtype=torch.float32)
        self.texture_exemplars = tuple(
            torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for value in data.texture_exemplars
        )
        self.patch_targets = self._build_patch_targets()
        self.morlet_target, self.morlet_normalizer, self.morlet_output = self._build_morlet_targets()
        self.power_target = self._build_power_target()
        self.global_power_target = self._build_global_power_target()
        self.md07_correction, self.md07_fit_metrics = self._resolve_md07_correction()
        self.md07 = self._build_md07_constraint()

    def _normalize_backbone(self) -> tuple[np.ndarray, dict[str, float]]:
        if not self.config.power.normalize_backbone_transition:
            return self.data.backbone, {"power_normalization_enabled": 0.0}
        inputs = self.config.inputs
        with fits.open(inputs.products, memmap=True) as hdul:
            radiance = np.asarray(hdul[inputs.planck_hdu].data, dtype=np.float32)
            herschel = np.asarray(hdul[inputs.herschel_full_hdu].data, dtype=np.float32)
        planck = self.data.radiance_slope * radiance + self.data.radiance_offset
        pixel_scale_arcmin = self.data.pixel_scale_arcsec / 60.0
        size = int(round(self.config.power.normalization_crop_arcmin / pixel_scale_arcmin))
        size = min(size, *planck.shape)
        y0 = (planck.shape[0] - size) // 2
        x0 = (planck.shape[1] - size) // 2
        region = (slice(y0, y0 + size), slice(x0, x0 + size))
        normalized, metrics = normalize_backbone_transition_power(
            self.data.backbone,
            planck[region],
            herschel[region],
            pixel_scale_arcsec=self.data.pixel_scale_arcsec,
            fine_scale_arcmin=self.config.transition.fine_scale_arcmin,
            coarse_scale_arcmin=self.config.transition.coarse_scale_arcmin,
            anchor_scale_arcmin=self.config.power.normalization_anchor_scale_arcmin,
            maximum_gain=self.config.power.normalization_max_gain,
        )
        metrics["power_normalization_crop_pixels"] = float(size)
        return normalized, metrics

    def _build_patch_targets(self):
        setup = {}
        rng = np.random.default_rng(self.config.runtime.seeds[0] + 101)
        for size in self.config.swd.patch_sizes:
            target_vectors = []
            per_cutout = max(1, self.config.swd.target_samples // len(self.texture_exemplars))
            for image, mask in zip(
                self.texture_exemplars, self.data.texture_training, strict=True
            ):
                origins = valid_patch_origins(
                    mask, size, self.config.split.minimum_valid_fraction
                )
                chosen = sample_origins(origins, per_cutout, rng)
                target_vectors.append(extract_patch_vectors(image, chosen, size))
            features = torch.cat(target_vectors, dim=0)
            if self.config.swd.orientation_mode == "isotropic":
                patches = features.reshape(-1, size, size)
                features = dihedral_augment_patches(patches).reshape(len(patches), -1)
            normalizer = RobustNormalizer.fit(features.detach().cpu().numpy())
            setup[size] = (normalizer.apply(features).detach(), normalizer, origins)
        return setup

    def _final_apodization_flat_mask(self) -> np.ndarray:
        shape = self.data.backbone.shape
        alpha = self.config.transition.final_split_cosine_alpha
        beta = self.config.transition.final_split_cosine_beta
        if self.config.transition.final_apodization_mode == "separable":
            window = separable_split_cosine_window(
                shape, alpha, beta, device="cpu", dtype=torch.float32
            ).numpy()
        else:
            window = split_cosine_bell_window(shape, alpha=alpha, beta=beta)
        return window >= 1.0 - 1e-6

    def _build_morlet_targets(self):
        scales = self.config.swd.morlet_scales
        j = max(scales) + 1
        target_calc = ScatteringMean2D(
            self.data.exemplar.shape,
            j,
            self.config.swd.morlet_orientations,
            pad_pixels=39,
            mask_boundary=False,
            device=str(self.device),
        )
        target_descriptor = OrientedMorletFeatures(
            target_calc,
            scales,
            spatial_stride=self.config.swd.morlet_stride,
            orientation_pool=self.config.swd.orientation_mode == "isotropic",
        )
        target = torch.cat(
            [
                target_descriptor.features(image, center_mask=mask)
                for image, mask in zip(
                    self.texture_exemplars, self.data.texture_training, strict=True
                )
            ],
            dim=0,
        )
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.runtime.seeds[0] + 102
        )
        target = subsample_rows(target, self.config.swd.morlet_samples, generator)
        normalizer = RobustNormalizer.fit(target.detach().cpu().numpy())
        render_margin = self.config.transition.render_margin_pixels
        output_shape = tuple(
            size + 2 * render_margin for size in self.data.backbone.shape
        )
        output_calc = ScatteringMean2D(
            output_shape,
            j,
            self.config.swd.morlet_orientations,
            pad_pixels=64,
            mask_boundary=False,
            device=str(self.device),
        )
        output_descriptor = OrientedMorletFeatures(
            output_calc,
            scales,
            spatial_stride=self.config.swd.morlet_stride,
            orientation_pool=self.config.swd.orientation_mode == "isotropic",
        )
        self.morlet_output_mask = np.zeros(output_shape, dtype=bool)
        delivered_mask = self._final_apodization_flat_mask()
        if render_margin:
            self.morlet_output_mask[
                render_margin:-render_margin, render_margin:-render_margin
            ] = delivered_mask
        else:
            self.morlet_output_mask[:] = delivered_mask
        return normalizer.apply(target).detach(), normalizer, output_descriptor

    def _resolve_md07_correction(self) -> tuple[float, dict[str, float]]:
        requested = self.config.md07.amplitude_correction
        if not self.config.md07.enabled:
            return 1.0, {"enabled": 0.0}
        if requested != "fit":
            return float(requested), {"fitted_correction": float(requested), "fitted": 0.0}
        correction, metrics = fit_md07_amplitude_correction(
            self.data.exemplar,
            self.data.exemplar_planck,
            self.data.exemplar_band_to_i100,
            self.data.pixel_scale_arcsec,
            scale_indices=self.config.md07.scale_indices,
            context_factor=self.config.md07.context_factor,
            spacing_factor=self.config.md07.spacing_factor,
            extrapolate=self.config.md07.extrapolate,
        )
        metrics["fitted"] = 1.0
        return correction, metrics

    def _build_power_target(self):
        if not self.config.power.enabled:
            return None
        setup = self.config.power
        # Power is an aggregate statistic rather than a memorization-sensitive
        # patch target, so use the complete valid cutout to resolve the 4--7'
        # transition despite the held-out square.
        rng = np.random.default_rng(self.config.runtime.seeds[0] + 103)
        exemplar_vectors = []
        per_cutout = max(1, setup.target_samples // len(self.texture_exemplars))
        for image in self.texture_exemplars:
            origins = valid_patch_origins(
                np.isfinite(image.detach().cpu().numpy()),
                setup.patch_size,
                self.config.split.minimum_valid_fraction,
            )
            chosen = sample_origins(origins, min(per_cutout, len(origins)), rng)
            exemplar_vectors.append(extract_patch_vectors(image, chosen, setup.patch_size))
        patches = torch.cat(exemplar_vectors, dim=0).reshape(-1, setup.patch_size, setup.patch_size)
        pixel_scale_arcmin = self.data.pixel_scale_arcsec / 60.0
        values = radial_log_power_features(
            patches,
            bins=setup.bins,
            minimum_frequency=pixel_scale_arcmin / setup.minimum_scale_arcmin,
            maximum_frequency=pixel_scale_arcmin / setup.maximum_scale_arcmin,
        )
        if len(values) == 1:
            normalizer = RobustNormalizer(
                center=values[0].detach().cpu().numpy().astype(np.float32),
                scale=np.ones(setup.bins, dtype=np.float32),
            )
        else:
            normalizer = RobustNormalizer.fit(
                values.detach().cpu().numpy(), floor_fraction=0.05
            )
        exemplar_center = values.mean(dim=0)

        backbone_origins = valid_patch_origins(
            np.ones(self.data.backbone.shape, dtype=bool), setup.patch_size, 1.0
        )
        backbone_chosen = sample_origins(backbone_origins, setup.target_samples, rng)
        backbone_vectors = extract_patch_vectors(
            self.backbone, backbone_chosen, setup.patch_size
        )
        backbone_values = radial_log_power_features(
            backbone_vectors.reshape(-1, setup.patch_size, setup.patch_size),
            bins=setup.bins,
            minimum_frequency=pixel_scale_arcmin / setup.minimum_scale_arcmin,
            maximum_frequency=pixel_scale_arcmin / setup.maximum_scale_arcmin,
        )
        backbone_center = backbone_values.mean(dim=0)
        edges = torch.logspace(
            np.log10(pixel_scale_arcmin / setup.minimum_scale_arcmin),
            np.log10(pixel_scale_arcmin / setup.maximum_scale_arcmin),
            setup.bins + 1,
            device=self.device,
        )
        frequencies = 0.5 * (edges[:-1] + edges[1:])
        lower = pixel_scale_arcmin / self.config.transition.coarse_scale_arcmin
        upper = pixel_scale_arcmin / self.config.transition.fine_scale_arcmin
        phase = ((frequencies - lower) / (upper - lower)).clamp(0.0, 1.0)
        alpha = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        planck_weight = (1.0 - alpha).clamp_min(torch.finfo(alpha.dtype).tiny)
        texture_weight = alpha.clamp_min(torch.finfo(alpha.dtype).tiny)
        target_center = torch.logaddexp(
            backbone_center + torch.log(planck_weight),
            exemplar_center + torch.log(texture_weight),
        )
        target_center = torch.where(alpha <= 0, backbone_center, target_center)
        target_center = torch.where(alpha >= 1, exemplar_center, target_center)
        return normalizer, normalizer.apply(target_center[None]).squeeze(0).detach()

    def _build_md07_constraint(self):
        if not self.config.md07.enabled:
            return None
        conversion = modified_blackbody_band_to_I100(
            self.data.dust_temperature,
            self.data.dust_beta,
            250.0,
        ).astype(np.float32)
        return M07LocalRMSConstraint(
            self.data.backbone,
            self.data.pixel_scale_arcsec,
            M07ModulationConfig(
                band_to_I100=conversion,
                amplitude_correction=self.md07_correction,
                scale_indices=self.config.md07.scale_indices,
                context_factor=self.config.md07.context_factor,
                spacing_factor=self.config.md07.spacing_factor,
                extrapolate=self.config.md07.extrapolate,
            ),
            device=self.device,
        )

    def _build_global_power_target(self):
        if not self.config.power.enabled or self.config.losses.global_power <= 0:
            return None
        setup = self.config.power
        window_centers = (
            (0.5, 0.5),
            (0.3, 0.3),
            (0.3, 0.7),
            (0.7, 0.3),
            (0.7, 0.7),
        )
        edges = torch.logspace(
            np.log10(1.0 / setup.global_minimum_scale_arcmin),
            np.log10(1.0 / setup.maximum_scale_arcmin),
            setup.global_bins + 1,
            device=self.device,
        )
        planck = radial_log_power_physical(
            self.backbone,
            pixel_scale_arcmin=self.data.pixel_scale_arcsec / 60.0,
            frequency_edges=edges,
            window_centers=window_centers,
        )
        exemplar = radial_log_power_physical(
            self.exemplar,
            pixel_scale_arcmin=self.data.pixel_scale_arcsec / 60.0,
            frequency_edges=edges,
            window_centers=window_centers,
        ).mean(dim=0)
        frequencies = torch.sqrt(edges[:-1] * edges[1:])
        lower = 1.0 / self.config.transition.coarse_scale_arcmin
        upper = 1.0 / self.config.transition.fine_scale_arcmin
        phase = ((frequencies - lower) / (upper - lower)).clamp(0.0, 1.0)
        alpha = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        epsilon = torch.finfo(alpha.dtype).tiny
        target = torch.logaddexp(
            planck + torch.log((1.0 - alpha).clamp_min(epsilon)),
            exemplar[None] + torch.log(alpha.clamp_min(epsilon)),
        )
        target = torch.where(alpha[None] <= 0, planck, target)
        target = torch.where(alpha[None] >= 1, exemplar[None], target)
        return edges, target.detach(), window_centers

    def train_seed(self, seed: int, output_directory: str | Path) -> dict[str, float]:
        """Run one explicit training job and save its complete provenance."""

        seed_everything(seed, self.config.runtime.deterministic)
        directory = Path(output_directory)
        if directory.exists() and any(directory.iterdir()) and not self.config.output.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        save_config(self.config, directory / "config.resolved.yaml")
        (directory / "calibration.json").write_text(
            json.dumps(self.md07_fit_metrics, indent=2, sort_keys=True) + "\n"
        )
        model = build_model(self.config, self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        use_amp = self.config.optimizer.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(seed + 1)
        history: list[dict[str, float]] = []
        taper_mask = np.ones(self.data.backbone.shape, dtype=bool)
        taper = int(
            np.ceil(
                self.config.transition.apodization_fraction
                * min(self.data.backbone.shape)
                / 2.0
            )
        )
        if taper > 0:
            taper_mask[:taper] = False
            taper_mask[-taper:] = False
            taper_mask[:, :taper] = False
            taper_mask[:, -taper:] = False
        taper_mask &= self._final_apodization_flat_mask()
        output_origins = {
            size: valid_patch_origins(taper_mask, size, 1.0)
            for size in self.config.swd.patch_sizes
        }
        power_origins = None
        if self.config.power.enabled:
            power_origins = valid_patch_origins(
                taper_mask, self.config.power.patch_size, 1.0
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        for step in range(self.config.optimizer.steps):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                synthesis, projected, _, expanded_synthesis = render_synthesis(
                    model,
                    self.backbone,
                    transition_pixels=self.transition_pixels,
                    transition_width_fraction=self.config.transition.width_fraction,
                    transition_range_pixels=self.transition_range_pixels,
                    apodization_fraction=self.config.transition.apodization_fraction,
                    boundary_padding_pixels=self.config.transition.boundary_padding_pixels,
                    boundary_taper_fraction=self.config.transition.boundary_taper_fraction,
                    complementary_backbone=self.config.transition.complementary_blend,
                    resolution_fwhm_pixels=self.resolution_pixels,
                    coordinate_chunk=self.config.runtime.coordinate_chunk,
                    render_margin_pixels=self.config.transition.render_margin_pixels,
                    final_split_cosine_alpha=(
                        self.config.transition.final_split_cosine_alpha
                    ),
                    final_split_cosine_beta=(
                        self.config.transition.final_split_cosine_beta
                    ),
                    final_apodization_mode=self.config.transition.final_apodization_mode,
                )
            generator = torch.Generator(device=self.device).manual_seed(seed + 1000 + step)
            patch_loss = torch.zeros((), device=self.device)
            if self.config.losses.patch_swd > 0:
                patch_losses = []
                for size in self.config.swd.patch_sizes:
                    chosen = sample_origins(
                        output_origins[size], self.config.swd.patch_batch, rng
                    )
                    features = extract_patch_vectors(synthesis, chosen, size)
                    target, normalizer, _ = self.patch_targets[size]
                    patch_losses.append(
                        sliced_wasserstein_loss(
                            normalizer.apply(features),
                            target,
                            projection_count=self.config.swd.projections,
                            quantile_count=self.config.swd.quantiles,
                            generator=generator,
                            chunk_size=self.config.swd.projection_chunk,
                        )
                    )
                patch_loss = torch.stack(patch_losses).mean()
            morlet_loss = torch.zeros((), device=self.device)
            if self.config.losses.morlet_swd > 0:
                morlet_features = self.morlet_output.features(
                    expanded_synthesis, center_mask=self.morlet_output_mask
                )
                morlet_features = subsample_rows(
                    morlet_features, self.config.swd.morlet_samples, generator
                )
                morlet_loss = sliced_wasserstein_loss(
                    self.morlet_normalizer.apply(morlet_features),
                    self.morlet_target,
                    projection_count=self.config.swd.projections,
                    quantile_count=self.config.swd.quantiles,
                    generator=generator,
                    chunk_size=self.config.swd.projection_chunk,
                )
            md07_loss = torch.zeros((), device=self.device)
            if self.md07 is not None:
                residuals = self.md07(synthesis[None])
                md07_loss = torch_f.huber_loss(
                    residuals,
                    torch.zeros_like(residuals),
                    delta=self.config.md07.huber_delta,
                )
            power_loss = torch.zeros((), device=self.device)
            isotropy_loss = torch.zeros((), device=self.device)
            global_power_loss = torch.zeros((), device=self.device)
            if self.power_target is not None:
                setup = self.config.power
                chosen = sample_origins(power_origins, setup.patch_batch, rng)
                vectors = extract_patch_vectors(synthesis, chosen, setup.patch_size)
                patches = vectors.reshape(-1, setup.patch_size, setup.patch_size)
                pixel_scale_arcmin = self.data.pixel_scale_arcsec / 60.0
                power_values = radial_log_power_features(
                    patches,
                    bins=setup.bins,
                    minimum_frequency=pixel_scale_arcmin / setup.minimum_scale_arcmin,
                    maximum_frequency=pixel_scale_arcmin / setup.maximum_scale_arcmin,
                )
                normalizer, target_center = self.power_target
                delta_power = normalizer.apply(power_values).mean(dim=0) - target_center
                power_loss = torch_f.huber_loss(
                    delta_power,
                    torch.zeros_like(delta_power),
                    delta=setup.huber_delta,
                )
                if self.config.losses.isotropy > 0:
                    residual_vectors = extract_patch_vectors(
                        projected, chosen, setup.patch_size
                    )
                    residual_patches = residual_vectors.reshape(
                        -1, setup.patch_size, setup.patch_size
                    )
                    anisotropy = angular_power_anisotropy(
                        residual_patches,
                        minimum_frequency=(
                            pixel_scale_arcmin
                            / self.config.transition.fine_scale_arcmin
                        ),
                        maximum_frequency=(
                            pixel_scale_arcmin / setup.maximum_scale_arcmin
                        ),
                    )
                    isotropy_loss = anisotropy.square().mean()
            if self.global_power_target is not None:
                global_edges, global_target, global_centers = self.global_power_target
                global_values = radial_log_power_physical(
                    synthesis,
                    pixel_scale_arcmin=self.data.pixel_scale_arcsec / 60.0,
                    frequency_edges=global_edges,
                    window_centers=global_centers,
                )
                # Do not clip large transition-band errors: a narrow bump must
                # remain expensive even when patch/Morlet statistics improve.
                global_power_loss = torch_f.mse_loss(global_values, global_target)
            boundary_loss = torch.zeros((), device=self.device)
            if self.config.losses.boundary > 0:
                boundary_loss = boundary_gradient_excess(projected, width=8)
            ramp = min(1.0, (step + 1) / max(self.config.md07.warmup_steps, 1))
            power_ramp = min(
                1.0, (step + 1) / max(self.config.power.warmup_steps, 1)
            )
            loss = (
                self.config.losses.patch_swd * patch_loss
                + self.config.losses.morlet_swd * morlet_loss
                + self.config.losses.md07 * ramp * md07_loss
                + self.config.losses.power * power_ramp * power_loss
                + self.config.losses.isotropy * power_ramp * isotropy_loss
                + self.config.losses.global_power * power_ramp * global_power_loss
                + self.config.losses.boundary * boundary_loss
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if self.config.optimizer.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "patch_swd": float(patch_loss.detach().cpu()),
                "morlet_swd": float(morlet_loss.detach().cpu()),
                "md07": float(md07_loss.detach().cpu()),
                "power": float(power_loss.detach().cpu()),
                "isotropy": float(isotropy_loss.detach().cpu()),
                "global_power": float(global_power_loss.detach().cpu()),
                "boundary": float(boundary_loss.detach().cpu()),
                "residual_rms": float(projected.detach().square().mean().sqrt().cpu()),
                "step_seconds": float(time.perf_counter() - step_started),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            history.append(row)
            if (step + 1) % self.config.runtime.checkpoint_interval == 0:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                    directory / f"checkpoint_{step + 1:06d}.pt",
                )
            if (step + 1) % self.config.runtime.diagnostic_interval == 0:
                preview_dir = directory / "intermediate"
                preview_dir.mkdir(exist_ok=True)
                np.save(
                    preview_dir / f"synthesis_{step + 1:06d}.npy",
                    synthesis.detach().cpu().numpy().astype(np.float32),
                )
                print(
                    f"seed={seed} step={step + 1}/{self.config.optimizer.steps} "
                    f"loss={row['loss']:.5g} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

        torch.save({"model": model.state_dict(), "step": len(history)}, directory / "final.pt")
        with (directory / "loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        model.eval()
        with torch.no_grad():
            synthesis, projected, raw, _ = render_synthesis(
                model,
                self.backbone,
                transition_pixels=self.transition_pixels,
                transition_width_fraction=self.config.transition.width_fraction,
                transition_range_pixels=self.transition_range_pixels,
                apodization_fraction=self.config.transition.apodization_fraction,
                boundary_padding_pixels=self.config.transition.boundary_padding_pixels,
                boundary_taper_fraction=self.config.transition.boundary_taper_fraction,
                complementary_backbone=self.config.transition.complementary_blend,
                resolution_fwhm_pixels=self.resolution_pixels,
                coordinate_chunk=self.config.runtime.coordinate_chunk,
                render_margin_pixels=self.config.transition.render_margin_pixels,
                final_split_cosine_alpha=self.config.transition.final_split_cosine_alpha,
                final_split_cosine_beta=self.config.transition.final_split_cosine_beta,
                final_apodization_mode=self.config.transition.final_apodization_mode,
            )
        synth_np = synthesis.cpu().numpy()
        np.save(directory / "synthesis_unprojected.npy", synth_np.astype(np.float32))
        synth_np = phase_preserving_power_projection(
            synth_np,
            self.backbone_numpy,
            self.data.exemplar,
            pixel_scale_arcsec=self.data.pixel_scale_arcsec,
            fine_scale_arcmin=self.config.transition.fine_scale_arcmin,
            coarse_scale_arcmin=self.config.transition.coarse_scale_arcmin,
            iterations=self.config.power.projection_iterations,
        )
        residual_np = synth_np - self.backbone_numpy
        metadata = {
            "METHOD": "NFGRID",
            "RANDSEED": int(seed),
            "TRANSJ": self.config.transition.dyadic_index,
            "EFFRES": self.config.inputs.effective_resolution_arcsec,
            "MD07AMP": self.md07_correction,
            "NSTEP": len(history),
            "TRNFINE": self.config.transition.fine_scale_arcmin or 0.0,
            "TRNCOARS": self.config.transition.coarse_scale_arcmin or 0.0,
            "APODFRAC": self.config.transition.apodization_fraction,
            "BNDPAD": self.config.transition.boundary_padding_pixels,
            "BNDTAPER": self.config.transition.boundary_taper_fraction,
            "CMPBLEND": self.config.transition.complementary_blend,
            "RNDMARG": self.config.transition.render_margin_pixels,
            "SCALPHA": self.config.transition.final_split_cosine_alpha,
            "SCBETA": self.config.transition.final_split_cosine_beta,
            "PWRLOSS": self.config.losses.power,
            "ISOTROPY": self.config.losses.isotropy,
            "GLBPWR": self.config.losses.global_power,
            "BNDLOSS": self.config.losses.boundary,
            "PWRITER": self.config.power.projection_iterations,
            "PWRNORM": self.config.power.normalize_backbone_transition,
            "PWRGAIN": self.power_normalization_metrics.get(
                "power_normalization_median_amplitude_gain", 1.0
            ),
        }
        write_synthesis_fits(
            directory / "synthesis.fits",
            synth_np,
            residual_np,
            self.backbone_numpy,
            self.data.output_header,
            metadata=metadata,
            overwrite=self.config.output.overwrite,
        )
        metrics = basic_metrics(
            synth_np,
            self.backbone_numpy,
            residual_np,
            transition_pixels=self.transition_pixels,
            transition_range_pixels=self.transition_range_pixels,
        )
        metrics.update(self.md07_fit_metrics)
        metrics.update(self.power_normalization_metrics)
        metrics["runtime_seconds"] = float(time.perf_counter() - started)
        metrics["mean_step_seconds"] = float(
            np.mean([row["step_seconds"] for row in history])
        )
        metrics["device"] = str(self.device)
        if self.device.type == "cuda":
            metrics["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
        metrics.update(distribution_metrics(synth_np, "synthesis"))
        metrics.update(distribution_metrics(self.data.exemplar, "exemplar"))
        metrics.update(structure_tensor_metrics(synth_np, "synthesis"))
        metrics.update(structure_tensor_metrics(self.data.exemplar, "exemplar"))
        metrics.update(
            {
                f"residual_{key}": value
                for key, value in boundary_metrics(residual_np).items()
            }
        )
        validation_rng = np.random.default_rng(seed + 9001)
        validation_size = max(self.config.swd.patch_sizes)
        holdout_vectors_all = []
        for image, mask in zip(
            self.texture_exemplars, self.data.texture_holdout, strict=True
        ):
            holdout_origins = valid_patch_origins(mask, validation_size, 1.0)
            if len(holdout_origins):
                holdout_vectors_all.append(
                    extract_patch_vectors(image, holdout_origins, validation_size)
                )
        if not holdout_vectors_all:
            raise ValueError("No held-out validation patch is available across texture cutouts")
        holdout_vectors_all = torch.cat(holdout_vectors_all, dim=0)
        synthesis_origins = valid_patch_origins(
            np.ones(synth_np.shape, dtype=bool), validation_size, 1.0
        )
        count = min(256, len(holdout_vectors_all), len(synthesis_origins))
        synthesis_selected = sample_origins(synthesis_origins, count, validation_rng)
        holdout_selected = validation_rng.choice(len(holdout_vectors_all), size=count, replace=False)
        holdout_vectors = holdout_vectors_all[holdout_selected]
        synthesis_vectors = extract_patch_vectors(
            synthesis, synthesis_selected, validation_size
        )
        _, validation_normalizer, _ = self.patch_targets[validation_size]
        standardized_holdout = validation_normalizer.apply(holdout_vectors).detach()
        standardized_synthesis = validation_normalizer.apply(synthesis_vectors).detach()
        validation_generator = torch.Generator(device=self.device).manual_seed(seed + 9002)
        heldout_swd = sliced_wasserstein_loss(
            standardized_synthesis,
            standardized_holdout,
            projection_count=self.config.swd.projections,
            quantile_count=self.config.swd.quantiles,
            generator=validation_generator,
            chunk_size=self.config.swd.projection_chunk,
        )
        metrics["heldout_patch_swd"] = float(heldout_swd.cpu())
        metrics.update(
            nearest_patch_metrics(
                standardized_synthesis.cpu().numpy(),
                standardized_holdout.cpu().numpy(),
            )
        )
        if self.md07 is not None:
            md07_validation = self.md07.validation(synth_np)
            for target in self.md07.targets:
                prefix = f"j{target.j}"
                ratio = (
                    md07_validation[f"{prefix}_output_sigma"]
                    / md07_validation[f"{prefix}_target_sigma"]
                )
                metrics[f"md07_{prefix}_median_ratio"] = float(np.median(ratio))
                metrics[f"md07_{prefix}_log_rms"] = float(
                    np.sqrt(np.mean(np.log(ratio) ** 2))
                )
        spectra = power_metrics(
            synth_np,
            self.backbone_numpy,
            self.data.exemplar,
            self.data.pixel_scale_arcsec,
        )
        write_metrics(directory / "metrics.json", {**metrics, **spectra})
        plot_power_comparison(
            directory / "power_spectra.png",
            spectra,
            transition_arcmin=(
                self.transition_pixels * self.data.pixel_scale_arcsec / 60.0
            ),
            resolution_arcmin=(
                self.config.inputs.effective_resolution_arcsec / 60.0
            ),
            transition_range_arcmin=(
                (
                    self.config.transition.fine_scale_arcmin,
                    self.config.transition.coarse_scale_arcmin,
                )
                if self.config.transition.fine_scale_arcmin is not None
                else None
            ),
        )
        with fits.open(self.config.inputs.products, memmap=True) as hdul:
            radiance = np.asarray(hdul[self.config.inputs.planck_hdu].data, dtype=np.float32)
            full_herschel = np.asarray(
                hdul[self.config.inputs.herschel_full_hdu].data, dtype=np.float32
            )
        crop_size = int(
            round(
                self.config.power.normalization_crop_arcmin
                / (self.data.pixel_scale_arcsec / 60.0)
            )
        )
        crop_size = min(crop_size, *radiance.shape)
        y0 = (radiance.shape[0] - crop_size) // 2
        x0 = (radiance.shape[1] - crop_size) // 2
        region = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
        plot_texture_reference_power_spectra(
            directory / "texture_reference_power_spectra.png",
            self.data.texture_exemplars,
            full_herschel[region],
            self.data.radiance_slope * radiance[region] + self.data.radiance_offset,
            self.data.pixel_scale_arcsec,
        )
        plot_training_history(directory / "loss_history.png", history)
        plot_core_diagnostics(
            directory / "core_diagnostics.png",
            synth_np,
            self.backbone_numpy,
            self.data.exemplar,
            residual_np,
            transition_pixels=self.transition_pixels,
        )
        np.save(directory / "raw_neural_field.npy", raw.cpu().numpy().astype(np.float32))
        plot_neural_field_state(
            directory / "neural_field_diagnostics.png",
            raw.cpu().numpy(),
            residual_np,
            model,
            transition_pixels=self.transition_pixels,
            transition_range_pixels=self.transition_range_pixels,
            apodization_fraction=self.config.transition.apodization_fraction,
        )
        return metrics
