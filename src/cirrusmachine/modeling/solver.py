"""Alternating cirrus optimization and detached observed halo/sky estimation."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelingConfig
from .diffuse import fit_diffuse
from .logging import ModelingProgress
from .operators import MaskedCovariance, as_numpy, beam_loss, resolve_device, to_tensor
from .priors import initialize_cirrus
from .statistics import Statistics, independent_targets

logger = logging.getLogger(__name__)


@dataclass
class ModelingResult:
    """One draw's cirrus, corrected image and conditional halo, in input image units.

    Maps cover the scientific crop. ``corrected`` is observed minus cirrus;
    sky is reported separately. ``normalized`` retains the full FFT canvas.
    The halo is annulus-relative and describes the delivered, PSF-blurred image.
    """

    cirrus: np.ndarray
    corrected: np.ndarray
    halo: np.ndarray
    sky: float
    normalized: np.ndarray
    seeds: object
    history: list
    diagnostics: dict


def _halo(
    field, config, data, image, previous=None, *, initial=False, diffuse_model=None
):
    residual = (data["science"][field.crop] - as_numpy(image)[field.crop]) * data["std"]
    if initial:
        # The reference initialization subtracts separately converted ADU maps.
        # Preserve this float32 operation order before nonlinear geometry fitting.
        observed = data["mu"] + data["std"] * data["science"][field.crop]
        cirrus = data["mu"] + data["std"] * as_numpy(image)[field.crop]
        residual = observed - cirrus
    model = fit_diffuse if diffuse_model is None else diffuse_model
    if not callable(model):
        raise TypeError("diffuse_model must be callable")
    proposal, info = model(
        residual,
        field.source_clean[field.crop],
        field.center_xy,
        field.radius[field.crop],
        config.sky_annulus,
        previous,
        pixel_scale=config.pixel_scale,
    )
    proposal = as_numpy(proposal)
    if (
        proposal.shape != residual.shape
        or not np.isrealobj(proposal)
        or not np.isfinite(proposal).all()
        or (proposal < 0).any()
    ):
        raise ValueError(
            "Diffuse model must return a finite, nonnegative real image matching the crop"
        )
    try:
        sky = as_numpy(info["sky"]["value_ADU"])
        valid_sky = sky.ndim == 0 and np.isrealobj(sky) and np.isfinite(sky)
    except (KeyError, TypeError, IndexError):
        valid_sky = False
    if not isinstance(info, dict) or not valid_sky:
        raise ValueError(
            "Diffuse model info must contain a finite scalar info['sky']['value_ADU']"
        )
    # Keep the saved metadata and downstream arithmetic independent of tensor devices.
    info = {**info, "sky": {**info["sky"], "value_ADU": float(sky)}}
    full = np.zeros(field.image.shape, dtype="f4")
    full[field.crop] = proposal / data["std"]
    return full, info


def prepare_likelihood_variance(field, prior, config, *, diffuse_model=None):
    """Freeze one observed-exterior discrepancy allowance for all requested draws.

    Uses the designated allowance seeds (historical draw0), independently of
    the output draw list. No optimization or truth is needed to construct it.
    ``diffuse_model`` follows the same interface as in
    :func:`run_cirrus_halo_modeling`; use the same model for both calls.
    """
    initial = initialize_cirrus(prior, config, seeds=(config.allowance_seeds,))[0]
    d = field.normalized(config)
    halo, info = _halo(
        field, config, d, initial.image, initial=True, diffuse_model=diffuse_model
    )
    exterior = d["donor"] & d["roi"]
    residual = (
        d["science"]
        - as_numpy(initial.image)
        - halo
        - info["sky"]["value_ADU"] / d["std"]
    )[exterior]
    sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    cov = prior.calibration.noise["covariance"]
    v = float(cov[len(cov) // 2, len(cov) // 2])
    allowance = max(sigma * sigma - v, v)
    extra = np.zeros(field.image.shape)
    mask = d["interior_mask"]
    extra[mask] = (v + allowance) / np.maximum(d["taper"][mask], 1e-8) - v
    return extra


def run_cirrus_halo_modeling(
    field,
    prior,
    initializations,
    config: ModelingConfig,
    *,
    diffuse_model: Callable | None = None,
    extra_variance=None,
    progress=True,
) -> list[ModelingResult]:
    """Optimize each supplied draw with fixed masks and alternating halo/sky updates.

    The six weights apply to S1/S2, P00, coarse, CDF, orientation and optical
    likelihood. ``target_refresh_steps`` and ``halo_refresh_steps`` control
    the target and halo/sky update intervals.
    Use ``progress=False`` to hide progress bar.
    ``extra_variance`` may supply a validated, frozen shared allowance in
    normalized intensity squared; otherwise it is derived from allowance_seeds.

    ``diffuse_model`` optionally replaces :func:`fit_diffuse` in allowance
    preparation, initialization, every refresh and the final update. Its call is
    ``model(residual, valid, center, measurement_radius, annulus, previous,
    pixel_scale=...)``. Image, mask and radius inputs are NumPy arrays on the
    scientific crop; ``valid`` is True for usable pixels and ``center`` is
    crop-relative (x, y) in pixels.
    Radii and annulus limits are in arcsec, and pixel scale is arcsec/pixel.
    Residuals are observed minus cirrus, in input image units, with sky retained.
    Exclude invalid pixels from fitting and do not modify the supplied arrays.

    Return ``(model_image, info)``: a finite nonnegative image covering the crop
    (including masked pixels), with sky excluded and in input image units, plus
    a dict containing finite scalar ``info['sky']['value_ADU']`` in those same
    units. Model the delivered, PSF-blurred image; no extra PSF convolution is
    applied by the solver. NumPy images and PyTorch tensors are accepted;
    tensors are detached and transferred to CPU. Remaining entries are
    model-specific diagnostics passed back as ``previous`` at the next update;
    ``previous=None`` starts each draw and the separate allowance
    fit. Keep diagnostics serializable if saving results. Custom models own
    their profile refinement: the default isophote-specific ``inner_refit`` is
    applied only when ``diffuse_model`` is omitted or is ``fit_diffuse`` itself.
    """
    if diffuse_model is not None and not callable(diffuse_model):
        raise TypeError("diffuse_model must be callable")
    prior.calibration.validate(field, config)
    if not initializations:
        raise ValueError("At least one initial realization is required")
    if not np.isclose(
        float(prior.beam["sigma"]), config.beam_FWHM / config.pixel_scale / 2.354820045
    ):
        raise ValueError("coarse beam mismatch; rebuild the prior")
    device = resolve_device(config.device)
    started = time.perf_counter()
    logger.info(
        "Preparing %d draw(s) on %s: %d steps each, %.1f arcsec backbone likelihood",
        len(initializations),
        device,
        config.steps,
        config.beam_FWHM,
    )
    d = field.normalized(config)
    nd, targets, beam = prior.calibration.noise, prior.calibration.targets, prior.beam
    if extra_variance is None:
        extra_variance = prepare_likelihood_variance(
            field, prior, config, diffuse_model=diffuse_model
        )
    extra_variance = np.asarray(extra_variance)
    if (
        extra_variance.shape != field.image.shape
        or not np.isfinite(extra_variance).all()
        or (extra_variance < 0).any()
    ):
        raise ValueError(
            "Shared variance must be finite, nonnegative and match the canvas"
        )
    m = Statistics(d["donor"], d["roi"], targets["classes"], config)
    like = MaskedCovariance(
        d["likelihood_mask"], nd["covariance"], extra_variance, device=device
    )

    def tensor(value, dtype=None):
        return to_tensor(value, dtype=dtype, device=device)

    obs = tensor(d["science"])
    factors = [tensor(targets[f"factor{j}"]) for j in range(5)]
    offsets = [tensor(targets[f"mask_offset{j}"]) for j in range(5)]
    sigmas = [tensor(targets[f"mask_sigma{j}"]) for j in range(5)]
    grid, bt = (
        tensor(beam["sample_grid"], torch.float32),
        tensor(beam["target"], torch.float32),
    )
    chol = tensor(beam["chol"], torch.float64)
    guide = tensor(beam["backbone"], torch.float32)
    n = field.image.shape[0]
    frequency = torch.fft.fftfreq(n, device=device)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    k2 = (2 * np.pi) ** 2 * (kx * kx + ky * ky)
    shape = (3 / config.broad_length_pixels**2 + k2).pow(-2.5)
    spectrum = prior.amplitude * shape / shape.mean()
    spectrum[0, 0] += 4 * prior.amplitude * n * n
    lowpass = torch.exp(-(config.prior_lowpass_pixels**2) * k2)
    precision, effective = lowpass / spectrum.clamp_min(1e-12), float(lowpass.sum())
    angles = []
    for filt in m.filters[0]:
        a = as_numpy(filt)
        iy, ix = np.unravel_index(a.argmax(), a.shape)
        freq = np.fft.fftfreq(a.shape[0])
        angles.append(np.arctan2(freq[iy], freq[ix]))
    phase = tensor(np.exp(2j * np.array(angles)), torch.complex64)[None, :, None]
    weights = tensor(config.weights, torch.float32)
    stage_ends = np.cumsum(config.stage_steps)
    results = []
    for index, initial in enumerate(initializations):
        if (
            tuple(initial.image.shape) != field.image.shape
            or not torch.isfinite(initial.image).all()
        ):
            raise ValueError("Initial realization must be finite and match the canvas")
        c = torch.nn.Parameter(initial.image.detach().to(device).clone())
        opt = torch.optim.Adam([c], lr=config.learning_rates[0])
        halo, halo_info = _halo(
            field, config, d, c, initial=True, diffuse_model=diffuse_model
        )
        sky = halo_info["sky"]["value_ADU"] / d["std"]
        H = tensor(halo)
        halo_history = [dict(step=0, **halo_info)]
        history, solves, target_history = [], [], []
        target = [tensor(targets[f"observed{j}"]) for j in range(5)]
        uncertainty = [tensor(targets[f"uncertainty{j}"]) for j in range(5)]

        def chosen(j, target, uncertainty):
            a, u = target[j], uncertainty[j]
            if j > 1:
                return a, u
            small = torch.zeros(a.shape[0], dtype=torch.bool, device=device)
            if j == 0:
                small[:2] = True
            else:
                for k, pair in enumerate(m.pairs):
                    small[k] = pair == (0, 1)
            small = small.reshape((-1,) + (1,) * (a.ndim - 1))
            sn = a.clamp_min(0).square()
            alpha = torch.where(
                small, sn / (sn + u.square() + 1e-12), torch.ones_like(a)
            )
            ref, sd = tensor(targets[f"M1_{j}"]), tensor(targets[f"M1_std{j}"])
            return alpha * a + (1 - alpha) * ref, (
                (alpha * u) ** 2 + ((1 - alpha) * sd) ** 2
            ).sqrt()

        logger.debug("Draw %d seeds: %s", index + 1, initial.seeds)
        with ModelingProgress(
            config.steps, index, len(initializations), progress
        ) as status:
            for step in range(config.steps):
                if step % config.target_refresh_steps == 0:
                    target, uncertainty = independent_targets(
                        m,
                        obs - H - sky,
                        c.detach(),
                        nd["kernels"],
                        nd["variances"],
                        config.noise_seeds,
                        factors,
                        offsets,
                        sigmas,
                    )
                    target_history.append(
                        dict(
                            step=step,
                            target=[as_numpy(t) for t in target],
                            uncertainty=[as_numpy(t) for t in uncertainty],
                        )
                    )
                lr = config.learning_rates[
                    min(
                        int(np.searchsorted(stage_ends, step, side="right")),
                        len(config.learning_rates) - 1,
                    )
                ]
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad()
                stats, parts = m.full(c), []
                for j in range(3):
                    a, u = chosen(j, target, uncertainty)
                    dims = tuple(range(1, a.ndim))
                    den = a.square().mean(dims, keepdim=True).clamp_min(
                        1e-6
                    ) + u.square().mean(dims, keepdim=True)
                    reliable = (a > u).float()
                    parts.append(
                        (((stats[j] - a) ** 2 / den) * reliable).sum()
                        / reliable.sum().clamp_min(1)
                    )
                beam_term = beam_loss(c, grid, bt, chol, float(beam["sigma"]))
                q, qt = m.CDF(c), target[3].sort(dim=-1).values
                CDF = (
                    (q - qt) ** 2
                    / (
                        qt.square().mean(1, keepdim=True).clamp_min(0.01)
                        + uncertainty[3] ** 2
                    )
                ).mean()
                e, et, eu = m.angular_energy(c), target[4].clamp_min(0), uncertainty[4]
                am = (et * phase).sum(1).abs()
                confidence = am**2 / (am**2 + eu.square().sum(1) + 1e-10)
                orientation = (
                    (
                        (
                            (
                                e / e.sum(1, keepdim=True).clamp_min(1e-6)
                                - et / et.sum(1, keepdim=True).clamp_min(1e-6)
                            )
                            * 8
                        )
                        ** 2
                    ).mean(1)
                    * confidence
                ).sum() / confidence.sum().clamp_min(1)
                optical = (
                    like.quadratic(obs - c - H - sky)
                    * float(d["likelihood_mask"].sum())
                    / d["original_normalizer"]
                )
                terms = torch.stack(
                    [
                        (parts[0] + parts[1]) / 2,
                        parts[2],
                        beam_term,
                        CDF,
                        orientation,
                        optical,
                    ]
                )
                penalty = (
                    config.prior_weight
                    * (
                        precision
                        * torch.fft.fft2(c - guide, norm="ortho").abs().square()
                    ).sum()
                    / effective
                )
                loss = (terms * weights).sum() + penalty
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite objective at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([c], 10)
                opt.step()
                history.append(
                    [
                        step + 1,
                        float(loss.detach()),
                        *terms.detach().tolist(),
                        float(penalty.detach()),
                    ]
                )
                solves.append(dict(like.last))
                if (step + 1) % config.halo_refresh_steps == 0:
                    fresh, halo_info = _halo(
                        field, config, d, c, halo_info, diffuse_model=diffuse_model
                    )
                    halo = (
                        1 - config.halo_damping
                    ) * halo + config.halo_damping * fresh
                    sky = halo_info["sky"]["value_ADU"] / d["std"]
                    H = tensor(halo)
                    halo_history.append(dict(step=step + 1, **halo_info))
                status.update(step + 1, history[-1][1], lr)
        halo, halo_info = _halo(
            field, config, d, c, halo_info, diffuse_model=diffuse_model
        )
        halo_history.append(dict(step=config.steps, final_fresh=True, **halo_info))
        sky = float(halo_info["sky"]["value_ADU"])
        normalized = as_numpy(c).copy()
        cirrus = (d["mu"] + d["std"] * normalized)[field.crop]
        halo_image = halo[field.crop] * d["std"]
        inner = None
        if config.inner_refit and (
            diffuse_model is None or diffuse_model is fit_diffuse
        ):
            from .diffuse import refit_inner_diffuse

            halo_image, inner = refit_inner_diffuse(
                field.image[field.crop] - cirrus - sky,
                halo_image,
                field.source_clean[field.crop],
                field.center_xy,
                halo_info,
                nd["covariance"] * d["std"] ** 2,
                config,
            )
        results.append(
            ModelingResult(
                cirrus,
                field.image[field.crop] - cirrus,
                halo_image,
                sky,
                normalized,
                initial.seeds,
                history,
                dict(
                    halo=halo_info,
                    halo_history=halo_history,
                    covariance_solves=solves,
                    target_history=target_history,
                    inner_refit=inner,
                    device=str(device),
                ),
            )
        )
    logger.info(
        "Completed %d draw(s) in %.1f s; final loss %.5g, sky %.5g ADU",
        len(results),
        time.perf_counter() - started,
        history[-1][1],
        sky,
    )
    return results


def summarize_models(results):
    """Return ensemble mean/std and per-draw maps; one draw has undefined scatter."""
    if not results:
        raise ValueError("No results to summarize")
    images = np.stack([r.cirrus for r in results])
    return dict(
        cirrus=images,
        mean=images.mean(0),
        std=images.std(0, ddof=1)
        if len(results) > 1
        else np.full(images.shape[1:], np.nan),
    )
