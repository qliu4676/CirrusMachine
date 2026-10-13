"""FIR-conditioned broad fields and independent five-band texture phases."""

from dataclasses import dataclass

import numpy as np
import torch

from .config import DrawSeeds, ModelingConfig
from .operators import gaussian_filter, power_field, resolve_device, to_tensor


def _cg(apply, rhs, precondition, *, rtol=2e-5, maxiter=1500):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    norm = torch.linalg.vector_norm(rhs)
    if norm < 1e-25:
        return x, dict(iterations=0, relative_residual=0.0)
    z = precondition(r)
    p = z.clone()
    rz = torch.dot(r, z)
    for i in range(maxiter):
        ap = apply(p)
        denom = torch.dot(p, ap)
        if denom <= 0 or not torch.isfinite(denom):
            raise RuntimeError("Conditional covariance is not positive definite")
        alpha = rz / denom
        x += alpha * p
        r -= alpha * ap
        if torch.linalg.vector_norm(r) / norm <= rtol:
            break
        z = precondition(r)
        new = torch.dot(r, z)
        p = z + (new / rz) * p
        rz = new
    error = float(torch.linalg.vector_norm(apply(x) - rhs) / norm)
    if error > 3e-5 or i == maxiter - 1:
        raise RuntimeError(f"Conditional covariance solve failed: residual={error}")
    return x, dict(iterations=i + 1, relative_residual=error)


class BroadModel:
    """Restricted stationary Matérn covariance and Matheron conditional draws."""

    def __init__(self, observed, mask, guide, amp, shortamp, covariance, config):
        self.config = config
        self.device = resolve_device(config.device)
        self.obs = to_tensor(observed, dtype=torch.float64, device=self.device)
        self.mask = to_tensor(mask, dtype=torch.bool, device=self.device)
        self.guide = to_tensor(guide, dtype=torch.float64, device=self.device)
        self.shape = self.obs.shape
        n, m = self.shape
        fy = torch.fft.fftfreq(n, dtype=torch.float64, device=self.device)[:, None]
        fx = torch.fft.rfftfreq(m, dtype=torch.float64, device=self.device)[None, :]
        k2 = (2 * np.pi) ** 2 * (fx * fx + fy * fy)
        spec = (3 / config.broad_length_pixels**2 + k2).pow(-2.5)
        spec *= amp / torch.fft.irfft2(spec, s=self.shape)[0, 0]
        spec[0, 0] += 4 * amp * n * m
        self.broad = spec
        short = torch.exp(-0.5 * config.split_sigma_pixels**2 * k2)
        short *= shortamp / torch.fft.irfft2(short, s=self.shape)[0, 0].clamp_min(1e-30)
        kernel = torch.zeros(self.shape, dtype=torch.float64, device=self.device)
        cov = to_tensor(covariance, dtype=torch.float64, device=self.device)
        size = cov.shape[0]
        kernel[:size, :size] = cov
        kernel = torch.roll(kernel, (-(size // 2), -(size // 2)), (0, 1))
        power = torch.fft.rfft2(kernel).real
        negative = float(
            (-power).clamp_min(0).sum() / power.abs().sum().clamp_min(1e-30)
        )
        if negative > 1e-6:
            raise ValueError("Instrumental covariance is not positive semidefinite")
        self.nuisance = short + power.clamp_min(0) + 1e-6
        self.total = self.broad + self.nuisance
        self.solves = []

    def apply(self, spectrum, image):
        return torch.fft.irfft2(spectrum * torch.fft.rfft2(image), s=self.shape)

    def embed(self, values):
        full = torch.zeros(self.shape, dtype=torch.float64, device=self.device)
        full[self.mask] = values
        return full

    def solve(self, rhs):
        answer, record = _cg(
            lambda v: self.apply(self.total, self.embed(v))[self.mask],
            rhs,
            lambda v: self.apply(1 / self.total, self.embed(v))[self.mask],
        )
        self.solves.append(record)
        return answer

    def mean(self):
        a = self.solve((self.obs - self.guide)[self.mask])
        return self.guide + self.apply(self.broad, self.embed(a))

    def draw(self, mean, seed):
        rng = np.random.default_rng(seed)
        u = self.apply(
            self.broad.sqrt(),
            to_tensor(rng.standard_normal(self.shape), device=self.device),
        )
        e = self.apply(
            self.nuisance.sqrt(),
            to_tensor(rng.standard_normal(self.shape), device=self.device),
        )
        a = self.solve((u + e)[self.mask])
        return mean + u - self.apply(self.broad, self.embed(a))


class TextureGenerator:
    """Five calibrated frequency bands; independent, delivered-VIS fields."""

    def __init__(self, model, psf, config):
        self.config = config
        self.device = resolve_device(config.device)
        self.d = {k: to_tensor(v, device=self.device) for k, v in model.items()}
        self.psf = psf
        self.n = model["mean"].shape[0]
        fy = torch.fft.fftfreq(self.n, dtype=torch.float64, device=self.device)[:, None]
        fx = torch.fft.rfftfreq(self.n, dtype=torch.float64, device=self.device)[
            None, :
        ]
        rad = torch.hypot(fy, fx)
        self.phi = torch.atan2(fy, fx)
        centers = self.d["band_centers"][:, None, None]
        self.bands = torch.exp(
            -0.5 * (torch.log2(rad.clamp_min(1 / self.n)[None] / centers) / 0.65) ** 2
        )
        self.bands /= self.bands.sum(0, keepdim=True)
        self.hp = 1 - torch.exp(
            -2
            * np.pi**2
            * (config.texture_highpass_fwhm / config.pixel_scale / 2.35482) ** 2
            * rad**2
        )

    def texture(self, seed, adapted=True):
        z = power_field(
            (self.n, self.n),
            float(self.d["beta"]),
            seed,
            self.psf,
            coherence=0,
            device=self.device,
        )
        ft = torch.fft.rfft2(z) * self.hp
        if not adapted:
            return torch.fft.irfft2(ft, s=z.shape)
        out = torch.zeros_like(z)
        for j in range(5):
            band = ft * self.bands[j]
            iso = torch.fft.irfft2(band, s=z.shape)
            idx = (self.d["theta"][j] % np.pi) / np.pi * 8
            lo, frac = idx.floor().long() % 8, idx - idx.floor()
            part = torch.zeros_like(out)
            for k in range(8):
                f = torch.fft.irfft2(
                    band
                    * (1 + 0.95 * torch.cos(2 * (self.phi - k * np.pi / 8))).sqrt(),
                    s=z.shape,
                )
                part += f * ((lo == k) * (1 - frac) + ((lo + 1) % 8 == k) * frac)
            strength = self.d["strength"][j] / 0.95
            out += iso * (1 - strength) + part * strength
        return out

    def field(self, seed):
        return (
            self.d["mean"]
            + self.texture(seed) * self.d["variance_correction"] * self.d["ampmap"]
        ).float()


@dataclass
class CirrusPrior:
    """Normalized guide, conditional broad mean, and calibrated texture model."""

    broad: BroadModel
    mean: torch.Tensor
    texture: TextureGenerator
    amplitude: float
    beam: dict
    calibration: object


@dataclass
class InitialRealization:
    """Normalized cirrus canvas and its reproducible independent phases."""

    image: torch.Tensor
    seeds: DrawSeeds


def build_cirrus_prior(
    field, beam: dict, calibration, config: ModelingConfig
) -> CirrusPrior:
    """Condition a broad cirrus field on the retained observations and FIR guide.

    ``beam`` contains calibrated native FIR targets and their full covariance
    Cholesky factor. Its beam must match config; stale 18 arcsec products are
    rejected for a 36 arcsec run. No injected truth is used.
    """
    expected = config.beam_fwhm / config.pixel_scale / 2.354820045
    if not np.isclose(float(beam["sigma"]), expected, rtol=1e-7):
        raise ValueError(
            "Beam mismatch: recalibrate FIR targets/covariance at the requested resolution"
        )
    calibration.validate(field, config)
    from .calibration import calibration_signature

    if str(beam.get("optical_signature", "")) != calibration_signature(field, config):
        raise ValueError(
            "FIR calibration is missing or stale for these observations and masks"
        )
    d = field.normalized(config)
    device = resolve_device(config.device)
    obs = to_tensor(d["observed"], dtype=torch.float64, device=device)
    guide = to_tensor(beam["backbone"], dtype=torch.float64, device=device)
    mask = to_tensor(d["donor"] & d["roi"], device=device)
    sigma = config.split_sigma_pixels
    cover = gaussian_filter(mask.double(), sigma)
    use = mask & (cover > 0.5)
    if not use.any():
        raise ValueError("Insufficient retained support for broad-field calibration")
    sm = gaussian_filter(torch.where(mask, obs - guide, 0), sigma) / cover.clamp_min(
        1e-9
    )
    amp = max(float(sm[use].var(correction=0)), 1e-4)
    sv = gaussian_filter(torch.where(mask, obs, 0), sigma) / cover.clamp_min(1e-9)
    cov = to_tensor(calibration.noise["covariance"], dtype=torch.float64, device=device)
    nc = len(cov) // 2
    nh = (
        cov[nc, nc]
        - 2 * gaussian_filter(cov, sigma, "constant")[nc, nc]
        + gaussian_filter(cov, sigma * np.sqrt(2), "constant")[nc, nc]
    )
    short = max(float(((obs[use] - sv[use]) ** 2).mean() - nh), 0)
    model = BroadModel(obs, mask, guide, amp, short, cov, config)
    return CirrusPrior(
        model,
        model.mean(),
        TextureGenerator(calibration.texture, field.psf, config),
        amp,
        beam,
        calibration,
    )


def initialize_cirrus(
    prior: CirrusPrior, config: ModelingConfig, *, seeds=None
) -> list[InitialRealization]:
    """Draw independently conditioned broad structure plus high-pass texture."""
    result = []
    for pair in config.draws if seeds is None else seeds:
        draw = prior.broad.draw(prior.mean, pair.broad)
        texture = (
            prior.texture.field(pair.texture).double()
            - prior.texture.d["mean"].double()
        )
        texture -= gaussian_filter(texture, config.split_sigma_pixels)
        guide = prior.broad.guide
        initial = (
            guide + gaussian_filter(draw - guide, config.split_sigma_pixels) + texture
        )
        result.append(InitialRealization(initial.float(), pair))
    return result
