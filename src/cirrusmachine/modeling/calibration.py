"""Calibration extracted from retained-pixel reference and five-band preparation.

Bootstrap targets are used only to calibrate angular texture guides. The final
and online targets use independent masked noise corrections, never candidate
statistics inside hidden pixels. See docs/modeling.md for the source lineage.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from . import operators
from .config import ModelingConfig
from .operators import as_numpy, covariance_kernel, noise_bases, to_tensor
from .priors import TextureGenerator
from .statistics import Statistics, independent_targets


class NumpyOperators:
    def __init__(self, PSF, config):
        self.PSF = PSF
        self.config = config
        self.device = operators.resolve_device(config.device)

    def tensor(self, x, dtype=None):
        return to_tensor(x, device=self.device, dtype=dtype)

    def gaussian_filter(self, x, sigma, mode="reflect"):
        return as_numpy(operators.gaussian_filter(self.tensor(x), sigma, mode))

    def smooth_observed(self, x, mask, sigma):
        return tuple(
            (
                as_numpy(z)
                for z in operators.smooth_observed(
                    self.tensor(x), self.tensor(mask), sigma
                )
            )
        )

    def draw_noise(self, shape, kernels, variances, seed):
        return as_numpy(
            operators.draw_noise(shape, kernels, variances, seed, device=self.device)
        )

    def power_field(self, shape, beta, seed, theta=0.0, coherence=0.4):
        return as_numpy(
            operators.power_field(
                shape, beta, seed, self.PSF, theta, coherence, device=self.device
            )
        )

    def PSF_transfer(self, shape):
        return as_numpy(operators.PSF_transfer(shape, self.PSF, device=self.device))

    def fft2(self, x):
        return as_numpy(torch.fft.fft2(self.tensor(x)))

    def rfft2(self, x):
        return as_numpy(torch.fft.rfft2(self.tensor(x)))

    def irfft2(self, x, s):
        return as_numpy(torch.fft.irfft2(self.tensor(x), s=s))

    def generator(self, model):
        outer = self

        class Generator:
            def texture(self, seed, adapted=True):
                return as_numpy(
                    TextureGenerator(model, outer.PSF, outer.config).texture(
                        seed, adapted
                    )
                )

            def field(self, seed):
                return as_numpy(
                    TextureGenerator(model, outer.PSF, outer.config).field(seed)
                )

        return Generator()


def interpolate(z, n):
    coord = np.linspace(10, n - 11, 33)
    yy, xx = np.indices((n, n))
    points = np.column_stack(
        [
            np.clip(yy.ravel(), coord[0], coord[-1]),
            np.clip(xx.ravel(), coord[0], coord[-1]),
        ]
    )
    return RegularGridInterpolator(
        (coord, coord), z.reshape(33, 33), bounds_error=True
    )(points).reshape(n, n)


@torch.no_grad()
def _bootstrap_targets(
    measure, observed, proxy, kernels, variances, seeds, mask_factors
):
    raw = measure.all(observed, True)
    clean = measure.all(proxy, False)
    sims = [
        measure.all(
            proxy
            + operators.draw_noise(
                proxy.shape, kernels, variances, s, device=proxy.device
            ).to(proxy.dtype),
            True,
        )
        for s in seeds
    ]
    targets = []
    uncertainty = []
    for j, (r, t, f) in enumerate(zip(raw, clean, mask_factors)):
        samples = torch.stack([z[j] * f for z in sims])
        targets.append(r * f - (samples.mean(0) - t))
        uncertainty.append(samples.std(0))
    return (targets, uncertainty)


def calibrate_spectrum(d, PSF, config, crop):
    ops = NumpyOperators(PSF, config)
    smooth_observed = ops.smooth_observed
    draw_noise = ops.draw_noise
    power_field = ops.power_field
    s = crop
    x = d["observed"][s]
    mask = d["donor"][s]
    shape = x.shape
    smooth, cov = smooth_observed(x, mask, 20)
    res = x - smooth
    yy, xx = np.indices(shape)
    block = (xx // 96 + 2 * (yy // 96)) % 4
    train = mask & (block != 0)
    test = mask & (block == 0)
    freq = np.fft.fftfreq(shape[0])
    fy = freq[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    rad = np.hypot(fy, fx)
    phi = np.arctan2(fy, fx)
    edges = np.geomspace(0.025, 0.45, 19)
    ids = np.digitize(rad, edges) - 1
    window = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
    angular = np.abs(np.cos(phi)) > np.abs(np.sin(phi))

    def spec(z, m):
        w = m * window
        if not np.any(m):
            raise ValueError("A spectral calibration block has no retained support")
        mean = (z * w).sum() / w.sum()
        power = abs(ops.fft2((z - mean) * w)) ** 2 / (w * w).sum()
        return np.array(
            [
                power[(ids == i) & (angular == a)].mean()
                for a in [False, True]
                for i in range(18)
            ]
        )

    if any(
        not np.any((ids == i) & (angular == a))
        for a in (False, True)
        for i in range(18)
    ):
        raise ValueError(
            "Scientific crop is too small for the calibrated spectral bins"
        )
    target = spec(res, train)
    kernels = noise_bases()
    betas = np.array(config.beta_grid)
    fields = {
        b: [power_field(d["observed"].shape, b, 830000 + i)[s] for i in range(4)]
        for b in betas
    }
    noises = [
        draw_noise(d["observed"].shape, kernels, np.eye(5)[j], 840000 + i)[s]
        for j in range(5)
        for i in range(4)
    ]

    def templates(m):
        return (
            np.array([np.mean([spec(z, m) for z in fields[b]], axis=0) for b in betas]),
            np.array(
                [
                    np.mean([spec(z, m) for z in noises[j * 4 : j * 4 + 4]], axis=0)
                    for j in range(5)
                ]
            ),
        )

    def fit(m):
        st, nt = templates(m)
        tar = spec(res, m)
        records = []
        for beta, signal in zip(betas, st):
            design = np.concatenate([signal[None], nt], axis=0)
            z = least_squares(
                lambda v, design=design: (
                    np.log(np.exp(v) @ design + 1e-15) - np.log(tar + 1e-15)
                ),
                np.log([0.5, 0.08, 0.03, 0.03, 0.02, 0.02]),
                bounds=(-16, 4),
                loss="soft_l1",
                f_scale=0.2,
            )
            records.append((np.mean(z.fun**2), beta, np.exp(z.x), np.exp(z.x) @ design))
        return (min(records, key=lambda t: t[0]), records, st, nt)

    best, records, st, nt = fit(train)
    err, beta, coef, pred = best
    variances = coef[1:].copy()
    floor = 0.02 * variances.sum()
    variances[0] = max(variances[0], floor)
    sensitivity = []
    for k in range(4):
        b, _, _, _ = fit(mask & (block != k))
        sensitivity.append(
            {
                "leave_block_class": k,
                "beta": b[1],
                "noise_variance": b[2][1:].sum(),
                "log_power_rmse": np.sqrt(b[0]),
            }
        )
    kernel = covariance_kernel(kernels, variances)
    stt, ntt = templates(test)
    prediction_test = coef[0] * stt[list(betas).index(beta)] + coef[1:] @ ntt
    report = {
        "status": "model_dependent_single_image_estimate",
        "beta": beta,
        "signal_variance_normalized": coef[0],
        "noise_variances_normalized": variances,
        "noise_sigma_ADU": np.sqrt(variances.sum()) * float(d["std"]),
        "train_log_power_rmse": np.sqrt(err),
        "test_log_power_rmse": np.sqrt(
            np.mean((np.log(prediction_test) - np.log(spec(res, test))) ** 2)
        ),
        "beta_grid_scores": [
            {"beta": b, "log_power_rmse": np.sqrt(e), "noise_variance": c[1:].sum()}
            for e, b, c, _ in records
        ],
        "blocked_fits": sensitivity,
        "white_variance_floor": floor,
        "limitations": [
            "No independent VIS exposure differences available in inspected inputs.",
            "Power-law/compact-noise decomposition is not unique.",
            "Stationary covariance must be bracketed; unresolved sources and residual stripes can contaminate it.",
        ],
        "S1_S2_used_for_calibration": False,
    }
    values = dict(
        kernels=kernels,
        variances=variances,
        covariance=kernel,
        frequency=np.sqrt(edges[:-1] * edges[1:]),
        train_power=target,
        predicted_train=pred,
        test_power=spec(res, test),
        predicted_test=prediction_test,
        beta=beta,
        signal_variance=coef[0],
    )
    return (values, report)


def bootstrap_references(d, nd, PSF, config):
    ops = NumpyOperators(PSF, config)
    smooth_observed = ops.smooth_observed
    gaussian_filter = ops.gaussian_filter
    draw_noise = ops.draw_noise
    power_field = ops.power_field
    tensor = ops.tensor
    n = d["observed"].shape[0]
    roi = d["roi"]
    donor = d["donor"]
    x = d["observed"]
    back = gaussian_filter(d["initial"], 2)
    edges = np.quantile(back[roi & donor], [1 / 3, 2 / 3])
    classes = np.digitize(back, edges)
    m = Statistics(donor, roi, classes, config)
    sm, cov = smooth_observed(x, donor, 4)
    gy, gx = np.gradient(sm)
    support = donor & (cov > 0.55)

    def structure_tensor(v):
        sm, _ = smooth_observed(v, donor, 4)
        dy, dx = np.gradient(sm)
        return np.array(
            [smooth_observed(a, support, 24)[0] for a in [dx * dx, dy * dy, dx * dy]]
        )

    observed_tensor = structure_tensor(x)
    nt = np.array(
        [
            structure_tensor(
                draw_noise((n, n), nd["kernels"], nd["variances"], 959000 + i)
            )
            for i in range(8)
        ]
    )
    signal = observed_tensor - nt.mean(0)
    energy = np.maximum(signal[0] + signal[1], 0)
    sigma = (nt[:, 0] + nt[:, 1]).std(0)
    snr = energy / np.maximum(sigma, 1e-08)
    tcov = gaussian_filter(support.astype(float), 24)
    reliability = np.clip((snr - 1) / 3, 0, 1) * np.clip(tcov / 0.25, 0, 1)
    q = (signal[0] - signal[1] + 2j * signal[2]) / np.maximum(energy, 1e-08)
    q = q / np.maximum(abs(q), 1)
    eligible = donor & roi & (tcov > 0.25)
    qglobal = np.sum(q[eligible] * energy[eligible]) / np.sum(energy[eligible])
    q = reliability * q + (1 - reliability) * qglobal
    theta = np.angle(q) / 2
    coherence = np.clip(2 * abs(q), 0, 0.8)
    beta = float(nd["beta"])
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    rad = np.hypot(fy, fx)
    phi = np.arctan2(fy, fx)
    hp = 1 - np.exp(
        -2
        * np.pi**2
        * (config.texture_highpass_FWHM / config.pixel_scale / 2.35482) ** 2
        * rad**2
    )
    lo = np.floor(theta % np.pi / np.pi * 8).astype(int)
    frac = theta % np.pi / np.pi * 8 - lo

    def generate(seed, bb=beta):
        z = power_field((n, n), bb, seed, coherence=0)
        ft = ops.rfft2(z) * hp
        iso = ops.irfft2(ft, s=(n, n))
        out = np.zeros_like(iso)
        for k in range(8):
            field = ops.irfft2(
                ft * np.sqrt(1 + 0.8 * np.cos(2 * (phi - k * np.pi / 8))), s=(n, n)
            )
            weight = (lo == k) * (1 - frac) + ((lo + 1) % 8 == k) * frac
            out += field * weight
        return out * (coherence / 0.8) + iso * (1 - coherence / 0.8)

    sc, coverage = smooth_observed(x, donor, 3)
    hpobs = x - sc
    noisevar = np.zeros_like(x)
    mockvar = np.zeros_like(x)
    for seed in range(8):
        noise = draw_noise((n, n), nd["kernels"], nd["variances"], 960000 + seed)
        sn, _ = smooth_observed(noise, donor, 3)
        noisevar += (noise - sn) ** 2 / 8
        z = generate(961000 + seed)
        sz, _ = smooth_observed(z, donor, 3)
        mockvar += (z - sz) ** 2 / 8
    amps = []
    centers = []
    for k in range(3):
        sel = roi & donor & (classes == k) & (coverage > 0.55)
        amps.append(
            np.sqrt(
                max(np.mean(hpobs[sel] ** 2 - noisevar[sel]), 0)
                / max(np.mean(mockvar[sel]), 1e-08)
            )
        )
        centers.append(np.median(back[sel]))
    amp = gaussian_filter(np.interp(back, centers, amps), 20)
    full = []
    masked = []
    noisy_stats = []
    first = []
    for index in range(config.reference_count + config.validation_count):
        seed = 970000 + index
        clean = (back + amp * generate(seed)).astype("f4")
        noisy = clean + draw_noise(
            (n, n), nd["kernels"], nd["variances"], 980000 + index
        ).astype("f4")
        with torch.no_grad():
            f = m.all(tensor(clean))
            z = m.all(tensor(clean), True)
            nz = m.all(tensor(noisy), True)
        full.append([v.detach().cpu().numpy() for v in f])
        masked.append([v.detach().cpu().numpy() for v in z])
        noisy_stats.append([v.detach().cpu().numpy() for v in nz])
        if index < 4:
            first.append(clean)
    fs = [np.array([z[j] for z in full]) for j in range(5)]
    ms = [np.array([z[j] for z in masked]) for j in range(5)]
    [np.array([z[j] for z in noisy_stats]) for j in range(5)]
    factors = []
    bias = []
    for j in range(5):
        if j < 3:
            f = fs[j][: config.reference_count].mean(0) / np.maximum(
                ms[j][: config.reference_count].mean(0), 1e-08
            )
        else:
            f = np.ones_like(fs[j][0])
        factors.append(f)
        denom = np.sqrt(np.mean(fs[j][config.reference_count :] ** 2))
        err = np.sqrt(
            np.mean(
                (ms[j][config.reference_count :] * f - fs[j][config.reference_count :])
                ** 2
            )
        ) / max(denom, 1e-08)
        bias.append(float(err))
    proxy = tensor(first[0])
    obs = tensor(x, dtype=torch.float32)
    targets, uncert = _bootstrap_targets(
        m,
        obs,
        proxy,
        nd["kernels"],
        nd["variances"],
        range(990000, 990008),
        [tensor(f) for f in factors],
    )
    mask_sigma = np.std(
        ms[4][: config.reference_count] - fs[4][: config.reference_count],
        axis=0,
        ddof=1,
    )
    uncert[4] = torch.sqrt(uncert[4] ** 2 + tensor(mask_sigma) ** 2)
    return dict(
        smooth_mean=back,
        classes=classes,
        theta=theta,
        coherence=coherence,
        ampmap=amp,
        observed4=as_numpy(targets[4]),
        uncertainty4=as_numpy(uncert[4]),
    )


def refine_texture(d, noise, t, PSF, config):
    ops = NumpyOperators(PSF, config)
    smooth_observed = ops.smooth_observed
    gaussian_filter = ops.gaussian_filter
    draw_noise = ops.draw_noise
    PSF_transfer = ops.PSF_transfer
    m = Statistics(d["donor"], d["roi"], t["classes"], config)
    n = d["observed"].shape[0]
    E = np.maximum(t["observed4"], 0)
    U = t["uncertainty4"]
    angles = []
    for f in m.filters[0]:
        a = f.detach().cpu().numpy()
        iy, ix = np.unravel_index(a.argmax(), a.shape)
        freq = np.fft.fftfreq(a.shape[0])
        angles.append(np.arctan2(freq[iy], freq[ix]))
    angles = np.array(angles)
    phase = np.exp(2j * angles)
    beta = float(noise["beta"])
    theta = []
    strength = []
    reliabilities = []
    responses = []
    centers = []
    PSF = np.abs(PSF_transfer((n, n))) ** 2
    for j in range(5):
        f = m.filters[j].detach().cpu().numpy()
        nn = f.shape[-1]
        freq = np.fft.fftfreq(nn) * nn / n
        ky = freq[:, None]
        kx = freq[None, :]
        r = np.hypot(ky, kx)
        ph = np.arctan2(ky, kx)
        ps = (
            np.maximum(r, 1 / n) ** (-beta)
            * (
                1
                - np.exp(
                    -2
                    * np.pi**2
                    * (config.texture_highpass_FWHM / config.pixel_scale / 2.35482) ** 2
                    * r
                    * r
                )
            )
            ** 2
        )
        coords = np.array(
            [
                np.broadcast_to(ky % 1 * n, (nn, nn)),
                np.broadcast_to(abs(kx) * n, (nn, nn)),
            ]
        )
        ps *= map_coordinates(PSF, coords, order=1, mode="nearest")
        e0 = (abs(f) ** 2 * ps[None]).sum((1, 2))
        e1 = (abs(f) ** 2 * (ps * np.cos(2 * ph))[None]).sum((1, 2))
        q0 = (e0 * phase).sum() / e0.sum()
        q1 = ((e0 + e1) * phase).sum() / (e0 + e1).sum()
        response = float(abs(q1 - q0))
        responses.append(response)
        iy, ix = np.unravel_index(f[0].argmax(), f[0].shape)
        centers.append(float(r[iy, ix]))
        total = E[j].sum(0)
        axial = (E[j] * phase[:, None]).sum(0)
        rel = abs(axial) ** 2 / (abs(axial) ** 2 + (U[j] ** 2).sum(0) + 1e-12)
        xx = np.rint(m.points[:, 0]).astype(int)
        yy = np.rint(m.points[:, 1]).astype(int)
        cov = gaussian_filter(d["donor"].astype(float), 24)[yy, xx]
        roi = d["roi"][yy, xx]
        rel *= np.clip(cov / 0.5, 0, 1)
        q = axial / np.maximum(total, 1e-12)
        weights = rel * total * roi
        globalq = (q * weights).sum() / max(weights.sum(), 1e-12)
        q = rel * q + (1 - rel) * globalq
        qr = interpolate(q.real, n)
        qi = interpolate(q.imag, n)
        qq = qr + 1j * qi
        theta.append(np.angle(qq) / 2)
        strength.append(np.clip(abs(qq) / max(response, 0.0001), 0, 0.95))
        reliabilities.append(rel)
    model = {
        "mean": t["smooth_mean"].astype("f4"),
        "beta": beta,
        "band_centers": np.array(centers),
        "theta": np.array(theta, dtype="f4"),
        "strength": np.array(strength, dtype="f4"),
        "variance_correction": np.ones((n, n), "f4"),
        "ampmap": np.ones((n, n), "f4"),
    }
    g = ops.generator(model)
    v0 = np.zeros((n, n))
    v1 = v0.copy()
    for seed in range(1060000, 1060008):
        v0 += g.texture(seed, False) ** 2 / 8
        v1 += g.texture(seed, True) ** 2 / 8
    correction = np.sqrt(
        gaussian_filter(v0, 32) / np.maximum(gaussian_filter(v1, 32), 1e-12)
    )
    model["variance_correction"] = np.clip(correction, 0.5, 2).astype("f4")
    x = d["observed"]
    donor = d["donor"]
    sc, cov = smooth_observed(x, donor, 3)
    noisevar = np.zeros_like(x)
    mockvar = np.zeros_like(x)
    for i in range(8):
        nn = draw_noise(x.shape, noise["kernels"], noise["variances"], 960000 + i)
        sm, _ = smooth_observed(nn, donor, 3)
        noisevar += (nn - sm) ** 2 / 8
        zz = g.texture(1061000 + i) * model["variance_correction"]
        sm, _ = smooth_observed(zz, donor, 3)
        mockvar += (zz - sm) ** 2 / 8
    amps = []
    bright = []
    for k in range(3):
        use = donor & d["roi"] & (t["classes"] == k) & (cov > 0.55)
        if use.sum() <= 1000:
            raise ValueError(
                "Each brightness class needs more than 1000 supported pixels for amplitude calibration"
            )
        amps.append(
            float(
                np.sqrt(
                    max(np.mean(((x - sc) ** 2 - noisevar)[use]), 0)
                    / max(np.mean(mockvar[use]), 1e-12)
                )
            )
        )
        bright.append(float(np.median(model["mean"][use])))
    model["ampmap"] = gaussian_filter(
        np.interp(model["mean"], bright, amps), 20
    ).astype("f4")
    return model


def calibrate_targets(d, nd, model, classes, PSF, config):
    ops = NumpyOperators(PSF, config)
    tensor = ops.tensor
    m = Statistics(d["donor"], d["roi"], classes, config)
    full = []
    masked = []
    first = None
    g = ops.generator(model)
    for j, seed in enumerate(
        range(1170000, 1170000 + config.reference_count + config.validation_count)
    ):
        z = g.field(seed)
        if first is None:
            first = z
        with torch.no_grad():
            f = m.all(tensor(z))
            q = m.all(tensor(z), True)
        full.append([a.detach().cpu().numpy() for a in f])
        masked.append([a.detach().cpu().numpy() for a in q])
    fs = [np.array([a[j] for a in full]) for j in range(5)]
    ms = [np.array([a[j] for a in masked]) for j in range(5)]
    factors = []
    offsets = []
    sigmas = []
    errors = []
    for j in range(5):
        factor = (
            fs[j][: config.reference_count].mean(0)
            / np.maximum(ms[j][: config.reference_count].mean(0), 1e-08)
            if j < 3
            else np.ones_like(fs[j][0])
        )
        delta = (
            fs[j][: config.reference_count] - factor * ms[j][: config.reference_count]
        )
        offset = delta.mean(0)
        sigma = delta.std(0, ddof=1)
        factors.append(tensor(factor))
        offsets.append(tensor(offset))
        sigmas.append(tensor(sigma))
        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            factor * ms[j][config.reference_count :]
                            + offset
                            - fs[j][config.reference_count :]
                        )
                        ** 2
                    )
                )
                / max(np.sqrt(np.mean(fs[j][config.reference_count :] ** 2)), 1e-08)
            )
        )
    if max(errors[:3]) >= 0.2:
        raise RuntimeError(f"Mask transfer validation failed: {errors}")
    observed = tensor(d["observed"], dtype=torch.float32)
    proxy = tensor(first)
    targets, unc = independent_targets(
        m,
        observed,
        proxy,
        nd["kernels"],
        nd["variances"],
        config.noise_seeds,
        factors,
        offsets,
        sigmas,
    )
    changed = proxy.clone()
    changed[~tensor(d["donor"])] = 1000000.0
    other, _ = independent_targets(
        m,
        observed,
        changed,
        nd["kernels"],
        nd["variances"],
        config.noise_seeds,
        factors,
        offsets,
        sigmas,
    )
    invariance = [float(abs(a - b).max()) for a, b in zip(targets, other)]
    if max(invariance) != 0:
        raise RuntimeError("Hidden pixels affected the fixed WST target")
    values = {
        "classes": classes,
        "M1_0": fs[0][: config.reference_count].mean(0),
        "M1_1": fs[1][: config.reference_count].mean(0),
        "M1_std0": fs[0][: config.reference_count].std(0, ddof=1),
        "M1_std1": fs[1][: config.reference_count].std(0, ddof=1),
    }
    for j in range(5):
        for key, a in [
            ("observed", targets[j]),
            ("uncertainty", unc[j]),
            ("factor", factors[j]),
            ("mask_offset", offsets[j]),
            ("mask_sigma", sigmas[j]),
        ]:
            values[f"{key}{j}"] = a.detach().cpu().numpy()
    return (
        values,
        dict(heldout_mask_transfer_RMSE=errors, hidden_value_invariance=invariance),
    )


"""Observed-only spectral, noise and WST mask-transfer calibration."""


def calibration_signature(field, config):
    """Bind caches to observations, masks, PSF, geometry and statistical settings."""
    h = hashlib.sha256()
    for value in (
        field.normalized(config)["observed"],
        field.donor,
        field.roi,
        field.PSF,
    ):
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    keys = (
        "pixel_scale",
        "scales",
        "orientations",
        "minimum_kernel_coverage",
        "orientation_pool_fwhm",
        "texture_highpass_fwhm",
        "cdf_fwhm",
        "beta_grid",
        "reference_count",
        "validation_count",
        "noise_seeds",
    )
    h.update(
        json.dumps(
            {
                k: getattr(
                    config,
                    {
                        "orientation_pool_fwhm": "orientation_pool_FWHM",
                        "texture_highpass_fwhm": "texture_highpass_FWHM",
                        "cdf_fwhm": "CDF_FWHM",
                    }.get(k, k),
                )
                for k in keys
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()


@dataclass
class StatisticalCalibration:
    """Noise covariance, texture parameters, fixed WST targets, and diagnostics."""

    noise: dict
    texture: dict
    targets: dict
    signature: str
    diagnostics: dict

    def validate(self, field, config):
        if self.signature != calibration_signature(field, config):
            raise ValueError(
                "Stale calibration: observations, masks, PSF or statistical settings changed"
            )
        if not np.isclose(float(self.noise["beta"]), float(self.texture["beta"])):
            raise ValueError("Noise and texture power-spectrum slopes disagree")


def calibrate_statistics(
    field, config: ModelingConfig, *, progress=None
) -> StatisticalCalibration:
    """Estimate noise and texture statistics from the retained observations.

    Calibration uses 16 reference fields and 8 held-out fields by default.
    Those fields calibrate the estimators; they are not optimized realizations.
    The fitted positive beta denotes P(k) proportional to k**(-beta).
    """

    def report(stage):
        logging.getLogger(__name__).info("Calibration: %s", stage)
        if progress is not None:
            progress(stage)

    d = field.normalized(config)
    ops = NumpyOperators(field.PSF, config)
    d["initial"], _ = ops.smooth_observed(d["observed"], d["donor"], 20)
    report("power spectrum and noise")
    noise, noise_report = calibrate_spectrum(d, field.PSF, config, field.crop)
    report("bootstrap angular reference")
    bootstrap = bootstrap_references(d, noise, field.PSF, config)
    report("five-band texture calibration")
    texture = refine_texture(d, noise, bootstrap, field.PSF, config)
    report("independent mask-transfer calibration")
    targets, target_report = calibrate_targets(
        d, noise, texture, bootstrap["classes"], field.PSF, config
    )
    return StatisticalCalibration(
        noise,
        texture,
        targets,
        calibration_signature(field, config),
        dict(noise=noise_report, targets=target_report),
    )
