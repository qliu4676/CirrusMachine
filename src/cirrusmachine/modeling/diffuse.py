"""Observed-only annulus-relative diffuse model with nonnegative intensity and smooth shape.

Source holes are excluded from every sample. Output over a hole is interpolation,
never an observation. No injected brightness, profile index or effective radius
is used to estimate the diffuse model. Photutils models the delivered PSF-convolved image.
"""

import time

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from photutils.isophote import Ellipse, EllipseGeometry, IsophoteList
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.optimize import least_squares, nnls

from .operators import MaskedCovariance, resolve_device


class _UnsupportedOutwardSample(RuntimeError):
    def __init__(self, sequence, sma):
        self.sequence = list(sequence)
        self.sma = float(sma)


class SupportedEllipse(Ellipse):
    """Terminate unsupported outward sampling instead of repeating the last SMA.

    Photutils does not append invalid stop-code-3 samples, but its outward loop
    advances using the last appended sample. This can repeat forever. Retain only
    the already sampled isophotes; existing reliability and freeze rules apply.
    Core intensity measurements below still use all source-clean core pixels.
    """

    def fit_isophote(self, sma, *args, **kwargs):
        result = super().fit_isophote(sma, *args, **kwargs)
        if result.stop_code == 3 and (not kwargs.get("going_inwards", False)):
            raise _UnsupportedOutwardSample(kwargs.get("isophote_list") or [], sma)
        return result

    def fit_image(self, *args, **kwargs):
        self.unsupported_outward_sma = None
        try:
            return super().fit_image(*args, **kwargs)
        except _UnsupportedOutwardSample as error:
            self.unsupported_outward_sma = error.sma
            return IsophoteList(error.sequence)


def clipped(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        raise ValueError("No source-clean samples")
    for _ in range(5):
        med = np.median(a)
        scale = 1.4826 * np.median(np.abs(a - med))
        if scale == 0:
            break
        keep = np.abs(a - med) <= 3 * scale
        if keep.all():
            break
        a = a[keep]
        if not len(a):
            raise ValueError("Clipping emptied samples")
    return (float(a.mean()), float(a.std()), int(len(a)))


def radius_map(shape, center, q, PA, pix=0.9):
    y, x = np.indices(shape)
    t = np.deg2rad(PA)
    u = (x - center[0]) * np.cos(t) + (y - center[1]) * np.sin(t)
    v = (-(x - center[0]) * np.sin(t) + (y - center[1]) * np.cos(t)) / q
    return pix * np.hypot(u, v)


def sky_estimate(residual, valid, radius, annulus):
    use = valid & (radius >= annulus[0]) & (radius < annulus[1])
    mean, scatter, n = clipped(residual[use])
    return dict(
        value_ADU=mean,
        scatter_ADU=scatter,
        retained_pixels=n,
        source_clean_pixels=int(use.sum()),
        annulus_arcsec=annulus,
        estimator="3sigma median/MAD clip, max5 passes, retained mean",
    )


def shape_values(radius, info):
    lo, hi = info["shape_range_arcsec"]
    t = np.clip(
        (np.log(np.maximum(radius, 0.1)) - np.log(lo)) / max(np.log(hi / lo), 1e-06),
        0,
        1,
    )
    k = len(info["shape_controls"])
    controls = np.asarray(info["shape_controls"])
    if k == 1:
        z = np.broadcast_to(controls[0], np.shape(t) + (2,))
    elif k == 2:
        z = (1 - t)[..., None] * controls[0] + t[..., None] * controls[1]
    else:
        z = (
            ((1 - t) ** 2)[..., None] * controls[0]
            + (2 * t * (1 - t))[..., None] * controls[1]
            + (t * t)[..., None] * controls[2]
        )
    e = np.minimum(np.linalg.norm(z, axis=-1), 2 / 3)
    q = (1 - e) / (1 + e)
    PA = 0.5 * np.arctan2(z[..., 1], z[..., 0])
    return (q, PA)


def nested(info, maxr):
    rs = np.geomspace(0.45, maxr, 180)
    q, PA = shape_values(rs, info)
    a = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    for i in range(len(rs) - 1):
        x = rs[i] * (np.cos(a) * np.cos(PA[i]) - q[i] * np.sin(a) * np.sin(PA[i]))
        y = rs[i] * (np.cos(a) * np.sin(PA[i]) + q[i] * np.sin(a) * np.cos(PA[i]))
        u = x * np.cos(PA[i + 1]) + y * np.sin(PA[i + 1])
        v = (-x * np.sin(PA[i + 1]) + y * np.cos(PA[i + 1])) / q[i + 1]
        if np.max(np.hypot(u, v)) > rs[i + 1] * (1 + 1e-08):
            return False
    return True


def smooth_shape(iso, previous=None, pixel_scale=0.9):
    good = [
        i
        for i in iso
        if i.stop_code == 0
        and i.n_iter > 0
        and (i.sma >= 1)
        and np.isfinite(i.eps)
        and (0 <= i.eps <= 0.8)
        and np.isfinite(i.pa)
    ]
    if not good:
        if previous is None:
            raise RuntimeError(
                "No reliable geometry; no previous observed diffuse model to retain"
            )
        return (
            {
                k: previous[k]
                for k in [
                    "shape_range_arcsec",
                    "shape_controls",
                    "last_reliable_geometry_arcsec",
                    "nested_geometry_damping",
                ]
            },
            0,
        )
    r = np.array([i.sma * pixel_scale for i in good])
    q = np.array([1 - i.eps for i in good])
    PA = np.array([i.pa for i in good])
    z = ((1 - q) / (1 + q))[:, None] * np.stack(
        [np.cos(2 * PA), np.sin(2 * PA)], axis=1
    )
    lo = max(r.min(), 0.45)
    hi = max(r.max(), lo * 1.01)
    t = np.clip(np.log(r / lo) / np.log(hi / lo), 0, 1)
    k = 3 if len(r) >= 6 else 2 if len(r) >= 3 else 1
    A = (
        np.ones((len(r), 1))
        if k == 1
        else np.stack([1 - t, t], axis=1)
        if k == 2
        else np.stack([(1 - t) ** 2, 2 * t * (1 - t), t * t], axis=1)
    )
    init = np.linalg.lstsq(A, z, rcond=None)[0]
    norm = np.linalg.norm(init, axis=1)
    init *= np.minimum(1, 2 / 3 / np.maximum(norm, 1e-12))[:, None]

    def unpack(v):
        zz = v.reshape(k, 2)
        nn = np.linalg.norm(zz, axis=1)
        return zz * np.minimum(1, 2 / 3 / np.maximum(nn, 1e-12))[:, None]

    controls = unpack(
        least_squares(
            lambda v: (A @ unpack(v) - z).ravel(), init.ravel(), max_nfev=200
        ).x
    )
    info = dict(
        shape_range_arcsec=[float(lo), float(hi)],
        shape_controls=controls.tolist(),
        last_reliable_geometry_arcsec=float(r.max()),
        nested_geometry_damping=1.0,
    )
    median = np.median(z, axis=0)
    for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0]:
        info["shape_controls"] = (alpha * controls + (1 - alpha) * median).tolist()
        info["nested_geometry_damping"] = alpha
        if nested(info, 1000.0):
            break
    if not nested(info, 1000.0):
        raise RuntimeError("Diffuse isophotes are not nested")
    return (info, len(good))


def elliptical_coordinates(shape, center, info, pixel_scale=0.9):
    y, x = np.indices(shape)
    dx = pixel_scale * (x - center[0])
    dy = pixel_scale * (y - center[1])
    lo = np.hypot(dx, dy)
    hi = lo / 0.2 + 0.01
    for _ in range(18):
        r = (lo + hi) / 2
        q, PA = shape_values(r, info)
        u = dx * np.cos(PA) + dy * np.sin(PA)
        v = (-dx * np.sin(PA) + dy * np.cos(PA)) / q
        outside = np.hypot(u, v) > r
        lo = np.where(outside, r, lo)
        hi = np.where(outside, hi, r)
    return (lo + hi) / 2


def fit_diffuse(
    residual,
    valid,
    center,
    measurement_radius,
    annulus,
    previous=None,
    *,
    pixel_scale=0.9,
):
    """Estimate smooth, nonnegative diffuse light and sky from retained pixels.

    The profile is relative to the supplied sky annulus. Elliptical isophotes
    describe the delivered image; no intrinsic profile or injected truth is used.
    """
    started = time.monotonic()
    sky = sky_estimate(residual, valid, measurement_radius, annulus)
    image = np.ma.array(residual - sky["value_ADU"], mask=~valid)
    attempts = []
    iso = []
    seeds = [(0.7, 0.0)]
    if previous:
        q, p = shape_values(np.asarray(10.0), previous)
        seeds.insert(0, (float(q), float(p)))
    y, x = np.indices(residual.shape)
    rr = pixel_scale * np.hypot(x - center[0], y - center[1])
    w = np.where(valid & (rr < 30), np.maximum(residual - sky["value_ADU"], 0), 0)
    if w.sum() > 0:
        w = np.minimum(w, np.percentile(w[valid & (rr < 30)], 99))
        dx = x - center[0]
        dy = y - center[1]
        cov = (
            np.array(
                [
                    [np.sum(w * dx * dx), np.sum(w * dx * dy)],
                    [np.sum(w * dx * dy), np.sum(w * dy * dy)],
                ]
            )
            / w.sum()
        )
        v, V = np.linalg.eigh(cov)
        seeds.append(
            (
                float(np.clip(np.sqrt(max(v[0], 0) / max(v[1], 1e-12)), 0.2, 0.95)),
                float(np.arctan2(V[1, 1], V[0, 1]) % np.pi),
            )
        )
    for q, p in seeds:
        for start in [10.0, 20.0, 5.0, 40.0]:
            try:
                ell = SupportedEllipse(
                    image,
                    geometry=EllipseGeometry(*center, start / pixel_scale, 1 - q, p),
                    threshold=np.inf,
                )
                iso = ell.fit_image(
                    sma0=start / pixel_scale,
                    minsma=1.0,
                    maxsma=annulus[1] / pixel_scale,
                    step=0.12,
                    conver=0.1,
                    minit=10,
                    maxit=100,
                    fflag=0.25,
                    maxgerr=1.0,
                    sclip=3.0,
                    n_clip=3,
                    integrmode="median",
                    linear=False,
                    fix_center=True,
                    fix_pa=False,
                    fix_eps=False,
                )
                attempts.append(
                    dict(
                        q=q,
                        PA_deg=float(np.rad2deg(p)),
                        start_arcsec=start,
                        isophotes=len(iso),
                        unsupported_outward_sma_pixels=ell.unsupported_outward_sma,
                    )
                )
                if len(iso) >= 5:
                    break
            except (ValueError, IndexError, RuntimeError) as e:
                attempts.append(dict(error=str(e), start_arcsec=start))
                iso = []
        if len(iso) >= 5:
            break
    geometry_refresh_supported = len(iso) >= 5
    if not geometry_refresh_supported:
        if previous is None:
            raise RuntimeError("No usable source-clean isophote sequence")
        iso = IsophoteList([])
    info, ngood = smooth_shape(iso, previous, pixel_scale)
    r = elliptical_coordinates(residual.shape, center, info, pixel_scale)
    edges = np.r_[
        0.0,
        pixel_scale,
        2 * pixel_scale,
        4 * pixel_scale,
        5.0,
        np.arange(10.0, 45.0, 5.0),
        np.geomspace(
            40, annulus[1], int(np.ceil(np.log(annulus[1] / 40) / np.log(1.15))) + 1
        )[1:],
    ]
    knots = []
    levels = []
    counts = []
    signed = []
    for a, b in zip(edges[:-1], edges[1:]):
        use = valid & (r >= a) & (r < b)
        if use.sum() < 8:
            continue
        mean, _, n = clipped(residual[use] - sky["value_ADU"])
        knots.append(float(np.sqrt(a * b) if a >= 40 else (a + b) / 2))
        levels.append(max(mean, 0.0))
        signed.append(mean)
        counts.append(n)
    if len(knots) < 5:
        raise RuntimeError("Insufficient unmasked radial support")
    knots = [0.0] + knots + [float(annulus[1])]
    levels = [levels[0]] + levels + [0.0]
    interp = PchipInterpolator(knots, levels, extrapolate=False)
    model = interp(np.minimum(r, annulus[1]))
    model[r >= annulus[1]] = 0.0
    model = np.maximum(model, 0.0)
    info.update(
        sky=sky,
        knots_arcsec=knots,
        amplitudes_ADU=levels,
        signed_annular_measurements_ADU=signed,
        annular_counts=counts,
        geometry_normal_samples=ngood,
        geometry_refresh_supported=geometry_refresh_supported,
        geometry_retained_from_previous=not geometry_refresh_supported,
        photutils_stop_codes={
            str(int(k)): int(np.sum(iso.stop_code == k))
            for k in np.unique(iso.stop_code)
        },
        attempts=attempts,
        geometry_frozen_beyond_arcsec=info["last_reliable_geometry_arcsec"],
        nonnegative=True,
        high_harmonics=False,
        source_holes_excluded=True,
        source_clean_central20_pixels=int(np.sum(valid & (measurement_radius < 20))),
        model_outer_support_arcsec=float(annulus[1]),
        outer_boundary="annulus-relative diffuse model tapers to zero at adopted annulus outer edge; not intrinsic total luminosity",
        runtime_seconds=time.monotonic() - started,
    )
    if not np.isfinite(model).all() or model.min() < 0:
        raise RuntimeError("Invalid nonnegative diffuse model model")
    return (model.astype("f4"), info)


def radii(dx, dy, info):
    lo = np.hypot(dx, dy)
    hi = lo / 0.2 + 0.01
    for _ in range(24):
        r = (lo + hi) * 0.5
        q, PA = shape_values(r, info)
        u = dx * np.cos(PA) + dy * np.sin(PA)
        v = (-dx * np.sin(PA) + dy * np.cos(PA)) / q
        outside = np.hypot(u, v) > r
        lo = np.where(outside, r, lo)
        hi = np.where(outside, hi, r)
    return (lo + hi) * 0.5


def refit_inner_diffuse(residual, original, clean, center, info, cov, config):
    """Refine the central diffuse profile with pixel integration and covariance."""
    boundary = config.core_radius + 0.5
    pixel = config.pixel_scale
    half = int(np.ceil(boundary / pixel)) + 2
    cx, cy = center
    xlo = int(cx) - half
    xhi = int(cx) + half + 2
    ylo = int(cy) - half
    yhi = int(cy) + half + 2
    if xlo < 0 or ylo < 0 or xhi > original.shape[1] or (yhi > original.shape[0]):
        raise ValueError("Inner diffuse model aperture leaves the crop")
    ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
    dx = pixel * (xs - cx)
    dy = pixel * (ys - cy)
    rcenter = radii(dx, dy, info)
    inner = rcenter < boundary
    valid = clean[ylo:yhi, xlo:xhi] & inner
    residual = residual[ylo:yhi, xlo:xhi]
    breaks = np.unique(
        np.r_[
            np.array(
                [0.0, pixel, 2 * pixel, 3 * pixel, 4.3, 7.5, 12.5, 17.5, 22.5, 27.5]
            )[
                np.array(
                    [0.0, pixel, 2 * pixel, 3 * pixel, 4.3, 7.5, 12.5, 17.5, 22.5, 27.5]
                )
                < boundary
            ],
            boundary,
        ]
    )
    k = 3
    knots = np.r_[[0.0] * 4, breaks[1:-1], [boundary] * 4]
    nc = len(knots) - k - 1
    nf = nc - 3
    old = PchipInterpolator(info["knots_arcsec"], info["amplitudes_ADU"])
    bv = float(old(boundary))
    bd = float(old.derivative()(boundary))
    fixed = np.zeros(nc)
    fixed[-1] = bv
    fixed[-2] = bv - (boundary - breaks[-2]) * bd / 3
    if np.min(fixed) < 0:
        raise RuntimeError("Inner diffuse model boundary coefficients are negative")
    transform = np.zeros((nc, nf))
    transform[0, 0] = transform[1, 0] = 1
    for j in range(2, nc - 2):
        transform[j, j - 1] = 1

    def design(order):
        nodes, w = leggauss(order)
        A = np.zeros(dx.shape + (nf,))
        offset = np.zeros(dx.shape)
        for a, wa in zip(nodes, w):
            for b, wb in zip(nodes, w):
                r = radii(dx + pixel / 2 * a, dy + pixel / 2 * b, info)
                inside = r < boundary
                rr = np.clip(r, 0, boundary)
                basis = (
                    BSpline.design_matrix(rr.ravel(), knots, k)
                    .toarray()
                    .reshape(dx.shape + (nc,))
                )
                weight = wa * wb / 4
                A += weight * (basis @ transform) * inside[..., None]
                offset += weight * np.where(
                    inside,
                    basis @ fixed,
                    old(np.minimum(r, info["model_outer_support_arcsec"])),
                )
        return (A, offset)

    like = MaskedCovariance(
        valid, cov, rtol=2e-09, maxiter=600, device=resolve_device(config.device)
    )
    cg = []

    def solve_vector(v):
        out = (
            like.solve(torch.tensor(v, dtype=torch.float64, device=like.device))
            .cpu()
            .numpy()
        )
        cg.append(dict(like.last))
        return out

    def fit(A, offset, observed):
        X = A[valid]
        y = (observed - offset)[valid]
        ciX = np.column_stack([solve_vector(X[:, j]) for j in range(nf)])
        ciy = solve_vector(y)
        G = X.T @ ciX
        G = (G + G.T) / 2
        g = X.T @ ciy
        scale = np.sqrt(np.diag(G))
        if not (scale > 0).all():
            raise RuntimeError("Unidentifiable inner diffuse model amplitudes")
        Gn = G / scale[:, None] / scale[None, :]
        gn = g / scale
        eig = np.linalg.eigvalsh(Gn)
        if eig.min() <= 1e-10:
            raise RuntimeError(f"Unidentifiable central basis: {eig.tolist()}")
        L = np.linalg.cholesky(Gn)
        u, _ = nnls(L.T, np.linalg.solve(L, gn), maxiter=5000)
        amps = u / scale
        grad = Gn @ u - gn
        kkt = max(
            float(np.max(np.abs(grad[u > 1e-10]), initial=0)),
            float(np.max(np.maximum(-grad[u <= 1e-10], 0), initial=0)),
        ) / max(float(np.max(np.abs(gn))), 1.0)
        if kkt >= 1e-8:
            raise RuntimeError(
                f"Inner diffuse model NNLS optimality check failed: {kkt}"
            )
        prediction = offset + A @ amps
        err = (observed - prediction)[valid]
        chi = float(err @ solve_vector(err))
        return (
            amps,
            prediction,
            dict(
                kkt=kkt,
                chi2=chi,
                reduced_chi2=chi / (valid.sum() - nf),
                parameters=nf,
                pixels=int(valid.sum()),
                normalized_Gram_eigenvalues=eig.tolist(),
                condition_number=float(eig.max() / eig.min()),
            ),
        )

    trials = []
    for order in [12, 24, 48]:
        A, off = design(order)
        amps, pred, solver = fit(A, off, residual)
        auditA, auditoff = design(order * 2)
        audit = auditoff + auditA @ amps
        delta = (pred - audit)[valid]
        qrms = float(np.sqrt(np.mean(delta**2)))
        qmax = float(np.max(np.abs(delta)))
        trials.append(
            dict(order=order, audit_order=order * 2, RMS_ADU=qrms, max_ADU=qmax)
        )
        if qrms < 0.001 and qmax < 0.01:
            break
    else:
        raise RuntimeError("Pixel quadrature did not reach required precision")
    if solver["kkt"] >= 1e-8 or max(v["relative_residual"] for v in cg) >= 2e-6:
        raise RuntimeError("Inner diffuse model solve failed numerical validation")
    output = original.copy()
    patch = output[ylo:yhi, xlo:xhi]
    patch[inner] = pred[inner]
    if not np.isfinite(output).all() or output.min() < 0:
        raise RuntimeError("Invalid inner diffuse model")
    return (
        output,
        dict(
            spline_knots=knots,
            coefficients=transform @ amps + fixed,
            degree=k,
            boundary_arcsec=boundary,
            quadrature=trials,
            solver=solver,
            covariance_solves=cg,
        ),
    )
