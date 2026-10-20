"""Coarse native-sample calibration with empirical discrepancy covariance."""

import logging

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from .calibration import NumpyOperators, calibration_signature
from .operators import as_numpy, resolve_device, to_tensor

logger = logging.getLogger(__name__)


def prepare_backbone_constraint(
    field,
    image,
    error,
    coverage,
    holes,
    coarse_WCS,
    config,
    *,
    effective_FWHM,
    sample_stride=4,
    sample_margin=90,
    calibration_margin=200,
):
    """Calibrate native coarse samples against retained optical pixels.

    Parameters
    ----------
    field : FieldData
        Working optical canvas with its WCS.
    image, error, coverage, holes : ndarray
        Native guide map and 1-sigma error in matching intensity units,
        coverage and exclusion mask (True for excluded samples).
    coarse_WCS : astropy.wcs.WCS
        WCS of the native coarse grid.
    effective_FWHM : float
        Actual effective resolution of the supplied coarse image in arcsec.
        Must equal config.beam_FWHM. No filled coarse samples enter the likelihood.

    Returns
    -------
    dict
        Normalized optical targets, covariance Cholesky, sample coordinates,
        and an interpolated guide used only for initialization.
    """
    if field.WCS is None:
        raise ValueError("An optical WCS is required for coarse alignment")
    if not np.isclose(effective_FWHM, config.beam_FWHM):
        raise ValueError("Input coarse resolution differs from the requested beam")
    if len({np.shape(v) for v in (image, error, coverage, holes)}) != 1:
        raise ValueError("Coarse image, error, coverage and holes must share a grid")
    holes = np.asarray(holes, dtype=bool)
    if sample_stride < 1 or sample_margin < 0 or calibration_margin < 0:
        raise ValueError(
            "Coarse sample stride and margins must be nonnegative (stride positive)"
        )
    d = field.normalized(config)
    w = field.WCS
    hw = coarse_WCS
    n = field.image.shape[0]
    smooth_observed = NumpyOperators(field.PSF, config).smooth_observed
    yh, xh = np.indices(image.shape)
    RA, Dec = hw.all_pix2world(xh, yh, 0)
    x, y = w.all_world2pix(RA, Dec, 0)
    sigma = config.beam_FWHM / config.pixel_scale / 2.354820045
    eligible = (
        (x > sample_margin)
        & (x < n - sample_margin)
        & (y > sample_margin)
        & (y < n - sample_margin)
        & ~holes
        & np.isfinite(image)
        & np.isfinite(error)
        & (error > 0)
        & (coverage > 0)
    )
    samples = eligible & (xh % sample_stride == 0) & (yh % sample_stride == 0)
    hx = x[samples]
    hy = y[samples]
    hv = image[samples]
    he = error[samples]
    sm, cov = smooth_observed(d["observed"], d["donor"], sigma)
    vv = map_coordinates(sm, [hy, hx], order=1)
    support = map_coordinates(cov, [hy, hx], order=1)
    threshold = config.calibration_min_support
    cal = (
        (support > threshold)
        & (hx > calibration_margin)
        & (hx < n - calibration_margin)
        & (hy > calibration_margin)
        & (hy < n - calibration_margin)
    )
    if int(cal.sum()) < 8:
        raise ValueError(
            f"Only {int(cal.sum())} coarse calibration samples exceed support {threshold}; at least eight are required"
        )
    pivot = float(np.median(hv[cal]))
    hh = hv - pivot

    def fit(sel):
        H = hh[sel]
        V = vv[sel]
        E = he[sel]
        if len(H) < 3:
            raise ValueError(
                "Insufficient coarse calibration samples in a spatial fold"
            )
        a, b = np.polyfit(H, V, 1)
        init = np.array([a, b, np.log(max(np.std(V - (a * H + b)), 0.03))])

        def fun(par):
            a, b, ls = par
            v = np.exp(2 * ls) + (a * E) ** 2
            return np.sum((V - a * H - b) ** 2 / v + np.log(v)) / len(V)

        opt = minimize(
            fun, init, method="L-BFGS-B", bounds=[(-10, 10), (-10, 10), (-7, 3)]
        )
        return opt.x

    pars = fit(cal)
    a, b, ls = pars
    delta_sigma = np.exp(ls)
    blocks = (hx // 100 + 3 * (hy // 100)).astype(int) % 4
    held = []
    jack = []
    for k in range(4):
        par = fit(cal & (blocks != k))
        sel = cal & (blocks == k)
        held.extend((vv[sel] - par[0] * hh[sel] - par[1]).tolist())
        jack.append(par[:2])
    jack = np.array(jack)
    cal_cov = np.cov(jack.T) * 3
    device = resolve_device(config.device)

    def tensor(value):
        return to_tensor(value, dtype=torch.float64, device=device)

    coordinates = tensor(np.column_stack([hx, hy]))
    distance = (
        torch.cdist(
            coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        * config.pixel_scale
    )
    rr = tensor(vv - (a * hh + b))
    use = to_tensor(cal, device=device)
    pairs = use[:, None] & use[None, :]
    lengths = (18.0, 30.0, 45.0, 60.0, 90.0, 120.0)
    scores = []
    for ell in lengths:
        model = (1 + np.sqrt(3) * distance / ell) * torch.exp(
            -np.sqrt(3) * distance / ell
        )
        score = 0.0
        for lo, hi in ((18, 36), (36, 60), (60, 100), (100, 160)):
            select = pairs & (distance >= lo) & (distance < hi)
            if select.any():
                score += float(
                    (
                        (rr[:, None] * rr[None, :])[select].mean()
                        - delta_sigma**2 * model[select].mean()
                    )
                    ** 2
                )
        scores.append(score)
    ell = lengths[int(np.argmin(scores))]
    matern = (1 + np.sqrt(3) * distance / ell) * torch.exp(-np.sqrt(3) * distance / ell)
    design = tensor(np.column_stack([hh, np.ones(len(hh))]))
    covar = (
        delta_sigma**2 * matern
        + torch.diag(tensor((a * he) ** 2 + 0.01**2))
        + design @ tensor(cal_cov) @ design.T
    )
    chol = as_numpy(torch.linalg.cholesky(covar))
    grid = np.stack([2 * (hx + 0.5) / n - 1, 2 * (hy + 0.5) / n - 1], axis=-1)[
        None, :, None, :
    ].astype("f4")
    target = a * hh + b
    RA, Dec = w.all_pix2world(*np.meshgrid(np.arange(n), np.arange(n)), 0)
    ix, iy = hw.all_world2pix(RA, Dec, 0)
    hsafe, hcov = smooth_observed(
        image, eligible | (coverage > 0) & ~holes & np.isfinite(image), 1.5
    )
    backbone = (
        a * (map_coordinates(hsafe, [iy, ix], order=1, mode="nearest") - pivot) + b
    )
    logger.info(
        "Backbone likelihood: %d native samples, %d calibration samples",
        len(target),
        int(cal.sum()),
    )
    return dict(
        optical_signature=np.array(calibration_signature(field, config)),
        effective_FWHM=effective_FWHM,
        sample_grid=grid,
        target=target,
        chol=chol,
        x=hx,
        y=hy,
        coarse_image=hv,
        coarse_error=he,
        support=support,
        cal=cal,
        predicted=target,
        observed_image=vv,
        sigma=sigma,
        backbone=backbone,
        calibration_covariance=cal_cov,
        discrepancy_sigma=delta_sigma,
        discrepancy_length_arcsec=ell,
    )
