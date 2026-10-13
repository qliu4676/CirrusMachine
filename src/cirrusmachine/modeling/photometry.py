"""Source-clean surface brightness profiles and signed aperture fluxes."""

import numpy as np
import torch

from .operators import as_numpy, resolve_device, to_tensor


def estimate_sky(image, valid, radius, annulus=(180.0, 220.0), *, device="auto"):
    """Return a 3-sigma median/MAD-clipped mean sky in image ADU."""
    x, r, mask = _inputs(image, valid, radius, device)
    if len(annulus) != 2 or not 0 <= annulus[0] < annulus[1]:
        raise ValueError("Sky annulus must be increasing and nonnegative")
    use = mask & (r >= annulus[0]) & (r < annulus[1])
    values = x[use]
    if not values.numel():
        raise ValueError("No source-clean sky samples")
    for _ in range(5):
        median = torch.quantile(values, 0.5)
        scale = 1.4826 * torch.quantile((values - median).abs(), 0.5)
        if scale == 0:
            break
        keep = (values - median).abs() <= 3 * scale
        if keep.all():
            break
        values = values[keep]
        if not values.numel():
            raise ValueError("Sky clipping emptied the annulus")
    return dict(
        value_ADU=float(values.mean()),
        scatter_ADU=float(values.std(correction=0)),
        retained_pixels=values.numel(),
        source_clean_pixels=int(use.sum()),
        annulus_arcsec=list(annulus),
        estimator="3sigma median/MAD clip, max5 passes, retained mean",
    )


def _inputs(image, valid, radius, device):
    device = resolve_device(device)
    x = to_tensor(image, dtype=torch.float64, device=device)
    r = to_tensor(radius, dtype=torch.float64, device=device)
    mask = to_tensor(valid, dtype=torch.bool, device=device)
    if x.ndim != 2 or r.shape != x.shape or mask.shape != x.shape:
        raise ValueError("Image, radius and mask must share a 2D grid")
    return x, r, mask & torch.isfinite(x) & torch.isfinite(r)


def measure_profile(
    image, valid, radius, edges, *, sky=0.0, zeropoint=25.13, device="auto"
):
    """Measure clipped annular intensity (ADU) and surface brightness (mag/arcsec²).

    Radius and edges are semimajor radii in arcsec. Each annulus uses the
    reference's 3-sigma median/MAD clipping, at most five passes. Bins with fewer
    than four pixels are undefined. Nonpositive intensities have NaN magnitude.
    ``zeropoint`` converts the supplied intensity convention to surface
    brightness; 25.13 is the provisional mean-binned VIS calibration.
    """
    x, r, mask = _inputs(image, valid, radius, device)
    e = to_tensor(edges, dtype=x.dtype, device=x.device)
    if (
        e.ndim != 1
        or len(e) < 2
        or not torch.isfinite(e).all()
        or not (e[1:] > e[:-1]).all()
    ):
        raise ValueError("Edges must be finite and increasing")
    means, counts, retained = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        values = x[mask & (r >= lo) & (r < hi)]
        counts.append(values.numel())
        if values.numel() < 4:
            means.append(x.new_tensor(float("nan")))
            retained.append(0)
            continue
        for _ in range(5):
            median = torch.quantile(values, 0.5)
            scale = 1.4826 * torch.quantile((values - median).abs(), 0.5)
            if scale == 0:
                break
            keep = (values - median).abs() <= 3 * scale
            if keep.all():
                break
            values = values[keep]
            if not values.numel():
                break
        means.append(values.mean() - sky)
        retained.append(values.numel())
    mean = torch.stack(means)
    sb = torch.where(mean > 0, zeropoint - 2.5 * torch.log10(mean), torch.nan)
    return dict(
        radius=as_numpy((e[:-1] + e[1:]) / 2),
        edges=as_numpy(e),
        intensity=as_numpy(mean),
        surface_brightness=as_numpy(sb),
        count=np.array(counts),
        retained_count=np.array(retained),
    )


def measure_cumulative_light(
    image, valid, radius, apertures, *, sky=0.0, pixel_scale=0.9, device="auto"
):
    """Sum signed source-clean pixels inside each semimajor aperture.

    Returns ``flux`` in summed working-pixel ADU and ``intensity_area`` as
    flux times pixel area in arcsec². This preserves the manuscript aperture
    convention: no clipping of aperture light, no area correction for holes,
    and no inference of total light. Compare truth and data on the same mask.
    Clipped sky is supplied separately and subtracted once per retained pixel.
    """
    x, r, mask = _inputs(image, valid, radius, device)
    a = to_tensor(apertures, dtype=x.dtype, device=x.device)
    if (
        a.ndim != 1
        or not len(a)
        or not torch.isfinite(a).all()
        or (a <= 0).any()
        or not (a[1:] > a[:-1]).all()
        or pixel_scale <= 0
    ):
        raise ValueError("Apertures must increase and pixel scale must be positive")
    flux, counts = [], []
    for hi in a:
        use = mask & (r < hi)
        count = use.sum()
        flux.append(x[use].sum() - sky * count)
        counts.append(count)
    flux = torch.stack(flux)
    return dict(
        radius=as_numpy(a),
        flux=as_numpy(flux),
        intensity_area=as_numpy(flux * pixel_scale**2),
        count=as_numpy(torch.stack(counts)),
    )
