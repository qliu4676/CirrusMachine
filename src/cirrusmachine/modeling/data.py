"""Observed images, explicit masks, and optional galaxy injection."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaincinv

from .config import ModelingConfig
from .operators import fft_convolve, resolve_device, to_tensor


@dataclass
class FieldData:
    """Delivered optical image and masks on a square working canvas.

    Intensities are ADU per delivered pixel (mean-binned images retain this
    convention). Masks are True for usable pixels. ``donor`` excludes both
    sources and the target; ``source_clean`` retains the target for halo fits.
    ``crop`` gives the scientific image's y and x slices. The surrounding
    context is retained for FFTs, with no imputation of measured source holes.
    ``radius`` is an elliptical semimajor radius in arcsec on the whole canvas.
    """

    image: np.ndarray
    donor: np.ndarray
    source_clean: np.ndarray
    crop: tuple[slice, slice]
    center_xy: tuple[float, float]
    radius: np.ndarray
    PSF: np.ndarray
    pixel_scale: float = 0.9
    exclusion_radius: float = 150.0
    WCS: object = None
    metadata: dict = field(default_factory=dict)
    normalization: tuple[float, float] | None = None

    def __post_init__(self):
        self.image = np.asarray(self.image, dtype=float)
        shape = self.image.shape
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 32:
            raise ValueError("J=5 requires a square canvas at least 32 pixels wide")
        for name in ("donor", "source_clean"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"{name} shape differs from image")
            setattr(self, name, value)
        self.radius = np.asarray(self.radius, dtype=float)
        if self.radius.shape != shape or np.isnan(self.radius).any():
            raise ValueError("radius must match the image and contain no NaN")
        if np.any(self.donor & ~self.source_clean) or not self.donor.any():
            raise ValueError("donor must be a nonempty subset of source_clean")
        if not np.isfinite(self.image[self.source_clean]).all():
            raise ValueError("Usable image pixels must be finite")
        if self.pixel_scale <= 0 or self.exclusion_radius <= 0:
            raise ValueError("Pixel scale and exclusion radius must be positive")
        if len(self.crop) != 2 or any(
            s.step not in (None, 1)
            or s.start is None
            or s.stop is None
            or not 0 <= s.start < s.stop <= n
            for s, n in zip(self.crop, shape)
        ):
            raise ValueError(
                "crop must contain two explicit, in-bounds unit-step slices"
            )
        if not (
            0 <= self.center_xy[0] < self.image[self.crop].shape[1]
            and 0 <= self.center_xy[1] < self.image[self.crop].shape[0]
        ):
            raise ValueError("center_xy must lie inside the scientific crop")
        self.PSF = np.asarray(self.PSF, dtype=float)
        if (
            self.PSF.ndim != 2
            or not np.isfinite(self.PSF).all()
            or not np.isclose(self.PSF.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("PSF must be a finite unit-sum delivered-VIS kernel")
        if self.normalization is None:
            self.normalization = (
                float(self.image[self.donor].mean()),
                float(self.image[self.donor].std()),
            )
        if not np.isfinite(self.normalization).all() or self.normalization[1] <= 0:
            raise ValueError(
                "Normalization needs a finite mean and positive standard deviation"
            )

    @property
    def roi(self):
        mask = np.zeros(self.image.shape, bool)
        mask[self.crop] = True
        return mask

    def normalized(self, config: ModelingConfig):
        """Return erased fitting inputs; source holes cannot become observations."""
        if not np.isclose(config.pixel_scale, self.pixel_scale):
            raise ValueError("Field and model pixel scales disagree")
        mu, std = self.normalization
        observed = np.where(self.donor, (self.image - mu) / std, 0).astype("f4")
        science = np.where(self.source_clean, (self.image - mu) / std, 0).astype("f4")
        interior = (
            self.source_clean
            & (self.radius > config.core_radius)
            & (self.radius < self.exclusion_radius)
        )
        mask = self.donor | interior
        if np.any(mask & (self.radius <= config.core_radius)):
            raise ValueError("Donor mask exposes the excluded core")
        x = np.clip((self.radius - config.core_radius) / config.core_taper, 0, 1)
        taper = x * x * (3 - 2 * x)
        taper[~interior] = 1
        return dict(
            observed=observed,
            science=science,
            donor=self.donor,
            roi=self.roi,
            source_clean=self.source_clean,
            radius=self.radius,
            taper=taper,
            interior_mask=interior,
            likelihood_mask=mask,
            original_normalizer=int(self.donor.sum()),
            mu=mu,
            std=std,
        )


@dataclass
class Injection:
    """Observed injected field and separate known delivered-image truth in ADU."""

    field: FieldData
    truth: np.ndarray
    parameters: dict


def sersic_image(
    shape,
    center_xy,
    components,
    *,
    pixel_scale,
    axis_ratio=0.6,
    PA_deg=30.0,
    device="auto",
):
    """Evaluate summed intrinsic Sérsic intensities.

    Each component supplies ``n``, ``Re_arcsec``, and either ``Ie`` (intensity
    at Re) or ``I0`` (central intensity), in image ADU. They obey
    I0 = Ie * exp(b_n). The older ``ie`` and ``re_arcsec`` keys remain readable.
    Position angle runs counterclockwise from image +x; center uses (x, y).
    """
    if not 0 < axis_ratio <= 1 or pixel_scale <= 0:
        raise ValueError("Invalid axis ratio or pixel scale")
    device = resolve_device(device)
    y, x = torch.meshgrid(
        torch.arange(shape[0], device=device, dtype=torch.float64),
        torch.arange(shape[1], device=device, dtype=torch.float64),
        indexing="ij",
    )
    angle = np.deg2rad(PA_deg)
    u = (x - center_xy[0]) * np.cos(angle) + (y - center_xy[1]) * np.sin(angle)
    v = (
        -(x - center_xy[0]) * np.sin(angle) + (y - center_xy[1]) * np.cos(angle)
    ) / axis_ratio
    r = pixel_scale * torch.hypot(u, v)
    out = torch.zeros_like(r)
    for comp in components:
        amplitudes = [key for key in ("Ie", "I0", "ie") if key in comp]
        radii = [key for key in ("Re_arcsec", "re_arcsec") if key in comp]
        if len(amplitudes) != 1 or len(radii) != 1:
            raise ValueError("Supply one of Ie/I0 (legacy ie), and one Re_arcsec")
        n, Re, intensity = comp["n"], comp[radii[0]], comp[amplitudes[0]]
        if (
            not np.isfinite([n, Re, intensity]).all()
            or n <= 0
            or Re <= 0
            or intensity < 0
        ):
            raise ValueError("Sérsic n/Re must be positive and Ie nonnegative")
        b_n = gammaincinv(2 * n, 0.5)
        if amplitudes[0] == "I0":
            out += intensity * torch.exp(-b_n * (r / Re).pow(1 / n))
        else:
            out += intensity * torch.exp(-b_n * ((r / Re).pow(1 / n) - 1))
    return out


def prepare_injection(
    field: FieldData,
    components,
    *,
    axis_ratio=0.6,
    PA_deg=30.0,
    PSF_native=None,
    oversample=9,
    margin_pixels=24,
    device="auto",
) -> Injection:
    """Sample intrinsic light, convolve once, mean-bin, and add to the field.

    ``PSF_native`` must be sampled at working pixel_scale / oversample.
    Supply the detector PSF explicitly; the field PSF describes the working
    pixels and is used only by the cirrus texture model. No sky is injected.
    """
    if PSF_native is None:
        raise ValueError("Supply the PSF at the intrinsic sampling scale")
    if not isinstance(oversample, int) or oversample < 1 or margin_pixels < 0:
        raise ValueError("Invalid sampling or convolution margin")
    device = resolve_device(device)
    kernel = to_tensor(PSF_native, dtype=torch.float64, device=device)
    if kernel.ndim != 2 or not torch.isfinite(kernel).all() or kernel.sum() <= 0:
        raise ValueError("Invalid detector PSF")
    kernel = kernel / kernel.sum()
    ny, nx = field.image.shape
    m, s = margin_pixels, oversample
    cx = field.center_xy[0] + field.crop[1].start
    cy = field.center_xy[1] + field.crop[0].start
    fine_center = ((cx + m + 0.5) * s - 0.5, (cy + m + 0.5) * s - 0.5)
    fine_shape = ((ny + 2 * m) * s, (nx + 2 * m) * s)
    # Remove coordinate roundoff at integer detector samples: a steep central
    # cusp can amplify tiny offsets that arise only from changing pixel grids.
    tolerance = 8 * np.finfo(float).eps * max(fine_shape)
    fine_center = tuple(
        float(round(c)) if abs(c - round(c)) <= tolerance else c for c in fine_center
    )
    # Match the reference's float32 intrinsic raster, without materializing
    # full-size coordinate/radius arrays at detector resolution.
    intrinsic = torch.empty(fine_shape, dtype=torch.float32, device=device)
    for start in range(0, fine_shape[0], 150):
        rows = min(150, fine_shape[0] - start)
        intrinsic[start : start + rows] = sersic_image(
            (rows, fine_shape[1]),
            (fine_center[0], fine_center[1] - start),
            components,
            pixel_scale=field.pixel_scale / s,
            axis_ratio=axis_ratio,
            PA_deg=PA_deg,
            device=device,
        )
    delivered = (
        fft_convolve(intrinsic, kernel)
        .reshape(ny + 2 * m, s, nx + 2 * m, s)
        .mean((1, 3))
    )
    truth = delivered[m : m + ny, m : m + nx].cpu().numpy()
    injected = FieldData(
        field.image + truth,
        field.donor.copy(),
        field.source_clean.copy(),
        field.crop,
        field.center_xy,
        field.radius.copy(),
        field.PSF.copy(),
        field.pixel_scale,
        field.exclusion_radius,
        field.WCS,
        dict(field.metadata),
    )
    return Injection(
        injected,
        truth,
        dict(
            components=components,
            axis_ratio=axis_ratio,
            PA_deg=PA_deg,
            oversample=s,
            margin_pixels=m,
        ),
    )


def load_field_data(path: str | Path, *, PSF: np.ndarray) -> FieldData:
    """Load a portable, pickle-free working-grid NPZ (see modeling documentation)."""
    with np.load(path, allow_pickle=False) as z:
        crop = tuple(slice(int(a), int(b)) for a, b in z["crop"])
        return FieldData(
            z["image"],
            z["donor"],
            z["source_clean"],
            crop,
            tuple(z["center_xy"]),
            z["radius"],
            PSF,
            float(z["pixel_scale"]),
            float(z["exclusion_radius"]),
        )


def prepare_field_data(
    image,
    source_mask,
    target_mask,
    PSF,
    *,
    center_xy,
    axis_ratio,
    PA_deg,
    native_pixel_scale=0.3,
    bin_factor=3,
    canvas_size=1024,
    exclusion_radius=150.0,
    WCS=None,
):
    """Mean-bin a native image and retain only complete source-clean bins.

    ``source_mask`` and ``target_mask`` are True for excluded native pixels.
    A centered integer crop discards incomplete edge bins (2000 pixels become
    666 bins using native [1:1999]). The field center uses native (x, y) pixels;
    output center and WCS account for bin centers and FFT-canvas placement.
    Masks must be prepared before injection; no source detection runs here.
    ``PSF`` is the delivered PSF on the resulting working grid.
    """
    image = np.asarray(image, dtype=float)
    source_mask = np.asarray(source_mask, bool)
    target_mask = np.asarray(target_mask, bool)
    if (
        image.ndim != 2
        or image.shape != source_mask.shape
        or image.shape != target_mask.shape
    ):
        raise ValueError("Native image and masks must share a grid")
    if (
        not isinstance(bin_factor, int)
        or bin_factor < 1
        or native_pixel_scale <= 0
        or not 0 < axis_ratio <= 1
    ):
        raise ValueError("Invalid binning, pixel scale or axis ratio")
    source_mask = source_mask | ~np.isfinite(image)
    ny, nx = (v // bin_factor for v in image.shape)
    if min(ny, nx) < 1 or max(ny, nx) > canvas_size:
        raise ValueError("Binned image does not fit the requested canvas")
    sy, sx = ((v - n * bin_factor) // 2 for v, n in zip(image.shape, (ny, nx)))
    native_crop = (slice(sy, sy + ny * bin_factor), slice(sx, sx + nx * bin_factor))

    def blocks(value):
        return value[native_crop].reshape(ny, bin_factor, nx, bin_factor)

    binned = blocks(image).mean((1, 3))
    clean = ~blocks(source_mask).any((1, 3))
    retained = ~blocks(source_mask | target_mask).any((1, 3))
    py, px = ((canvas_size - n) // 2 for n in (ny, nx))
    crop = (slice(py, py + ny), slice(px, px + nx))
    canvas = np.zeros((canvas_size, canvas_size))
    donor = np.zeros(canvas.shape, bool)
    valid = donor.copy()
    canvas[crop], donor[crop], valid[crop] = binned, retained, clean
    center = (
        (center_xy[0] - sx - (bin_factor - 1) / 2) / bin_factor,
        (center_xy[1] - sy - (bin_factor - 1) / 2) / bin_factor,
    )
    y, x = np.indices((ny, nx))
    angle = np.deg2rad(PA_deg)
    u = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle)
    v = (
        -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
    ) / axis_ratio
    pixel = native_pixel_scale * bin_factor
    radius = np.full(canvas.shape, np.inf)
    radius[crop] = pixel * np.hypot(u, v)
    working_WCS = None
    if WCS is not None:
        working_WCS = WCS.slice(
            (
                slice(sy, sy + ny * bin_factor, bin_factor),
                slice(sx, sx + nx * bin_factor, bin_factor),
            )
        )
        working_WCS.wcs.crpix += [px, py]
    return FieldData(
        canvas,
        donor,
        valid,
        crop,
        center,
        radius,
        PSF,
        pixel,
        exclusion_radius,
        working_WCS,
        dict(
            native_pixel_scale=native_pixel_scale,
            bin_factor=bin_factor,
            native_crop=[[s.start, s.stop] for s in native_crop],
        ),
    )
