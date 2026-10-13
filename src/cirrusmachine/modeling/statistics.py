"""Existing Morlet S1/S2/P00 convention, plus approximate masked-input targets.
Supplied pixels can include separately labeled median-filled estimates.

The target transform removes the mask response to a local constant and divides
by positive-kernel coverage. This is NOT an exact missing-data WST estimator.
Synthetic known-field calibration and separate bias checks are mandatory.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from scattering import Scattering2d
from scattering.Scattering2d import cut_high_k_off

from .operators import draw_noise, gaussian_filter, resolve_device


class _WaveletMeasure:
    def __init__(self, config, known, device="cpu"):
        self.device = torch.device(device)
        self.c = config
        self.N = known.shape[0]
        self.C = config["fft_canvas"]
        self.P = config["padding"]
        self.J = config["scales"]
        self.L = config["orientations"]
        self.pairs = [(j, k) for j in self.J for k in self.J if k > j]
        self.cal = Scattering2d(self.C, self.C, max(self.J) + 1, self.L, device="cpu")
        self.filters = {
            j: cut_high_k_off(self.cal.filters_set[j], *self.cal.get_dxdy(j))
            .contiguous()
            .to(self.device)
            for j in self.J
        }
        self.cal.filters_set = None
        self.cal.edge_masks = None
        self.cal.edge_masks_f = None
        self.g = {}
        self.window = {}
        self.observed = {}
        self.cov = {}
        self.maskresponse = {}
        self.known = torch.tensor(known, dtype=torch.float32, device=self.device)
        mask = F.pad(self.known[None, None], (self.P,) * 4, mode="reflect")[0, 0]
        self.mf = torch.fft.fft2(mask)
        canvas = F.pad(
            torch.ones((1, 1, self.N, self.N), device=self.device), (self.P,) * 4
        )
        for j in self.J:
            filt = self.filters[j]
            n = filt.shape[-1]
            factor = n * n / self.C**2
            kernel = torch.fft.ifft2(filt).abs().mean(0)
            kernel /= kernel.sum()
            self.g[j] = torch.fft.fft2(kernel)
            self.window[j] = F.adaptive_avg_pool2d(canvas, (n, n))[0, 0]
            self.observed[j] = F.adaptive_avg_pool2d(
                F.pad(self.known[None, None], (self.P,) * 4), (n, n)
            )[0, 0]
            fm = cut_high_k_off(self.mf, *self.cal.get_dxdy(j))
            self.cov[j] = (torch.fft.ifft2(fm * self.g[j]).real * factor).clamp(0, 1)
            self.maskresponse[j] = torch.fft.ifft2(fm[None] * filt) * factor
        self.support = {
            j: (self.cov[j] >= config["minimum_kernel_coverage"]).float()
            for j in self.J
        }
        self.target_weight = {j: self.support[j] * self.observed[j] for j in self.J}
        self.full_weight = {j: self.window[j] / self.window[j].sum() for j in self.J}
        coords = np.linspace(10, self.N - 11, 33)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        self.points = np.stack([xx.ravel(), yy.ravel()], 1)
        self.grid = (
            2
            * (
                torch.tensor(self.points, dtype=torch.float32, device=self.device)
                + self.P
                + 0.5
            )
            / self.C
            - 1
        )[None, :, None, :]
        n = 128
        freq = torch.fft.fftfreq(n, device=self.device)
        sigma = (
            config["orientation_pool_fwhm_arcsec"]
            / config["pixel_scale"]
            / 2.354820045
            / (self.C / n)
        )
        self.pool = torch.exp(
            -2 * np.pi**2 * sigma**2 * (freq[:, None] ** 2 + freq[None, :] ** 2)
        )

    def response(self, xf, j, masked=False):
        n = self.filters[j].shape[-1]
        factor = n * n / self.C**2
        fx = cut_high_k_off(xf, *self.cal.get_dxdy(j))
        z = torch.fft.ifft2(fx[None] * self.filters[j]) * factor
        if masked:
            coverage = self.cov[j]
            mean = (
                torch.fft.ifft2(fx * self.g[j]).real * factor / coverage.clamp_min(0.05)
            )
            z = (z - self.maskresponse[j] * mean[None]) / coverage.clamp_min(0.05)[None]
        return z

    def second(self, z, j, k, masked=False):
        n = z.shape[-1]
        nk = self.filters[k].shape[-1]
        factor = nk * nk / (n * n)
        if not masked:
            return (
                torch.fft.ifft2(
                    cut_high_k_off(torch.fft.fft2(z), *self.cal.get_dxdy(k))[None]
                    * self.filters[k]
                ).abs()
                * factor
            )
        weight = self.support[j]
        zf = cut_high_k_off(torch.fft.fft2(z * weight), *self.cal.get_dxdy(k))
        wf = cut_high_k_off(torch.fft.fft2(weight), *self.cal.get_dxdy(k))
        coverage = (torch.fft.ifft2(wf * self.g[k]).real * factor).clamp(0, 1)
        mean = torch.fft.ifft2(zf * self.g[k]).real * factor / coverage.clamp_min(0.05)
        out = torch.fft.ifft2(zf[None] * self.filters[k]) * factor
        dc = torch.fft.ifft2(wf[None] * self.filters[k]) * factor
        return (
            ((out - dc * mean[None]) / coverage.clamp_min(0.05)[None]).abs(),
            coverage,
        )

    def block(self, xf, j, orientation):
        n = self.filters[j].shape[-1]
        z = torch.fft.ifft2(
            cut_high_k_off(xf, *self.cal.get_dxdy(j)) * self.filters[j][orientation]
        ).abs() * (n * n / self.C**2)
        w = self.full_weight[j]
        result = [(z * w).sum().reshape(1), (z.square() * w).sum().reshape(1)]
        for k in self.J:
            if k <= j:
                continue
            result.append((self.second(z, j, k) * self.full_weight[k]).sum((-2, -1)))
        return torch.cat(result)

    def full(self, x):
        xp = F.pad(x[None, None], (self.P,) * 4, mode="reflect")[0, 0]
        xf = torch.fft.fft2(xp)
        s1 = []
        p00 = []
        s2 = []
        for j in self.J:
            rows = []
            for orientation in range(self.L):

                def fn(v, j=j, orientation=orientation):
                    return self.block(v, j, orientation)

                rows.append(
                    checkpoint(fn, xf, use_reentrant=False)
                    if x.requires_grad
                    else fn(xf)
                )
            rows = torch.stack(rows)
            s1.append(rows[:, 0])
            p00.append(rows[:, 1])
            for idx, k in enumerate([k for k in self.J if k > j]):
                s2.append(rows[:, 2 + idx * self.L : 2 + (idx + 1) * self.L])
        return (torch.stack(s1), torch.stack(s2), torch.stack(p00))

    @torch.no_grad()
    def approximate(self, x):
        xp = F.pad((x * self.known)[None, None], (self.P,) * 4, mode="reflect")[0, 0]
        xf = torch.fft.fft2(xp)
        s1 = []
        s2 = []
        p00 = []
        for j in self.J:
            z = self.response(xf, j, True).abs()
            w = self.target_weight[j]
            w = w / w.sum()
            s1.append((z * w).sum((-2, -1)))
            p00.append((z.square() * w).sum((-2, -1)))
            for k in self.J:
                if k <= j:
                    continue
                rows = []
                for orientation in range(self.L):
                    out, cov = self.second(z[orientation], j, k, True)
                    weight = (
                        self.observed[k]
                        * (cov >= self.c["minimum_kernel_coverage"])
                        * self.support[k]
                    )
                    weight /= weight.sum()
                    rows.append((out * weight).sum((-2, -1)))
                s2.append(torch.stack(rows))
        return (torch.stack(s1), torch.stack(s2), torch.stack(p00))


"""Padded real-context WST, dense CDF and orientation estimators.
Candidate statistics and observed estimators use the same central footprint.
"""


class Statistics(_WaveletMeasure):
    def __init__(self, donor, roi, classes, config):
        c = dict(
            fft_canvas=donor.shape[0],
            padding=0,
            scales=list(range(5)),
            orientations=config.orientations,
            minimum_kernel_coverage=config.minimum_kernel_coverage,
            orientation_pool_fwhm_arcsec=config.orientation_pool_fwhm,
            pixel_scale=config.pixel_scale,
        )
        super().__init__(c, donor, resolve_device(config.device))
        self.roi = torch.tensor(roi, device=self.device)
        self.mask_np = donor
        self.roi_np = roi
        for j in self.J:
            n = self.filters[j].shape[-1]
            self.window[j] = F.adaptive_avg_pool2d(
                self.roi.float()[None, None], (n, n)
            )[0, 0]
            self.full_weight[j] = self.window[j] / self.window[j].sum()
            self.observed[j] = F.adaptive_avg_pool2d(
                (self.known * self.roi)[None, None], (n, n)
            )[0, 0]
            self.target_weight[j] = self.support[j] * self.observed[j]
        self.classes = (
            torch.zeros(donor.shape, dtype=torch.long, device=self.device)
            if classes is None
            else torch.tensor(classes, dtype=torch.long, device=self.device)
        )
        self.quantiles = torch.linspace(0.01, 0.99, 199, device=self.device)
        self.cdf_sigma = config.cdf_fwhm / config.pixel_scale / 2.354820045
        self.cdf_cov = gaussian_filter(
            torch.tensor(donor, dtype=torch.float64, device=self.device),
            self.cdf_sigma,
            mode="constant",
        ).float()

    def lowpass(self, x):
        sig = self.cdf_sigma
        r = int(np.ceil(4 * sig))
        t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-0.5 * (t / sig) ** 2)
        k /= k.sum()
        z = F.conv2d(x[None, None], k[None, None, None, :], padding=(0, r))
        return F.conv2d(z, k[None, None, :, None], padding=(r, 0))[0, 0]

    def cdf(self, x, masked=False):
        sm = (
            self.lowpass(x * self.known) / self.cdf_cov.clamp_min(0.05)
            if masked
            else self.lowpass(x)
        )
        valid = (
            self.roi & self.known.bool() & (self.cdf_cov > 0.55) if masked else self.roi
        )
        return torch.stack(
            [
                torch.quantile((x - sm)[valid & (self.classes == k)], self.quantiles)
                for k in range(3)
            ]
        )

    def angular_energy(self, x, masked=False):
        xf = torch.fft.fft2(x * self.known if masked else x)
        result = []
        for j in self.J:
            energy = self.response(xf, j, masked).abs().square()
            v = self.target_weight[j] if masked else self.window[j]
            pooled = F.adaptive_avg_pool2d((energy * v)[None], (128, 128))[0]
            vv = F.adaptive_avg_pool2d(v[None, None], (128, 128))[0, 0]
            pooled = torch.fft.ifft2(torch.fft.fft2(pooled) * self.pool).real.clamp_min(
                0
            )
            vv = torch.fft.ifft2(torch.fft.fft2(vv) * self.pool).real.clamp_min(0)
            pooled = F.grid_sample(pooled[None], self.grid, align_corners=False)[
                0, :, :, 0
            ]
            vv = F.grid_sample(vv[None, None], self.grid, align_corners=False)[
                0, 0, :, 0
            ]
            result.append(pooled / vv.clamp_min(0.05))
        return torch.stack(result)

    def all(self, x, masked=False):
        return [
            *(self.approximate(x) if masked else self.full(x)),
            self.cdf(x, masked),
            self.angular_energy(x, masked),
        ]


@torch.no_grad()
def independent_targets(
    measure,
    observed,
    proxy,
    kernels,
    variances,
    seeds,
    mask_factors,
    mask_offsets,
    mask_sigmas,
):
    """Fixed independent mask calibration plus online MASKED noise correction.

    No full statistic of the evolving candidate appears in the target. If two
    proxies agree on observed pixels, they yield exactly the same target for
    common noise draws, regardless of their missing-region reconstructions.
    """
    observed = torch.where(measure.known.bool(), observed, 0)
    proxy = torch.where(measure.known.bool(), proxy, 0)
    raw = measure.all(observed, True)
    clean_masked = measure.all(proxy, True)
    simulations = [
        measure.all(
            proxy
            + draw_noise(proxy.shape, kernels, variances, s, device=proxy.device).to(
                proxy.dtype
            ),
            True,
        )
        for s in seeds
    ]
    targets = []
    uncertainty = []
    for j, (r, c, f, b, u) in enumerate(
        zip(raw, clean_masked, mask_factors, mask_offsets, mask_sigmas)
    ):
        noise_delta = torch.stack([z[j] - c for z in simulations])
        targets.append(f * (r - noise_delta.mean(0)) + b)
        uncertainty.append(torch.sqrt((f * noise_delta.std(0)) ** 2 + u**2))
    return (targets, uncertainty)
