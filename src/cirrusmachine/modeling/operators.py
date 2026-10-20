"""Torch numerical operators with the reference FFT and boundary conventions."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len


def to_tensor(value, *, dtype=None, device=None):
    """Accept native/FITS byte order and sliced NumPy arrays without losing dtype."""
    if isinstance(value, np.ndarray):
        if not value.dtype.isnative:
            value = value.astype(value.dtype.newbyteorder("="))
        if any(stride < 0 for stride in value.strides):
            value = np.ascontiguousarray(value)
    return torch.as_tensor(value, dtype=dtype, device=device)


def resolve_device(device="auto"):
    """Select a backend supporting float64 covariance solves and complex FFTs."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError(
            "Use CUDA or CPU; MPS lacks required float64 support. Use 'auto' for fallback."
        )
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use device='auto' for CPU fallback"
        )
    return result


def as_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def gaussian_filter(image, sigma, mode="reflect", *, truncate=4.0, radius=None):
    """Separable Gaussian matching SciPy's half-sample reflect/constant padding."""
    if sigma <= 0:
        return image.clone()
    radius = int(truncate * sigma + 0.5) if radius is None else radius
    t = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (t / sigma).square())
    kernel = kernel / kernel.sum()
    out = image
    # SciPy filters axis 0 first, then axis 1. Reflect includes the edge pixel.
    for axis in (0, 1):
        if mode == "reflect":
            n = image.shape[axis]
            idx = torch.arange(-radius, n + radius, device=image.device) % (2 * n)
            idx = torch.where(idx < n, idx, 2 * n - 1 - idx)
            padded = out.index_select(axis, idx)
        elif mode == "constant":
            padded = F.pad(
                out, (0, 0, radius, radius) if axis == 0 else (radius, radius, 0, 0)
            )
        else:
            raise ValueError("mode must be reflect or constant")
        k = kernel[None, None, :, None] if axis == 0 else kernel[None, None, None, :]
        out = F.conv2d(padded[None, None], k)[0, 0]
    return out


def smooth_observed(image, mask, sigma):
    """Normalized constant-boundary smoothing; invalid samples never enter."""
    coverage = gaussian_filter(mask.to(torch.float64), sigma, "constant")
    value = gaussian_filter(torch.where(mask, image, 0), sigma, "constant")
    return value / coverage.clamp_min(1e-8), coverage


def fft_convolve(image, kernel):
    """Linear convolution, same-size crop; no periodic wrap at image edges."""
    shape = tuple(next_fast_len(a + b - 1) for a, b in zip(image.shape, kernel.shape))
    full = torch.fft.irfft2(
        torch.fft.rfft2(image, s=shape) * torch.fft.rfft2(kernel, s=shape), s=shape
    )
    y, x = ((n - 1) // 2 for n in kernel.shape)
    return full[y : y + image.shape[0], x : x + image.shape[1]]


def noise_bases():
    """Five unit-variance compact filters used by the saved noise model."""
    yy, xx = np.indices((7, 7)) - 3
    kernels = []
    for sy, sx in [(0, 0), (0.55, 0.55), (1.0, 1.0), (0.55, 1.0), (1.0, 0.55)]:
        k = (
            np.zeros((7, 7))
            if sx == 0
            else np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
        )
        if sx == 0:
            k[3, 3] = 1
        kernels.append(k / np.sqrt((k * k).sum()))
    return np.array(kernels)


def covariance_kernel(kernels, variances):
    from scipy.signal import correlate2d

    return sum(v * correlate2d(k, k, mode="full") for k, v in zip(kernels, variances))


def draw_noise(shape, kernels, variances, seed, *, device="cpu", dtype=torch.float64):
    """Draw compact correlated noise with seeded, device-independent random samples.

    NumPy's PCG64 Generator creates the random samples on CPU.
    PyTorch convolves them on the selected device.
    """
    rng = np.random.default_rng(seed)
    pad = kernels.shape[-1]
    out = torch.zeros(tuple(shape), dtype=dtype, device=device)
    for kernel, variance in zip(kernels, variances):
        white = to_tensor(
            rng.normal(size=(shape[0] + 2 * pad, shape[1] + 2 * pad)),
            dtype=dtype,
            device=device,
        )
        k = to_tensor(kernel, dtype=dtype, device=device)
        out += (
            np.sqrt(max(float(variance), 0))
            * fft_convolve(white, k)[pad:-pad, pad:-pad]
        )
    return out


def PSF_transfer(shape, kernel, *, device="cpu"):
    """Transfer of the supplied delivered-image PSF, applied once to intrinsic texture."""
    kernel = to_tensor(kernel, dtype=torch.float64, device=device)
    if kernel.ndim != 2 or any(k > n for k, n in zip(kernel.shape, shape)):
        raise ValueError("PSF must fit the canvas")
    if not torch.isfinite(kernel).all() or not torch.isclose(
        kernel.sum(), kernel.new_tensor(1.0), atol=1e-5
    ):
        raise ValueError("PSF must be finite and normalized to unit sum")
    canvas = torch.zeros(shape, dtype=kernel.dtype, device=device)
    ny, nx = kernel.shape
    canvas[:ny, :nx] = kernel
    return torch.fft.rfft2(torch.roll(canvas, (-(ny // 2), -(nx // 2)), (0, 1)))


def power_field(shape, beta, seed, PSF, theta=0.0, coherence=0.4, *, device="cpu"):
    """PSF-filtered Gaussian field with P(k) proportional to k**(-beta)."""
    fy = torch.fft.fftfreq(shape[0], dtype=torch.float64, device=device)[:, None]
    fx = torch.fft.rfftfreq(shape[1], dtype=torch.float64, device=device)[None, :]
    rad, phi = torch.hypot(fy, fx), torch.atan2(fy, fx)
    amp = rad.clamp_min(1 / max(shape)).pow(-float(beta) / 2)
    amp[0, 0] = 0
    amp *= (1 + coherence * torch.cos(2 * (phi - theta))).clamp_min(0.05).sqrt()
    white = to_tensor(np.random.default_rng(seed).normal(size=shape), device=device)
    z = torch.fft.irfft2(
        torch.fft.rfft2(white) * amp * PSF_transfer(shape, PSF, device=device), s=shape
    )
    return z / z.std(correction=0)


class MaskedCovariance:
    """Restricted C_OO with linear covariance application and implicit CG gradient.

    The kernel is a stationary covariance in normalized intensity squared;
    extra_variance adds only a diagonal on retained pixels. Zero padding
    implements Toeplitz convolution, never periodic residual whitening.
    """

    def __init__(
        self, mask, kernel, extra_variance=None, rtol=1e-06, maxiter=400, device="cpu"
    ):
        self.device = torch.device(device)
        self.mask = to_tensor(mask, dtype=torch.bool, device=self.device)
        self.kernel = to_tensor(kernel, dtype=torch.float64, device=self.device)[
            None, None
        ]
        self.radius = kernel.shape[0] // 2
        self.extra = (
            torch.zeros(int(self.mask.sum()), dtype=torch.float64, device=self.device)
            if extra_variance is None
            else to_tensor(
                np.asarray(extra_variance)[np.asarray(mask)],
                dtype=torch.float64,
                device=self.device,
            )
        )
        self.diag = self.kernel[0, 0, self.radius, self.radius] + self.extra
        self.rtol = rtol
        self.maxiter = maxiter
        self.last = {}
        if (
            not self.mask.any()
            or kernel.ndim != 2
            or kernel.shape[0] != kernel.shape[1]
            or (kernel.shape[0] % 2 != 1)
        ):
            raise ValueError(
                "Covariance requires nonempty support and an odd square kernel"
            )
        if not torch.isfinite(self.diag).all() or (self.diag <= 0).any():
            raise ValueError("Covariance diagonal must be finite and positive")
        self.fft_shape = tuple(
            (next_fast_len(n + kernel.shape[i] - 1) for i, n in enumerate(mask.shape))
        )
        self.kernel_fft = torch.fft.rfft2(
            self.kernel[0, 0].flip((0, 1)), s=self.fft_shape
        )

    def apply(self, v):
        full = torch.zeros(self.mask.shape, dtype=torch.float64, device=self.device)
        full[self.mask] = v.double()
        linear = torch.fft.irfft2(
            torch.fft.rfft2(full, s=self.fft_shape) * self.kernel_fft, s=self.fft_shape
        )
        h, w = self.mask.shape
        r = self.radius
        out = linear[r : r + h, r : r + w]
        return out[self.mask] + self.extra * v

    @torch.no_grad()
    def solve(self, rhs):
        b = rhs.double()
        x = torch.zeros_like(b)
        r = b.clone()
        z = r / self.diag
        p = z.clone()
        rz = torch.dot(r, z)
        norm = torch.linalg.vector_norm(b).clamp_min(1e-30)
        if float(norm) < 1e-25:
            self.last = {"iterations": 0, "relative_residual": 0.0}
            return x
        for k in range(self.maxiter):
            ap = self.apply(p)
            alpha = rz / torch.dot(p, ap).clamp_min(1e-30)
            x += alpha * p
            r -= alpha * ap
            rel = torch.linalg.vector_norm(r) / norm
            if rel <= self.rtol:
                break
            z = r / self.diag
            new = torch.dot(r, z)
            p = z + new / rz * p
            rz = new
        self.last = {"iterations": k + 1, "relative_residual": float(rel)}
        if rel > self.rtol * 2:
            raise RuntimeError(f"Covariance solve failed: {self.last}")
        return x

    def quadratic(self, residual):
        return _Quadratic.apply(residual[self.mask], self)


class _Quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, operator):
        solved = operator.solve(r)
        ctx.save_for_backward(solved)
        ctx.dtype = r.dtype
        return (torch.dot(r.double(), solved) / r.numel()).to(r.dtype)

    @staticmethod
    def backward(ctx, grad):
        (solved,) = ctx.saved_tensors
        return ((grad * 2 * solved / solved.numel()).to(ctx.dtype), None)


def beam_loss(c, sample_grid, target, precision_cholesky, sigma):
    """Compare a latent image with native coarse samples at a common beam.

    All coordinates are in the real padded field. No map wrapping or filled
    guide values. Calibration/discrepancy covariance is supplied separately.
    """
    r = int(np.ceil(4 * sigma))
    t = torch.arange(-r, r + 1, dtype=c.dtype, device=c.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    sm = F.conv2d(c[None, None], k[None, None, None, :], padding=(0, r))
    sm = F.conv2d(sm, k[None, None, :, None], padding=(r, 0))
    sample = F.grid_sample(sm, sample_grid, align_corners=False)[0, 0, :, 0]
    residual = (sample - target).double()
    white = torch.linalg.solve_triangular(
        precision_cholesky, residual[:, None], upper=False
    )[:, 0]
    return white.square().mean().to(c.dtype)
