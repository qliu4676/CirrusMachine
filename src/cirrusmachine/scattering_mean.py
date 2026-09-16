"""Differentiable first- and second-order scattering means.

This module uses the Morlet filters from the bundled Cheng et al. ``scattering``
package, but exposes explicit coefficient selection and boundary-aware spatial
averages.  The latter are needed when a small texture exemplar is not periodic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as torch_f


@dataclass(frozen=True, order=True)
class MeanFeature:
    """One isotropic scattering-mean coefficient."""

    order: int
    j1: int
    j2: int = -1
    delta_l: int = -1

    @property
    def label(self) -> str:
        if self.order == 1:
            return f"S1(j={self.j1})"
        return f"S2(j1={self.j1},j2={self.j2},dl={self.delta_l})"


@dataclass(frozen=True, order=True)
class FullMeanFeature:
    """One orientation-resolved scattering-mean coefficient."""

    order: int
    j1: int
    j2: int = -1
    l1: int = -1
    l2: int = -1

    @property
    def label(self) -> str:
        if self.order == 1:
            return f"S1(j={self.j1},l={self.l1})"
        return f"S2(j1={self.j1},j2={self.j2},l1={self.l1},l2={self.l2})"


def largest_dyadic_j(shape: tuple[int, int]) -> int:
    """Return the largest integer ``J`` satisfying ``2**J < min(shape)``."""

    minimum = int(min(shape))
    if minimum <= 2:
        raise ValueError("Both image dimensions must exceed two pixels.")
    return int(math.ceil(math.log2(minimum)) - 1)


def isotropic_feature_set(
    first_order_scales: Iterable[int],
    second_order_pairs: Iterable[tuple[int, int]],
    L: int,
) -> tuple[MeanFeature, ...]:
    """Return lexicographically ordered isotropic ``S1`` and ``S2`` features."""

    features = [MeanFeature(1, int(j)) for j in sorted(set(first_order_scales))]
    for j1, j2 in sorted(set(second_order_pairs)):
        if j2 <= j1:
            raise ValueError(f"Second-order paths require j2>j1, got {(j1, j2)}")
        features.extend(MeanFeature(2, int(j1), int(j2), dl) for dl in range(L))
    return tuple(features)


def within_band_pairs(scales: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return all strict ``j2>j1`` pairs within a collection of scales."""

    ordered = sorted(set(int(j) for j in scales))
    return tuple((j1, j2) for j1 in ordered for j2 in ordered if j2 > j1)


def full_feature_set(
    first_order_scales: Iterable[int],
    second_order_pairs: Iterable[tuple[int, int]],
    L: int,
) -> tuple[FullMeanFeature, ...]:
    """Return lexicographically ordered full-orientation ``S1`` and ``S2`` features."""

    features = [
        FullMeanFeature(1, int(j), l1=l)
        for j in sorted(set(first_order_scales))
        for l in range(L)
    ]
    for j1, j2 in sorted(set(second_order_pairs)):
        if j2 <= j1:
            raise ValueError(f"Second-order paths require j2>j1, got {(j1, j2)}")
        features.extend(
            FullMeanFeature(2, int(j1), int(j2), l1, l2)
            for l1 in range(L)
            for l2 in range(L)
        )
    return tuple(features)


def resolve_torch_device(requested: str = "gpu") -> torch.device:
    """Resolve CUDA/MPS when available, otherwise return CPU."""

    name = requested.lower()
    if name in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"gpu", "mps"} and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class ScatteringMean2D:
    """Compute selected Cheng-style ``s_mean_iso`` coefficients.

    Parameters
    ----------
    shape
        Unpadded image shape.
    J, L
        Number of dyadic scales and Morlet orientations.
    pad_pixels
        Reflection padding used only as convolution support. Spatial means are
        always evaluated inside the original footprint.
    mask_boundary
        If true, exclude a scale-dependent rim from the spatial mean. This is
        intended for small exemplars whose surroundings are unknown.
    boundary_sigma
        Rim width in units of the Morlet Gaussian-envelope sigma.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        J: int,
        L: int = 4,
        *,
        pad_pixels: int = 0,
        mask_boundary: bool = False,
        boundary_sigma: float = 2.0,
        device: str = "cpu",
    ) -> None:
        self.shape = tuple(map(int, shape))
        self.J = int(J)
        self.L = int(L)
        self.pad_pixels = int(pad_pixels)
        self.mask_boundary = bool(mask_boundary)
        self.boundary_sigma = float(boundary_sigma)
        self.device = resolve_torch_device(device)
        if self.pad_pixels >= min(self.shape):
            raise ValueError("Reflection padding must be smaller than each image dimension.")

        from scattering import Scattering2d  # imported lazily

        padded_shape = tuple(size + 2 * self.pad_pixels for size in self.shape)
        calculator = Scattering2d(
            *padded_shape,
            self.J,
            self.L,
            device="cpu",
            wavelets="morlet",
            precision="single",
        )
        self.filters = calculator.filters_set.to(self.device)
        self.padded_shape = padded_shape

    def _prepare(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value[None]
        if value.ndim != 3 or tuple(value.shape[-2:]) != self.shape:
            raise ValueError(f"Expected (..., {self.shape[0]}, {self.shape[1]}), got {tuple(value.shape)}")
        if self.pad_pixels:
            p = self.pad_pixels
            value = torch_f.pad(value[:, None], (p, p, p, p), mode="reflect")[:, 0]
        return value

    def _weight(self, j: int) -> torch.Tensor:
        p = self.pad_pixels
        margin = 0
        if self.mask_boundary:
            sigma = 0.8 * 2**int(j)
            margin = int(math.ceil(self.boundary_sigma * sigma))
        ny, nx = self.shape
        if 2 * margin >= min(ny, nx):
            raise ValueError(
                f"Scale j={j} has no valid interior for shape {self.shape}; "
                "reduce J or the boundary exclusion."
            )
        weight = torch.zeros(self.padded_shape, dtype=torch.float32, device=self.device)
        weight[p + margin : p + ny - margin, p + margin : p + nx - margin] = 1.0
        return weight / weight.sum()

    def log_features(
        self,
        image: np.ndarray | torch.Tensor,
        features: Iterable[MeanFeature],
    ) -> torch.Tensor:
        """Return selected log ``s_mean_iso`` values in feature order."""

        requested = tuple(features)
        if not requested:
            raise ValueError("At least one scattering feature is required.")
        for feature in requested:
            if not 0 <= feature.j1 < self.J:
                raise ValueError(f"j1={feature.j1} lies outside J={self.J}")
            if feature.order == 2:
                if not feature.j1 < feature.j2 < self.J:
                    raise ValueError(f"Invalid strict path {feature}")
                if not 0 <= feature.delta_l < self.L:
                    raise ValueError(f"Invalid relative orientation {feature.delta_l}")

        data = self._prepare(image)
        data_f = torch.fft.fftn(data, dim=(-2, -1))
        needed_j1 = sorted({f.j1 for f in requested})
        i1: dict[int, torch.Tensor] = {}
        i1_f: dict[int, torch.Tensor] = {}
        for j in needed_j1:
            response = torch.fft.ifftn(
                data_f[:, None] * self.filters[j][None], dim=(-2, -1)
            ).abs()
            i1[j] = response
            if any(f.order == 2 and f.j1 == j for f in requested):
                i1_f[j] = torch.fft.fftn(response, dim=(-2, -1))

        s1: dict[int, torch.Tensor] = {}
        for j in {f.j1 for f in requested if f.order == 1}:
            weight = self._weight(j)
            s1[j] = (i1[j] * weight[None, None]).sum((-2, -1)).mean(-1)

        s2: dict[tuple[int, int, int], torch.Tensor] = {}
        for j1, j2 in sorted({(f.j1, f.j2) for f in requested if f.order == 2}):
            delta_l = sorted(
                {
                    feature.delta_l
                    for feature in requested
                    if feature.order == 2
                    and feature.j1 == j1
                    and feature.j2 == j2
                }
            )
            indices = torch.arange(self.L, device=self.device)
            selected_filters = torch.stack(
                [
                    self.filters[j2][(indices + dl) % self.L]
                    for dl in delta_l
                ],
                dim=1,
            )
            response = torch.fft.ifftn(
                i1_f[j1][:, :, None] * selected_filters[None],
                dim=(-2, -1),
            ).abs()
            weight = self._weight(j2)
            oriented = (response * weight[None, None, None]).sum((-2, -1))
            relative = oriented.mean(dim=1)
            for column, dl in enumerate(delta_l):
                s2[(j1, j2, dl)] = relative[:, column]

        tiny = torch.finfo(data.dtype).tiny
        values = []
        for feature in requested:
            if feature.order == 1:
                value = s1[feature.j1]
            else:
                value = s2[(feature.j1, feature.j2, feature.delta_l)]
            values.append(torch.log(value.clamp_min(tiny)))
        return torch.stack(values, dim=-1)

    def all_isotropic(self, image: np.ndarray | torch.Tensor) -> tuple[tuple[MeanFeature, ...], np.ndarray]:
        """Return all finite strict-path isotropic coefficients for diagnostics."""

        features: list[MeanFeature] = []
        for j in range(self.J):
            try:
                self._weight(j)
            except ValueError:
                continue
            features.append(MeanFeature(1, j))
        for j1, j2 in within_band_pairs(range(self.J)):
            try:
                self._weight(j2)
            except ValueError:
                continue
            features.extend(MeanFeature(2, j1, j2, dl) for dl in range(self.L))
        with torch.no_grad():
            values = self.log_features(image, features)[0].detach().cpu().numpy()
        return tuple(features), values

    def log_full_features(
        self,
        image: np.ndarray | torch.Tensor,
        features: Iterable[FullMeanFeature],
    ) -> torch.Tensor:
        """Return selected orientation-resolved log scattering means."""

        requested = tuple(features)
        if not requested:
            raise ValueError("At least one scattering feature is required.")
        for feature in requested:
            if not 0 <= feature.j1 < self.J or not 0 <= feature.l1 < self.L:
                raise ValueError(f"Invalid first wavelet index in {feature}")
            if feature.order == 2:
                if not feature.j1 < feature.j2 < self.J or not 0 <= feature.l2 < self.L:
                    raise ValueError(f"Invalid strict second-order path {feature}")

        data = self._prepare(image)
        data_f = torch.fft.fftn(data, dim=(-2, -1))
        needed_j1 = sorted({feature.j1 for feature in requested})
        i1: dict[int, torch.Tensor] = {}
        i1_f: dict[int, torch.Tensor] = {}
        for j in needed_j1:
            response = torch.fft.ifftn(
                data_f[:, None] * self.filters[j][None], dim=(-2, -1)
            ).abs()
            i1[j] = response
            if any(feature.order == 2 and feature.j1 == j for feature in requested):
                i1_f[j] = torch.fft.fftn(response, dim=(-2, -1))

        s1: dict[int, torch.Tensor] = {}
        for j in {feature.j1 for feature in requested if feature.order == 1}:
            weight = self._weight(j)
            s1[j] = (i1[j] * weight[None, None]).sum((-2, -1))

        s2: dict[tuple[int, int], torch.Tensor] = {}
        for j1, j2 in sorted({(f.j1, f.j2) for f in requested if f.order == 2}):
            response = torch.fft.ifftn(
                i1_f[j1][:, :, None] * self.filters[j2][None, None],
                dim=(-2, -1),
            ).abs()
            weight = self._weight(j2)
            s2[(j1, j2)] = (response * weight[None, None, None]).sum((-2, -1))

        tiny = torch.finfo(data.dtype).tiny
        values = []
        for feature in requested:
            if feature.order == 1:
                value = s1[feature.j1][:, feature.l1]
            else:
                value = s2[(feature.j1, feature.j2)][:, feature.l1, feature.l2]
            values.append(torch.log(value.clamp_min(tiny)))
        return torch.stack(values, dim=-1)
