"""Selected isotropic scattering-covariance constraints.

The wrapper follows the bundled Cheng et al. coefficient vector and index
conventions while making the selected coefficient types explicit. It omits the
mean/std and duplicated S1 entries for additive covariance experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .scattering_mean import resolve_torch_device


@dataclass(frozen=True, order=True)
class CovarianceFeature:
    """One entry in the bundled isotropic scattering-covariance vector."""

    index_type: int
    j1: int
    j2: int = -1
    j3: int = -1
    l2: int = -1
    l3: int = -1


def _descriptor(column: torch.Tensor) -> CovarianceFeature:
    # Rows are index_type, j1, j2, j3, l1, l2, l3. Isotropic l1 is -1.
    values = [int(value) for value in column.tolist()]
    return CovarianceFeature(values[0], values[1], values[2], values[3], values[5], values[6])


def covariance_feature_set(J: int, L: int, scales: Iterable[int]) -> tuple[CovarianceFeature, ...]:
    """Select P00, C01, and C11 terms wholly contained in one scale band."""

    from scattering.Scattering2d import get_scattering_index

    allowed = set(int(j) for j in scales)
    index = get_scattering_index(J, L, "P00", "j2>=j1")["index_for_synthesis_iso"]
    selected: list[CovarianceFeature] = []
    for column in index.T:
        feature = _descriptor(column)
        if feature.index_type not in {1, 3, 5}:
            continue
        involved = {j for j in (feature.j1, feature.j2, feature.j3) if j >= 0}
        if involved and involved.issubset(allowed):
            selected.append(feature)
    return tuple(selected)


class ScatteringCovariance2D:
    """Differentiable selected ``s_cov_iso`` features on a fixed image shape."""

    def __init__(
        self,
        shape: tuple[int, int],
        J: int,
        L: int = 4,
        *,
        device: str = "gpu",
        remove_edge: bool = True,
    ) -> None:
        from scattering import Scattering2d
        from scattering.Scattering2d import get_scattering_index

        self.shape = tuple(map(int, shape))
        self.J = int(J)
        self.L = int(L)
        resolved = resolve_torch_device(device)
        # The bundled calculator supports CUDA or CPU, but not MPS.
        self.device = resolved if resolved.type == "cuda" else torch.device("cpu")
        backend_device = "gpu" if self.device.type == "cuda" else "cpu"
        self.calculator = Scattering2d(*self.shape, self.J, self.L, device=backend_device)
        self.remove_edge = bool(remove_edge)
        self.use_reference = False
        index = get_scattering_index(
            self.J, self.L, "P00", "j2>=j1"
        )["index_for_synthesis_iso"]
        self.positions = {_descriptor(column): position for position, column in enumerate(index.T)}

    def p00(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return orientation-resolved wavelet power for reference normalization."""

        value = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value[None]
        return self.calculator.scattering_cov(
            value,
            if_large_batch=False,
            C11_criteria="j2>=j1",
            use_ref=False,
            normalization="P00",
            remove_edge=self.remove_edge,
        )["P00"]

    def set_reference_p00(self, reference_p00: np.ndarray | torch.Tensor) -> None:
        """Set a fixed P00 denominator used by C01 and C11 coefficients."""

        value = torch.as_tensor(reference_p00, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value[None]
        if tuple(value.shape[-2:]) != (self.J, self.L):
            raise ValueError(f"Expected reference P00 ending in {(self.J, self.L)}, got {tuple(value.shape)}")
        self.calculator.ref_scattering_cov = {"P00": value}
        self.use_reference = True

    def features(
        self,
        image: np.ndarray | torch.Tensor,
        requested: tuple[CovarianceFeature, ...],
    ) -> torch.Tensor:
        """Return selected values in the requested descriptor order."""

        value = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value[None]
        if value.ndim != 3 or tuple(value.shape[-2:]) != self.shape:
            raise ValueError(f"Expected image stack ending in {self.shape}, got {tuple(value.shape)}")
        result = self.calculator.scattering_cov(
            value,
            if_large_batch=False,
            C11_criteria="j2>=j1",
            use_ref=self.use_reference,
            normalization="P00",
            remove_edge=self.remove_edge,
        )["for_synthesis_iso"]
        try:
            selection = [self.positions[feature] for feature in requested]
        except KeyError as exc:
            raise ValueError(f"Covariance feature is unavailable for J={self.J}: {exc}") from exc
        selected = result[:, selection]
        if not torch.isfinite(selected).all():
            raise ValueError("Selected covariance vector contains non-finite values.")
        return selected
