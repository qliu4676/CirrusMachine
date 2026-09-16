"""Coordinate-based multiresolution latent-grid neural field."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_f


class MultiResolutionLatentField(torch.nn.Module):
    """Continuous residual field sampled from learned grids and a small MLP.

    Coordinates are ``(x, y)`` values in ``[-1, 1]``.  The model contains no
    convolutional generator; bilinear grid sampling only interpolates learned
    coordinate embeddings before point-wise decoding.
    """

    def __init__(
        self,
        grid_resolutions: tuple[int, ...] = (16, 32, 64, 128),
        channels_per_level: int = 4,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        output_scale: float = 1.0,
        include_coordinates: bool = True,
        rotate_and_offset_grids: bool = False,
        grid_frame_seed: int = 0,
        fourier_feature_count: int = 0,
        fourier_min_cycles: float = 1.0,
        fourier_max_cycles: float = 1.0,
        fourier_seed: int = 0,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if not grid_resolutions or any(value < 2 for value in grid_resolutions):
            raise ValueError("grid_resolutions must contain values >= 2")
        if channels_per_level < 1 or hidden_features < 1 or hidden_layers < 1:
            raise ValueError("Network dimensions must be positive")
        self.grid_resolutions = tuple(int(value) for value in grid_resolutions)
        self.channels_per_level = int(channels_per_level)
        self.output_scale = float(output_scale)
        self.include_coordinates = bool(include_coordinates)
        self.rotate_and_offset_grids = bool(rotate_and_offset_grids)
        self.fourier_feature_count = int(fourier_feature_count)
        if interpolation_mode not in {"bilinear", "bicubic"}:
            raise ValueError("interpolation_mode must be 'bilinear' or 'bicubic'")
        self.interpolation_mode = str(interpolation_mode)
        if self.fourier_feature_count < 0:
            raise ValueError("fourier_feature_count cannot be negative")
        if self.fourier_feature_count and not (
            0 < fourier_min_cycles < fourier_max_cycles
        ):
            raise ValueError("Fourier cycle limits must satisfy 0 < min < max")
        self.grids = torch.nn.ParameterList()
        for resolution in self.grid_resolutions:
            grid = torch.empty(1, self.channels_per_level, resolution, resolution)
            torch.nn.init.normal_(grid, mean=0.0, std=0.01)
            self.grids.append(torch.nn.Parameter(grid))

        angles = torch.zeros(len(self.grid_resolutions), dtype=torch.float32)
        offsets = torch.zeros(len(self.grid_resolutions), 2, dtype=torch.float32)
        if self.rotate_and_offset_grids:
            # Golden-angle increments modulo 90 degrees avoid repeatedly
            # aligning the separable bilinear kernels between grid levels.
            angles = torch.tensor(
                [
                    math.radians(((index + 1) * 137.50776405003785) % 90.0)
                    for index in range(len(self.grid_resolutions))
                ],
                dtype=torch.float32,
            )
            generator = torch.Generator(device="cpu").manual_seed(int(grid_frame_seed))
            unit_offsets = torch.rand(
                (len(self.grid_resolutions), 2), generator=generator
            ) - 0.5
            for index, resolution in enumerate(self.grid_resolutions):
                # One grid cell spans 2/(resolution-1) normalized units.
                # This gives a fixed random shift of at most half a cell.
                offsets[index] = unit_offsets[index] * (2.0 / (resolution - 1))
        self.register_buffer("grid_frame_angles", angles)
        self.register_buffer("grid_frame_offsets", offsets)

        wavevectors = torch.empty((0, 2), dtype=torch.float32)
        if self.fourier_feature_count:
            generator = torch.Generator(device="cpu").manual_seed(int(fourier_seed))
            angle = torch.rand(self.fourier_feature_count, generator=generator) * math.pi
            log_cycles = (
                math.log(float(fourier_min_cycles))
                + torch.rand(self.fourier_feature_count, generator=generator)
                * math.log(float(fourier_max_cycles) / float(fourier_min_cycles))
            )
            cycles = torch.exp(log_cycles)
            # Coordinates span [-1, 1], hence pi*cycles gives 2*pi*cycles
            # phase change across the complete canvas.
            wavevectors = math.pi * cycles[:, None] * torch.stack(
                (torch.cos(angle), torch.sin(angle)), dim=1
            )
        self.register_buffer("fourier_wavevectors", wavevectors)

        input_features = len(self.grids) * self.channels_per_level
        input_features += 2 * self.fourier_feature_count
        if self.include_coordinates:
            input_features += 2
        layers: list[torch.nn.Module] = []
        current = input_features
        for _ in range(int(hidden_layers)):
            linear = torch.nn.Linear(current, hidden_features)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            torch.nn.init.zeros_(linear.bias)
            layers.extend((linear, torch.nn.SiLU()))
            current = hidden_features
        final = torch.nn.Linear(current, 1)
        torch.nn.init.zeros_(final.bias)
        torch.nn.init.normal_(final.weight, std=0.01)
        layers.append(final)
        self.decoder = torch.nn.Sequential(*layers)

    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Interpolate every latent level at arbitrary coordinates."""

        if coordinates.shape[-1] != 2:
            raise ValueError("Coordinates must end with (x, y)")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(1, -1, 1, 2)
        embeddings = []
        flat_coordinates = coordinates.reshape(-1, 2)
        for index, grid in enumerate(self.grids):
            if self.rotate_and_offset_grids:
                angle = self.grid_frame_angles[index]
                cosine = torch.cos(angle)
                sine = torch.sin(angle)
                x, y = flat_coordinates.unbind(dim=-1)
                level_coordinates = torch.stack(
                    (
                        cosine * x - sine * y,
                        sine * x + cosine * y,
                    ),
                    dim=-1,
                )
                level_coordinates = (
                    level_coordinates + self.grid_frame_offsets[index]
                )
                level_flat = level_coordinates.reshape(1, -1, 1, 2)
            else:
                level_flat = flat
            sampled = torch_f.grid_sample(
                grid,
                level_flat,
                mode=self.interpolation_mode,
                padding_mode="border",
                align_corners=True,
            )
            embeddings.append(sampled[0, :, :, 0].T)
        if self.fourier_feature_count:
            phase = flat_coordinates @ self.fourier_wavevectors.T
            embeddings.extend((torch.sin(phase), torch.cos(phase)))
        values = (*embeddings, coordinates.reshape(-1, 2)) if self.include_coordinates else tuple(embeddings)
        encoded = torch.cat(values, dim=-1)
        return encoded.reshape(*original_shape, encoded.shape[-1])

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(coordinates)
        return self.output_scale * self.decoder(encoded).squeeze(-1)


def coordinate_grid(
    shape: tuple[int, int], *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return a global ``(y, x, 2)`` coordinate grid in ``[-1, 1]``."""

    height, width = map(int, shape)
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)
