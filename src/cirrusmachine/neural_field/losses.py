"""Memory-controlled Sliced Wasserstein losses."""

from __future__ import annotations

import torch


def random_projections(
    feature_count: int,
    projection_count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    vectors = torch.randn(
        feature_count,
        projection_count,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return vectors / vectors.norm(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).eps)


def chunked_project(features: torch.Tensor, projections: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Project feature rows in chunks, concatenating before global quantiles."""

    pieces = [
        features[start : start + chunk_size] @ projections
        for start in range(0, features.shape[0], int(chunk_size))
    ]
    return torch.cat(pieces, dim=0)


def sliced_wasserstein_loss(
    synthesis_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    projection_count: int,
    quantile_count: int,
    generator: torch.Generator,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compare projected feature quantiles with freshly sampled directions."""

    if synthesis_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("SWD features must be matrices")
    if synthesis_features.shape[1] != target_features.shape[1]:
        raise ValueError("SWD feature dimensions must match")
    dtype = torch.float32
    synthesis = synthesis_features.to(dtype)
    target = target_features.to(device=synthesis.device, dtype=dtype)
    projections = random_projections(
        synthesis.shape[1],
        projection_count,
        generator=generator,
        device=synthesis.device,
        dtype=dtype,
    )
    synth_projected = chunked_project(synthesis, projections, chunk_size)
    target_projected = chunked_project(target, projections, chunk_size)
    levels = torch.linspace(0.0, 1.0, quantile_count, device=synthesis.device, dtype=dtype)
    synth_quantiles = torch.quantile(synth_projected, levels, dim=0)
    target_quantiles = torch.quantile(target_projected, levels, dim=0)
    return torch.mean((synth_quantiles - target_quantiles) ** 2)
