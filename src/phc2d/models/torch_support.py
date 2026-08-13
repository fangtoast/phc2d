"""Shared lazy Torch, shape, and Full-Wh model-building helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any

__all__ = [
    "drop_path_rates",
    "image_contract",
    "make_band_head",
    "make_drop_path",
    "numel",
    "normalize_head_kind",
    "require_torch",
    "shape_tuple",
    "target_contract",
]


def require_torch(
    model_name: str,
    *,
    need_functional: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Import Torch lazily for a model builder."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            f"Building model '{model_name}' requires optional dependency 'torch'. "
            "Install phc2d with the train extra or skip torch-only model checks."
        ) from exc
    if need_functional:
        from torch.nn import functional

        return torch, nn, functional
    return torch, nn


def shape_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    """Normalize and validate a positive shape tuple."""

    if value is None:
        raise ValueError(f"{name} is required")
    shape = (int(value),) if isinstance(value, int) else tuple(int(dim) for dim in value)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} must contain positive dimensions, got {value!r}")
    return shape


def numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements represented by a shape."""

    return int(prod(shape))


def image_contract(input_shape: Any, *, model_name: str) -> tuple[tuple[int, ...], int]:
    """Validate an image input shape and return its channel count."""

    shape = shape_tuple(input_shape, name="input_shape")
    if len(shape) == 2:
        return shape, 1
    if len(shape) == 3:
        return shape, int(shape[0])
    raise ValueError(
        f"{model_name} input_shape must be (height, width) or (channels, height, width)"
    )


def target_contract(output_shape: Any) -> tuple[tuple[int, int], int, int]:
    """Validate a Full-Wh target shape."""

    shape = shape_tuple(output_shape, name="output_shape")
    if len(shape) != 2:
        raise ValueError("Full-Wh output_shape must be (n_k_points, n_bands)")
    return (int(shape[0]), int(shape[1])), int(shape[0]), int(shape[1])


def normalize_head_kind(value: str) -> str:
    """Normalize the supported Full-Wh head kind."""

    kind = str(value).strip().lower()
    if kind not in {"dense", "k_conditioned"}:
        raise ValueError("head_kind must be 'dense' or 'k_conditioned'")
    return kind


def drop_path_rates(depth: int, maximum: float) -> list[float]:
    """Return a linearly increasing stochastic-depth schedule."""

    if depth <= 1:
        return [float(maximum)]
    return [float(maximum) * index / (depth - 1) for index in range(depth)]


def make_drop_path(nn: Any, torch: Any) -> type[Any]:
    """Build the lightweight stochastic-depth module used by upgrade models."""

    class DropPath(nn.Module):
        def __init__(self, probability: float) -> None:
            super().__init__()
            self.probability = float(probability)
            if not 0.0 <= self.probability < 1.0:
                raise ValueError("drop_path probability must be in [0, 1)")

        def forward(self, x: Any) -> Any:
            if not self.training or self.probability == 0.0:
                return x
            keep = 1.0 - self.probability
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.floor(keep + torch.rand(shape, dtype=x.dtype, device=x.device))
            return x * mask / keep

    return DropPath


def make_band_head(
    *,
    torch: Any,
    nn: Any,
    latent_dim: int,
    output_shape: tuple[int, int],
    head_kind: str,
    k_points: Any | None,
    head_hidden_dim: int,
    band_embedding_dim: int,
    fourier_frequencies: Any,
    dropout: float,
) -> Any:
    """Build a dense or k-conditioned Full-Wh prediction head."""

    n_k_points, n_bands = output_shape
    kind = normalize_head_kind(head_kind)
    hidden = int(head_hidden_dim)
    embedding_dim = int(band_embedding_dim)
    frequencies = tuple(float(value) for value in fourier_frequencies)
    if latent_dim <= 0 or hidden <= 0 or embedding_dim <= 0:
        raise ValueError("head dimensions must be positive")
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("fourier_frequencies must contain positive values")

    class DenseBandHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(latent_dim)
            self.projection = nn.Linear(latent_dim, n_k_points * n_bands)

        def forward(self, latent: Any, _: Any | None = None) -> Any:
            output = self.projection(self.norm(latent))
            return output.reshape(latent.shape[0], n_k_points, n_bands)

    class KConditionedBandHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            default = (
                torch.empty((0, 2), dtype=torch.float32)
                if k_points is None
                else torch.as_tensor(k_points, dtype=torch.float32)
            )
            if default.ndim != 2 or default.shape[-1] != 2:
                raise ValueError("k_points must have shape (n_k_points, 2)")
            if default.numel() and int(default.shape[0]) != n_k_points:
                raise ValueError(
                    f"default k_points count {default.shape[0]} does not match "
                    f"output_shape {n_k_points}"
                )
            self.register_buffer("default_k_points", default.contiguous(), persistent=True)
            feature_dim = 2 + 4 * len(frequencies)
            self.band_embedding = nn.Embedding(n_bands, embedding_dim)
            self.query_encoder = nn.Sequential(
                nn.Linear(feature_dim + embedding_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
            )
            self.structure_projection = nn.Linear(latent_dim, hidden)
            self.fusion_norm = nn.LayerNorm(hidden)
            self.predictor = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )

        def _fourier_features(self, points: Any) -> Any:
            features = [points]
            for frequency in frequencies:
                angles = 2.0 * torch.pi * frequency * points
                features.extend((torch.sin(angles), torch.cos(angles)))
            return torch.cat(features, dim=-1)

        def forward(self, latent: Any, points: Any | None = None) -> Any:
            resolved = self.default_k_points if points is None else points
            if resolved is None or int(resolved.numel()) == 0:
                raise ValueError("k_conditioned head requires k_points")
            resolved = torch.as_tensor(resolved, dtype=latent.dtype, device=latent.device)
            if resolved.ndim != 2 or resolved.shape[-1] != 2:
                raise ValueError("k_points must have shape (n_k_points, 2)")
            point_features = self._fourier_features(resolved)
            band_indices = torch.arange(n_bands, device=latent.device)
            band_features = self.band_embedding(band_indices).to(dtype=latent.dtype)
            point_grid = point_features[:, None, :].expand(-1, n_bands, -1)
            band_grid = band_features[None, :, :].expand(resolved.shape[0], -1, -1)
            query = self.query_encoder(torch.cat((point_grid, band_grid), dim=-1))
            structure = self.structure_projection(latent)[:, None, None, :]
            fused = self.fusion_norm(structure + query[None, :, :, :])
            return self.predictor(fused).squeeze(-1)

    return DenseBandHead() if kind == "dense" else KConditionedBandHead()
