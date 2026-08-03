"""Minimal Gate 2 torch baselines built through the model registry.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any


def _require_torch(model_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            f"Building model '{model_name}' requires optional dependency 'torch'. "
            "Install phc2d with the train extra or skip torch-only smoke tests."
        ) from exc
    return torch, nn


def _shape_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, int):
        shape = (value,)
    else:
        shape = tuple(int(dim) for dim in value)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} must contain positive dimensions, got {value!r}")
    return shape


def _numel(shape: tuple[int, ...]) -> int:
    return int(prod(shape))


def build_res_mlp(
    *,
    input_shape: Any,
    output_shape: Any,
    hidden_dim: int,
    depth: int,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> Any:
    """Build a residual MLP for vector-to-band-matrix smoke tests."""

    _, nn = _require_torch("res_mlp")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    in_dim = _numel(in_shape)
    out_dim = _numel(out_shape)
    hidden = int(hidden_dim)
    n_blocks = int(depth)
    if hidden <= 0 or n_blocks <= 0:
        raise ValueError("hidden_dim and depth must be positive")

    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden, hidden),
                activation_layer(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
            )
            self.act = activation_layer()

        def forward(self, x: Any) -> Any:
            return self.act(x + self.net(x))

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_dim, hidden),
                activation_layer(),
                *[ResidualBlock() for _ in range(n_blocks)],
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: Any) -> Any:
            y = self.net(x)
            return y.reshape(x.shape[0], *self.output_shape)

    return ResidualMLP()


def build_mlp(
    *,
    input_shape: Any,
    output_shape: Any,
    hidden_dim: int,
    depth: int,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> Any:
    """Build a plain MLP matched to ``build_res_mlp`` without skip connections."""

    _, nn = _require_torch("mlp")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    in_dim = _numel(in_shape)
    out_dim = _numel(out_shape)
    hidden = int(hidden_dim)
    n_blocks = int(depth)
    if hidden <= 0 or n_blocks <= 0:
        raise ValueError("hidden_dim and depth must be positive")

    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class PlainBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden, hidden),
                activation_layer(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
            )
            self.act = activation_layer()

        def forward(self, x: Any) -> Any:
            return self.act(self.net(x))

    class PlainMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_dim, hidden),
                activation_layer(),
                *[PlainBlock() for _ in range(n_blocks)],
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: Any) -> Any:
            y = self.net(x)
            return y.reshape(x.shape[0], *self.output_shape)

    return PlainMLP()


def build_pod_res_mlp(**_: Any) -> Any:
    """Registered placeholder until the POD basis/projection contract is native."""

    raise NotImplementedError(
        "pod_res_mlp is registered as a Gate 2 placeholder, but the native POD "
        "basis/projection contract is not implemented yet."
    )


def build_lifted_cnn(
    *,
    input_shape: Any,
    output_shape: Any,
    channels: int,
    activation: str = "gelu",
) -> Any:
    """Build a tiny CNN baseline for grid/image representations."""

    _, nn = _require_torch("lifted_cnn")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    if len(in_shape) == 2:
        in_channels = 1
    elif len(in_shape) == 3:
        in_channels = int(in_shape[0])
    else:
        raise ValueError("lifted_cnn input_shape must be (height, width) or (channels, height, width)")
    width = int(channels)
    if width <= 0:
        raise ValueError("channels must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class LiftedCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                activation_layer(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                activation_layer(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
            )
            self.head = nn.Linear(width, _numel(out_shape))

        def forward(self, x: Any) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            y = self.head(self.encoder(x))
            return y.reshape(x.shape[0], *self.output_shape)

    return LiftedCNN()


def build_fno_lite(
    *,
    input_shape: Any,
    output_shape: Any,
    width: int,
    depth: int,
    modes: Any,
    activation: str = "gelu",
) -> Any:
    """Build a minimal spectral-convolution baseline for CPU shape smoke."""

    torch, nn = _require_torch("fno_lite")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    if len(in_shape) == 2:
        in_channels = 1
    elif len(in_shape) == 3:
        in_channels = int(in_shape[0])
    else:
        raise ValueError("fno_lite input_shape must be (height, width) or (channels, height, width)")
    model_width = int(width)
    n_layers = int(depth)
    modes_y, modes_x = _shape_tuple(modes, name="modes")
    if model_width <= 0 or n_layers <= 0:
        raise ValueError("width and depth must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class SpectralConv2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / max(float(model_width * model_width), 1.0)
            self.weight = nn.Parameter(
                scale
                * torch.randn(
                    model_width,
                    model_width,
                    modes_y,
                    modes_x,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x: Any) -> Any:
            batch, _, height, width_in = x.shape
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros(
                batch,
                model_width,
                height,
                width_in // 2 + 1,
                dtype=x_ft.dtype,
                device=x.device,
            )
            keep_y = min(modes_y, height)
            keep_x = min(modes_x, width_in // 2 + 1)
            out_ft[:, :, :keep_y, :keep_x] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :keep_y, :keep_x],
                self.weight[:, :, :keep_y, :keep_x],
            )
            return torch.fft.irfft2(out_ft, s=(height, width_in))

    class FnoBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectral = SpectralConv2d()
            self.pointwise = nn.Conv2d(model_width, model_width, kernel_size=1)
            self.act = activation_layer()

        def forward(self, x: Any) -> Any:
            return self.act(self.spectral(x) + self.pointwise(x))

    class FnoLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.lift = nn.Conv2d(in_channels, model_width, kernel_size=1)
            self.blocks = nn.Sequential(*[FnoBlock() for _ in range(n_layers)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(model_width, _numel(out_shape))

        def forward(self, x: Any) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            y = self.blocks(self.lift(x))
            y = self.pool(y).flatten(start_dim=1)
            return self.head(y).reshape(x.shape[0], *self.output_shape)

    return FnoLite()


def build_fno2d_band(
    *,
    input_shape: Any,
    output_shape: Any,
    width: int,
    depth: int,
    modes: Any,
    band_modes: Any | None = None,
    fourier_rank: int = 4,
    activation: str = "gelu",
) -> Any:
    """Build a dedicated band-aware spectral baseline for the `epsilon_grid -> Wh` task."""

    torch, nn = _require_torch("fno2d_band")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    if len(in_shape) == 2:
        in_channels = 1
    elif len(in_shape) == 3:
        in_channels = int(in_shape[0])
    else:
        raise ValueError("fno2d_band input_shape must be (height, width) or (channels, height, width)")
    model_width = int(width)
    n_layers = int(depth)
    modes_y, modes_x = _shape_tuple(modes, name="modes")
    band_modes_y, band_modes_x = _shape_tuple(
        band_modes if band_modes is not None else modes,
        name="band_modes",
    )
    fourier_rank = max(int(fourier_rank), 1)
    if model_width <= 0 or n_layers <= 0:
        raise ValueError("width and depth must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU
    channel_factor = 1.0 / max(float(model_width * model_width), 1.0)

    class BandSpectralConv2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_full = nn.Parameter(
                channel_factor
                * torch.randn(
                    model_width,
                    model_width,
                    modes_y,
                    modes_x,
                    dtype=torch.cfloat,
                )
            )
            self.weight_band = nn.Parameter(
                channel_factor
                * torch.randn(
                    model_width,
                    model_width,
                    band_modes_y,
                    band_modes_x,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x: Any) -> Any:
            batch, _, height, width_in = x.shape
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros(
                batch,
                model_width,
                height,
                width_in // 2 + 1,
                dtype=x_ft.dtype,
                device=x.device,
            )
            keep_y = min(modes_y, height)
            keep_x = min(modes_x, width_in // 2 + 1)
            keep_y_band = min(band_modes_y, height)
            keep_x_band = min(band_modes_x, width_in // 2 + 1)
            out_ft[:, :, :keep_y, :keep_x] += torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :keep_y, :keep_x],
                self.weight_full[:, :, :keep_y, :keep_x],
            )
            out_ft[:, :, :keep_y_band, :keep_x_band] += torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :keep_y_band, :keep_x_band],
                self.weight_band[:, :, :keep_y_band, :keep_x_band],
            )
            return torch.fft.irfft2(out_ft, s=(height, width_in))

    class BandBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectral = BandSpectralConv2d()
            self.pointwise = nn.Conv2d(model_width, model_width, kernel_size=1)
            rank_width = min(model_width, fourier_rank)
            self.low_rank = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(model_width, rank_width),
                nn.ReLU(inplace=False),
                nn.Linear(rank_width, model_width),
            )
            self.band_gate = nn.Sequential(
                nn.Conv2d(model_width, model_width, kernel_size=1),
                nn.Sigmoid(),
            )
            self.act = activation_layer()
            self.skip = nn.Conv2d(model_width, model_width, kernel_size=1)

        def forward(self, x: Any) -> Any:
            residual = x
            x = self.spectral(x) + self.pointwise(x)
            x = self.band_gate(x) * x
            low_rank = self.low_rank(residual).reshape(residual.shape[0], -1, 1, 1)
            x = x + low_rank
            x = self.skip(x)
            return self.act(x + residual)

    class Fno2DBand(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.lift = nn.Conv2d(in_channels, model_width, kernel_size=1)
            self.blocks = nn.Sequential(*[BandBlock() for _ in range(n_layers)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(model_width, _numel(out_shape))

        def forward(self, x: Any) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            y = self.blocks(self.lift(x))
            y = self.pool(y).flatten(start_dim=1)
            return self.head(y).reshape(x.shape[0], *out_shape)

    return Fno2DBand()


def build_mlp_query(
    *,
    input_shape: Any,
    output_shape: Any,
    k_dim: int = 1,
    hidden_dim: int = 64,
    depth: int = 2,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> Any:
    """Build a query MLP baseline for ``epsilon_grid + k_coordinate -> band_vector``."""

    _, nn = _require_torch("mlp_query")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    hidden = int(hidden_dim)
    n_layers = int(depth)
    q_dim = int(k_dim)
    if hidden <= 0 or n_layers <= 0 or q_dim <= 0:
        raise ValueError("hidden_dim, depth, and k_dim must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU
    layers: list[Any] = [nn.Linear(_numel(in_shape) + q_dim, hidden), activation_layer()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Dropout(float(dropout)), nn.Linear(hidden, hidden), activation_layer()])
    layers.append(nn.Linear(hidden, _numel(out_shape)))

    class MLPQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.k_dim = q_dim
            self.net = nn.Sequential(*layers)

        def forward(self, structure: Any, k_coordinate: Any | None = None) -> Any:
            if isinstance(structure, dict):
                k_coordinate = structure["k_coordinate"]
                structure = structure["structure"]
            if k_coordinate is None:
                raise ValueError("mlp_query requires k_coordinate")
            x = structure.flatten(start_dim=1)
            k = k_coordinate.reshape(k_coordinate.shape[0], -1)
            y = self.net(__import__("torch").cat([x, k], dim=1))
            return y.reshape(structure.shape[0], *self.output_shape)

    return MLPQuery()


def build_deep_onet_query(
    *,
    input_shape: Any,
    output_shape: Any,
    k_dim: int = 1,
    branch_width: int = 64,
    trunk_width: int = 64,
    latent_dim: int = 32,
    depth: int = 2,
    activation: str = "gelu",
) -> Any:
    """Build a minimal DeepONet-style query model with branch/trunk product fusion."""

    _, nn = _require_torch("deep_onet_query")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    q_dim = int(k_dim)
    latent = int(latent_dim)
    branch_hidden = int(branch_width)
    trunk_hidden = int(trunk_width)
    n_layers = int(depth)
    if min(q_dim, latent, branch_hidden, trunk_hidden, n_layers) <= 0:
        raise ValueError("DeepONet query dimensions must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    def _mlp(in_dim: int, hidden: int, out_dim: int) -> Any:
        layers: list[Any] = [nn.Linear(in_dim, hidden), activation_layer()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), activation_layer()])
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    class DeepONetQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.k_dim = q_dim
            self.branch = _mlp(_numel(in_shape), branch_hidden, latent)
            self.trunk = _mlp(q_dim, trunk_hidden, latent)
            self.bias = nn.Parameter(__import__("torch").zeros(_numel(out_shape)))
            self.band_head = nn.Linear(latent, _numel(out_shape), bias=False)

        def forward(self, structure: Any, k_coordinate: Any | None = None) -> Any:
            if isinstance(structure, dict):
                k_coordinate = structure["k_coordinate"]
                structure = structure["structure"]
            if k_coordinate is None:
                raise ValueError("deep_onet_query requires k_coordinate")
            branch = self.branch(structure.flatten(start_dim=1))
            trunk = self.trunk(k_coordinate.reshape(k_coordinate.shape[0], -1))
            fused = branch * trunk
            y = self.band_head(fused) + self.bias
            return y.reshape(structure.shape[0], *self.output_shape)

    return DeepONetQuery()


def build_fno2d_query(
    *,
    input_shape: Any,
    output_shape: Any,
    k_dim: int = 1,
    width: int = 16,
    depth: int = 2,
    modes: Any = (6, 6),
    query_hidden_dim: int = 32,
    activation: str = "gelu",
) -> Any:
    """Build a spectral-grid encoder with explicit k-query conditioning."""

    torch, nn = _require_torch("fno2d_query")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape = _shape_tuple(output_shape, name="output_shape")
    if len(in_shape) == 2:
        in_channels = 1
    elif len(in_shape) == 3:
        in_channels = int(in_shape[0])
    else:
        raise ValueError("fno2d_query input_shape must be (height, width) or (channels, height, width)")
    q_dim = int(k_dim)
    model_width = int(width)
    n_layers = int(depth)
    modes_y, modes_x = _shape_tuple(modes, name="modes")
    query_hidden = int(query_hidden_dim)
    if min(q_dim, model_width, n_layers, query_hidden) <= 0:
        raise ValueError("fno2d_query dimensions must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class QuerySpectralConv2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / max(float(model_width * model_width), 1.0)
            self.weight = nn.Parameter(
                scale * torch.randn(model_width, model_width, modes_y, modes_x, dtype=torch.cfloat)
            )

        def forward(self, x: Any) -> Any:
            batch, _, height, width_in = x.shape
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros(
                batch,
                model_width,
                height,
                width_in // 2 + 1,
                dtype=x_ft.dtype,
                device=x.device,
            )
            keep_y = min(modes_y, height)
            keep_x = min(modes_x, width_in // 2 + 1)
            out_ft[:, :, :keep_y, :keep_x] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :keep_y, :keep_x],
                self.weight[:, :, :keep_y, :keep_x],
            )
            return torch.fft.irfft2(out_ft, s=(height, width_in))

    class QueryFNOBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectral = QuerySpectralConv2d()
            self.pointwise = nn.Conv2d(model_width, model_width, kernel_size=1)
            self.act = activation_layer()

        def forward(self, x: Any) -> Any:
            return self.act(self.spectral(x) + self.pointwise(x))

    class FNO2DQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.k_dim = q_dim
            self.lift = nn.Conv2d(in_channels, model_width, kernel_size=1)
            self.blocks = nn.Sequential(*[QueryFNOBlock() for _ in range(n_layers)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.query_encoder = nn.Sequential(
                nn.Linear(q_dim, query_hidden),
                activation_layer(),
                nn.Linear(query_hidden, model_width),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.Linear(model_width * 2, query_hidden),
                activation_layer(),
                nn.Linear(query_hidden, _numel(out_shape)),
            )

        def forward(self, structure: Any, k_coordinate: Any | None = None) -> Any:
            if isinstance(structure, dict):
                k_coordinate = structure["k_coordinate"]
                structure = structure["structure"]
            if k_coordinate is None:
                raise ValueError("fno2d_query requires k_coordinate")
            if structure.ndim == 3:
                structure = structure.unsqueeze(1)
            grid_features = self.blocks(self.lift(structure))
            pooled = self.pool(grid_features).flatten(start_dim=1)
            query_gate = self.query_encoder(k_coordinate.reshape(k_coordinate.shape[0], -1))
            fused = torch.cat([pooled, pooled * query_gate], dim=1)
            return self.head(fused).reshape(structure.shape[0], *self.output_shape)

    return FNO2DQuery()


__all__ = [
    "build_fno_lite",
    "build_fno2d_band",
    "build_fno2d_query",
    "build_deep_onet_query",
    "build_lifted_cnn",
    "build_mlp",
    "build_mlp_query",
    "build_pod_res_mlp",
    "build_res_mlp",
]
