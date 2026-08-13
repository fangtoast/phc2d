"""Query-conditioned MLP, DeepONet, and spectral model builders.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models._common import _numel, _require_torch, _shape_tuple


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

    MLPQuery.__phc2d_architecture_module__ = "phc2d.models.baselines"
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

    DeepONetQuery.__phc2d_architecture_module__ = "phc2d.models.baselines"
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
        raise ValueError(
            "fno2d_query input_shape must be (height, width) or (channels, height, width)"
        )
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

    FNO2DQuery.__phc2d_architecture_module__ = "phc2d.models.baselines"
    return FNO2DQuery()


__all__ = [
    "build_deep_onet_query",
    "build_fno2d_query",
    "build_mlp_query",
]
