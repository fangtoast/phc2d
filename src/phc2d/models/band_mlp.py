"""Vector MLP builders for Full-Wh forward baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models._common import _numel, _require_torch, _shape_tuple


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

    ResidualMLP.__phc2d_architecture_module__ = "phc2d.models.baselines"
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

    PlainMLP.__phc2d_architecture_module__ = "phc2d.models.baselines"
    return PlainMLP()


def build_pod_res_mlp(**_: Any) -> Any:
    """Registered placeholder until the POD basis/projection contract is native."""

    raise NotImplementedError(
        "pod_res_mlp is registered as a Gate 2 placeholder, but the native POD "
        "basis/projection contract is not implemented yet."
    )
