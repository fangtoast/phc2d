"""Gated residual MLP builder for the project-native Full-Wh upgrade.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any

from phc2d.models.torch_support import (
    drop_path_rates,
    make_band_head,
    make_drop_path,
    normalize_head_kind,
    require_torch,
    shape_tuple,
    target_contract,
)


def build_res_mlp_gated_v2(
    *,
    input_shape: Any,
    output_shape: Any,
    hidden_dim: int = 320,
    depth: int = 8,
    expansion: int = 2,
    dropout: float = 0.05,
    drop_path: float = 0.1,
    layer_scale_init: float = 1.0e-4,
    head_kind: str = "dense",
    k_points: Any | None = None,
    head_hidden_dim: int = 256,
    band_embedding_dim: int = 32,
    fourier_frequencies: Any = (1, 2, 4, 8),
) -> Any:
    """Build the deeper pre-normalized gated residual MLP upgrade."""

    torch, nn, _ = require_torch("res_mlp_gated_v2", need_functional=True)
    in_shape = shape_tuple(input_shape, name="input_shape")
    out_shape, _, _ = target_contract(output_shape)
    in_dim = int(prod(in_shape))
    width = int(hidden_dim)
    n_blocks = int(depth)
    expansion_value = int(expansion)
    intermediate = width * expansion_value
    if min(width, n_blocks, expansion_value) <= 0:
        raise ValueError("ResMLP dimensions must be positive")
    DropPath = make_drop_path(nn, torch)

    class GatedResidualMLPBlock(nn.Module):
        def __init__(self, probability: float) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.gate = nn.Linear(width, intermediate)
            self.value = nn.Linear(width, intermediate)
            self.projection = nn.Linear(intermediate, width)
            self.dropout = nn.Dropout(float(dropout))
            self.scale = nn.Parameter(torch.full((width,), float(layer_scale_init)))
            self.drop_path = DropPath(probability)

        def forward(self, x: Any) -> Any:
            normalized = self.norm(x)
            gated = torch.nn.functional.gelu(self.gate(normalized)) * self.value(normalized)
            update = self.projection(self.dropout(gated))
            return x + self.drop_path(self.scale * update)

    class ResMLPGatedV2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.head_kind = normalize_head_kind(head_kind)
            self.input_projection = nn.Sequential(
                nn.Flatten(start_dim=1), nn.Linear(in_dim, width), nn.GELU()
            )
            self.blocks = nn.ModuleList(
                GatedResidualMLPBlock(rate)
                for rate in drop_path_rates(n_blocks, float(drop_path))
            )
            self.norm = nn.LayerNorm(width)
            self.head = make_band_head(
                torch=torch,
                nn=nn,
                latent_dim=width,
                output_shape=out_shape,
                head_kind=self.head_kind,
                k_points=k_points,
                head_hidden_dim=int(head_hidden_dim),
                band_embedding_dim=int(band_embedding_dim),
                fourier_frequencies=fourier_frequencies,
                dropout=float(dropout),
            )

        def forward(
            self,
            x: Any,
            *,
            k_points: Any | None = None,
            lane_id: Any | None = None,
        ) -> Any:
            if lane_id is not None:
                raise ValueError("res_mlp_gated_v2 does not accept lane_id")
            if int(x[0].numel()) != in_dim:
                raise ValueError(f"input sample has {x[0].numel()} values, expected {in_dim}")
            latent = self.input_projection(x)
            for block in self.blocks:
                latent = block(latent)
            return self.head(self.norm(latent), k_points)

    ResMLPGatedV2.__phc2d_architecture_module__ = "phc2d.models.full_cell_upgrades"
    return ResMLPGatedV2()
