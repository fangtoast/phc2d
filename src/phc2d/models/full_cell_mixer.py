"""MLP-Mixer builder for the project-native Full-Wh upgrade.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any

from phc2d.models._full_cell_common import (
    _drop_path_rates,
    _head_kind,
    _image_contract,
    _make_band_head,
    _make_drop_path,
    _require_torch,
    _shape_tuple,
    _target_contract,
)


def build_mlp_mixer_band_v1(
    *,
    input_shape: Any,
    output_shape: Any,
    token_grid: Any = (16, 16),
    embed_dim: int = 256,
    depth: int = 8,
    token_mlp_dim: int = 128,
    channel_mlp_dim: int = 512,
    dropout: float = 0.05,
    drop_path: float = 0.1,
    layer_scale_init: float = 1.0e-4,
    head_kind: str = "dense",
    k_points: Any | None = None,
    head_hidden_dim: int = 256,
    band_embedding_dim: int = 32,
    fourier_frequencies: Any = (1, 2, 4, 8),
) -> Any:
    """Build the grid-token MLP-Mixer-style Full-Wh upgrade."""

    torch, nn, functional = _require_torch("mlp_mixer_band_v1")
    in_shape, in_channels = _image_contract(input_shape, model_name="mlp_mixer_band_v1")
    out_shape, _, _ = _target_contract(output_shape)
    token_shape = _shape_tuple(token_grid, name="token_grid")
    if len(token_shape) != 2:
        raise ValueError("token_grid must contain two dimensions")
    width = int(embed_dim)
    n_blocks = int(depth)
    token_hidden = int(token_mlp_dim)
    channel_hidden = int(channel_mlp_dim)
    if min(width, n_blocks, token_hidden, channel_hidden) <= 0:
        raise ValueError("Mixer dimensions must be positive")
    n_tokens = int(prod(token_shape))
    DropPath = _make_drop_path(nn, torch)

    class MixerBlock(nn.Module):
        def __init__(self, probability: float) -> None:
            super().__init__()
            self.token_norm = nn.LayerNorm(width)
            self.token_mlp = nn.Sequential(
                nn.Linear(n_tokens, token_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(token_hidden, n_tokens),
            )
            self.channel_norm = nn.LayerNorm(width)
            self.channel_mlp = nn.Sequential(
                nn.Linear(width, channel_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(channel_hidden, width),
            )
            self.token_scale = nn.Parameter(torch.full((width,), float(layer_scale_init)))
            self.channel_scale = nn.Parameter(torch.full((width,), float(layer_scale_init)))
            self.drop_path = DropPath(probability)

        def forward(self, tokens: Any) -> Any:
            mixed_tokens = self.token_mlp(self.token_norm(tokens).transpose(1, 2)).transpose(1, 2)
            tokens = tokens + self.drop_path(self.token_scale * mixed_tokens)
            mixed_channels = self.channel_mlp(self.channel_norm(tokens))
            return tokens + self.drop_path(self.channel_scale * mixed_channels)

    class MLPMixerBandV1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.head_kind = _head_kind(head_kind)
            self.token_grid = token_shape
            self.token_projection = nn.Linear(in_channels, width)
            self.position = nn.Parameter(torch.zeros(1, n_tokens, width))
            nn.init.trunc_normal_(self.position, std=0.02)
            self.blocks = nn.ModuleList(
                MixerBlock(rate) for rate in _drop_path_rates(n_blocks, float(drop_path))
            )
            self.norm = nn.LayerNorm(width)
            self.head = _make_band_head(
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
                raise ValueError("mlp_mixer_band_v1 does not accept lane_id")
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4 or int(x.shape[1]) != in_channels:
                raise ValueError(f"expected image input with {in_channels} channels")
            pooled = functional.adaptive_avg_pool2d(x, token_shape)
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = self.token_projection(tokens) + self.position
            for block in self.blocks:
                tokens = block(tokens)
            latent = self.norm(tokens).mean(dim=1)
            return self.head(latent, k_points)

    MLPMixerBandV1.__phc2d_architecture_module__ = "phc2d.models.full_cell_upgrades"
    return MLPMixerBandV1()
