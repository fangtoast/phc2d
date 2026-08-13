"""Periodic ConvNeXt builder for the project-native Full-Wh upgrade.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

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


def build_convnext_band_v1(
    *,
    input_shape: Any,
    output_shape: Any,
    stage_dims: Any = (48, 96, 192),
    stage_depths: Any = (2, 3, 6),
    stem_stride: int = 4,
    kernel_size: int = 7,
    channel_expansion: int = 4,
    drop_path: float = 0.1,
    layer_scale_init: float = 1.0e-6,
    head_kind: str = "dense",
    k_points: Any | None = None,
    head_hidden_dim: int = 256,
    band_embedding_dim: int = 32,
    fourier_frequencies: Any = (1, 2, 4, 8),
    dropout: float = 0.05,
) -> Any:
    """Build the periodic ConvNeXt-style Full-Wh upgrade."""

    torch, nn, functional = _require_torch("convnext_band_v1")
    in_shape, in_channels = _image_contract(input_shape, model_name="convnext_band_v1")
    out_shape, _, _ = _target_contract(output_shape)
    dims = _shape_tuple(stage_dims, name="stage_dims")
    depths = _shape_tuple(stage_depths, name="stage_depths")
    if len(dims) != 3 or len(depths) != 3:
        raise ValueError("stage_dims and stage_depths must each contain three values")
    stride = int(stem_stride)
    kernel = int(kernel_size)
    expansion_value = int(channel_expansion)
    if stride <= 0 or kernel <= 0 or kernel % 2 == 0 or expansion_value <= 0:
        raise ValueError("invalid ConvNeXt stem/kernel/expansion")
    total_depth = sum(depths)
    rates = iter(_drop_path_rates(total_depth, float(drop_path)))
    DropPath = _make_drop_path(nn, torch)

    def circular_pad2d(x: Any, padding: int) -> Any:
        """Periodic padding that also supports feature maps smaller than padding."""

        if padding <= 0:
            return x
        height, width = int(x.shape[-2]), int(x.shape[-1])
        rows = torch.arange(-padding, height + padding, device=x.device).remainder(height)
        cols = torch.arange(-padding, width + padding, device=x.device).remainder(width)
        return x.index_select(-2, rows).index_select(-1, cols)

    class ConvNeXtBlock(nn.Module):
        def __init__(self, width: int, probability: float) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(width, width, kernel_size=kernel, groups=width, bias=True)
            self.norm = nn.LayerNorm(width)
            self.expand = nn.Linear(width, width * expansion_value)
            self.project = nn.Linear(width * expansion_value, width)
            self.scale = nn.Parameter(torch.full((width,), float(layer_scale_init)))
            self.drop_path = DropPath(probability)

        def forward(self, x: Any) -> Any:
            padding = kernel // 2
            update = self.depthwise(circular_pad2d(x, padding))
            update = update.permute(0, 2, 3, 1)
            update = self.project(functional.gelu(self.expand(self.norm(update))))
            update = (self.scale * update).permute(0, 3, 1, 2)
            return x + self.drop_path(update)

    class PeriodicDownsample(nn.Module):
        def __init__(self, in_width: int, out_width: int) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(1, in_width)
            self.projection = nn.Conv2d(in_width, out_width, kernel_size=1)

        def forward(self, x: Any) -> Any:
            return self.projection(functional.avg_pool2d(self.norm(x), kernel_size=2, stride=2))

    class ConvNeXtBandV1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.head_kind = _head_kind(head_kind)
            self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=stride, stride=stride)
            self.stages = nn.ModuleList()
            self.downsamples = nn.ModuleList()
            for stage_index, (width, stage_depth) in enumerate(zip(dims, depths, strict=True)):
                self.stages.append(
                    nn.Sequential(*(ConvNeXtBlock(width, next(rates)) for _ in range(stage_depth)))
                )
                if stage_index < len(dims) - 1:
                    self.downsamples.append(PeriodicDownsample(width, dims[stage_index + 1]))
            self.norm = nn.LayerNorm(dims[-1])
            self.head = _make_band_head(
                torch=torch,
                nn=nn,
                latent_dim=dims[-1],
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
                raise ValueError("convnext_band_v1 does not accept lane_id")
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4 or int(x.shape[1]) != in_channels:
                raise ValueError(f"expected image input with {in_channels} channels")
            required_divisor = stride * (2 ** (len(dims) - 1))
            if int(x.shape[-2]) % required_divisor or int(x.shape[-1]) % required_divisor:
                raise ValueError(
                    f"input height/width must be divisible by {required_divisor}, got {tuple(x.shape[-2:])}"
                )
            features = self.stem(x)
            for stage_index, stage in enumerate(self.stages):
                features = stage(features)
                if stage_index < len(self.downsamples):
                    features = self.downsamples[stage_index](features)
            latent = self.norm(features.mean(dim=(-2, -1)))
            return self.head(latent, k_points)

    ConvNeXtBandV1.__phc2d_architecture_module__ = "phc2d.models.full_cell_upgrades"
    return ConvNeXtBandV1()
