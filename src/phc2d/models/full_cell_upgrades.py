"""Project-native upgraded full-cell Full-Wh model families.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any


def _require_torch(model_name: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise ImportError(
            f"Building model '{model_name}' requires optional dependency 'torch'. "
            "Install phc2d with the train extra or skip torch-only model checks."
        ) from exc
    return torch, nn, functional


def _shape_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} is required")
    shape = (int(value),) if isinstance(value, int) else tuple(int(dim) for dim in value)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} must contain positive dimensions, got {value!r}")
    return shape


def _image_contract(input_shape: Any, *, model_name: str) -> tuple[tuple[int, ...], int]:
    shape = _shape_tuple(input_shape, name="input_shape")
    if len(shape) == 2:
        return shape, 1
    if len(shape) == 3:
        return shape, int(shape[0])
    raise ValueError(
        f"{model_name} input_shape must be (height, width) or "
        "(channels, height, width)"
    )


def _target_contract(output_shape: Any) -> tuple[tuple[int, int], int, int]:
    shape = _shape_tuple(output_shape, name="output_shape")
    if len(shape) != 2:
        raise ValueError("Full-Wh output_shape must be (n_k_points, n_bands)")
    return (int(shape[0]), int(shape[1])), int(shape[0]), int(shape[1])


def _head_kind(value: str) -> str:
    kind = str(value).strip().lower()
    if kind not in {"dense", "k_conditioned"}:
        raise ValueError("head_kind must be 'dense' or 'k_conditioned'")
    return kind


def _drop_path_rates(depth: int, maximum: float) -> list[float]:
    if depth <= 1:
        return [float(maximum)]
    return [float(maximum) * index / (depth - 1) for index in range(depth)]


def _make_drop_path(nn: Any, torch: Any) -> type[Any]:
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


def _make_band_head(
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
    n_k_points, n_bands = output_shape
    kind = _head_kind(head_kind)
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
                    f"default k_points count {default.shape[0]} does not match output_shape {n_k_points}"
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

    return MLPMixerBandV1()


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

    torch, nn, _ = _require_torch("res_mlp_gated_v2")
    in_shape = _shape_tuple(input_shape, name="input_shape")
    out_shape, _, _ = _target_contract(output_shape)
    in_dim = int(prod(in_shape))
    width = int(hidden_dim)
    n_blocks = int(depth)
    expansion_value = int(expansion)
    intermediate = width * expansion_value
    if min(width, n_blocks, expansion_value) <= 0:
        raise ValueError("ResMLP dimensions must be positive")
    DropPath = _make_drop_path(nn, torch)

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
            self.head_kind = _head_kind(head_kind)
            self.input_projection = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_dim, width), nn.GELU())
            self.blocks = nn.ModuleList(
                GatedResidualMLPBlock(rate)
                for rate in _drop_path_rates(n_blocks, float(drop_path))
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
                raise ValueError("res_mlp_gated_v2 does not accept lane_id")
            if int(x[0].numel()) != in_dim:
                raise ValueError(f"input sample has {x[0].numel()} values, expected {in_dim}")
            latent = self.input_projection(x)
            for block in self.blocks:
                latent = block(latent)
            return self.head(self.norm(latent), k_points)

    return ResMLPGatedV2()


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
                    nn.Sequential(
                        *(ConvNeXtBlock(width, next(rates)) for _ in range(stage_depth))
                    )
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

    return ConvNeXtBandV1()


def _build_factorized_band_operator(
    *,
    model_name: str,
    input_shape: Any,
    output_shape: Any,
    width: int,
    depth: int,
    modes: Any,
    factorization_rank: float,
    norm_groups: int,
    channel_expansion: int,
    local_kernel: int,
    drop_path: float,
    head_kind: str,
    k_points: Any | None,
    head_hidden_dim: int,
    band_embedding_dim: int,
    fourier_frequencies: Any,
    dropout: float,
    lane_conditioned: bool,
    num_lanes: int,
    lane_embedding_dim: int,
) -> Any:
    torch, nn, functional = _require_torch(model_name)
    in_shape, in_channels = _image_contract(input_shape, model_name=model_name)
    out_shape, _, _ = _target_contract(output_shape)
    model_width = int(width)
    n_blocks = int(depth)
    mode_shape = _shape_tuple(modes, name="modes")
    if len(mode_shape) != 2:
        raise ValueError("modes must contain two values")
    modes_y, modes_x = mode_shape
    rank_fraction = float(factorization_rank)
    groups = int(norm_groups)
    expansion_value = int(channel_expansion)
    kernel = int(local_kernel)
    lane_count = int(num_lanes)
    lane_dim = int(lane_embedding_dim)
    if min(model_width, n_blocks, groups, expansion_value, kernel) <= 0:
        raise ValueError("operator dimensions must be positive")
    if model_width % groups:
        raise ValueError("width must be divisible by norm_groups")
    if kernel % 2 == 0:
        raise ValueError("local_kernel must be odd")
    if not 0.0 < rank_fraction <= 1.0:
        raise ValueError("factorization_rank must be in (0, 1]")
    if lane_conditioned and min(lane_count, lane_dim) <= 0:
        raise ValueError("lane dimensions must be positive")
    DropPath = _make_drop_path(nn, torch)

    rank_channel = max(1, round(model_width * rank_fraction))
    rank_y = max(1, round(modes_y * rank_fraction))
    rank_x = max(1, round(modes_x * rank_fraction))

    class TuckerFactors(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / max(float(model_width), 1.0)
            self.input_factor = nn.Parameter(scale * torch.randn(model_width, rank_channel))
            self.output_factor = nn.Parameter(scale * torch.randn(model_width, rank_channel))
            self.y_factor = nn.Parameter(scale * torch.randn(modes_y, rank_y))
            self.x_factor = nn.Parameter(scale * torch.randn(modes_x, rank_x))
            self.core = nn.Parameter(
                scale
                * torch.randn(
                    rank_channel,
                    rank_channel,
                    rank_y,
                    rank_x,
                    dtype=torch.cfloat,
                )
            )

        def weight(self, keep_y: int, keep_x: int) -> Any:
            dtype = self.core.dtype
            return torch.einsum(
                "ia,ob,yc,xd,abcd->ioyx",
                self.input_factor.to(dtype=dtype),
                self.output_factor.to(dtype=dtype),
                self.y_factor[:keep_y].to(dtype=dtype),
                self.x_factor[:keep_x].to(dtype=dtype),
                self.core,
            )

    class TuckerSpectralConv2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.positive = TuckerFactors()
            self.negative = TuckerFactors()
            self.last_effective_modes = (0, 0)

        def forward(self, x: Any) -> Any:
            batch, _, height, width_in = x.shape
            x_ft = torch.fft.rfft2(x)
            keep_y = min(modes_y, max(height // 2, 1))
            keep_x = min(modes_x, width_in // 2 + 1)
            self.last_effective_modes = (int(keep_y), int(keep_x))
            out_ft = torch.zeros(
                batch,
                model_width,
                height,
                width_in // 2 + 1,
                dtype=x_ft.dtype,
                device=x.device,
            )
            positive_weight = self.positive.weight(keep_y, keep_x).to(device=x.device)
            negative_weight = self.negative.weight(keep_y, keep_x).to(device=x.device)
            out_ft[:, :, :keep_y, :keep_x] = torch.einsum(
                "biyx,ioyx->boyx",
                x_ft[:, :, :keep_y, :keep_x],
                positive_weight,
            )
            out_ft[:, :, -keep_y:, :keep_x] = torch.einsum(
                "biyx,ioyx->boyx",
                x_ft[:, :, -keep_y:, :keep_x],
                negative_weight,
            )
            return torch.fft.irfft2(out_ft, s=(height, width_in))

    class FactorizedOperatorBlock(nn.Module):
        def __init__(self, probability: float) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(groups, model_width)
            self.film = nn.Linear(lane_dim, 2 * model_width) if lane_conditioned else None
            self.spectral = TuckerSpectralConv2d()
            padding = kernel // 2
            self.local = nn.Conv2d(
                model_width,
                model_width,
                kernel_size=kernel,
                padding=padding,
                padding_mode="circular",
            )
            self.channel = nn.Sequential(
                nn.Conv2d(model_width, model_width * expansion_value, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(model_width * expansion_value, model_width, kernel_size=1),
            )
            self.gates = nn.Parameter(torch.zeros(3))
            self.drop_path = DropPath(probability)

        def forward(self, x: Any, lane_features: Any | None) -> Any:
            normalized = self.norm(x)
            if self.film is not None:
                if lane_features is None:
                    raise ValueError("lane-conditioned operator block requires lane features")
                gamma, beta = self.film(lane_features).chunk(2, dim=-1)
                normalized = normalized * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            branches = (self.spectral(normalized), self.local(normalized), self.channel(normalized))
            weights = torch.sigmoid(self.gates)
            update = sum(weight * branch for weight, branch in zip(weights, branches, strict=True))
            update = update / torch.clamp(weights.sum(), min=1.0e-6)
            return functional.gelu(x + self.drop_path(update))

    class FactorizedBandOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.head_kind = _head_kind(head_kind)
            self.lane_conditioned = bool(lane_conditioned)
            self.lift = nn.Conv2d(in_channels, model_width, kernel_size=1)
            self.lane_embedding = (
                nn.Embedding(lane_count, lane_dim) if self.lane_conditioned else None
            )
            self.blocks = nn.ModuleList(
                FactorizedOperatorBlock(rate)
                for rate in _drop_path_rates(n_blocks, float(drop_path))
            )
            self.norm = nn.GroupNorm(groups, model_width)
            self.head = _make_band_head(
                torch=torch,
                nn=nn,
                latent_dim=model_width,
                output_shape=out_shape,
                head_kind=self.head_kind,
                k_points=k_points,
                head_hidden_dim=int(head_hidden_dim),
                band_embedding_dim=int(band_embedding_dim),
                fourier_frequencies=fourier_frequencies,
                dropout=float(dropout),
            )

        def _lane_features(self, lane_id: Any | None, *, batch: int, device: Any) -> Any | None:
            if not self.lane_conditioned:
                if lane_id is not None:
                    raise ValueError(f"{model_name} does not accept lane_id")
                return None
            if lane_id is None:
                raise ValueError("dual_lane_band_operator_v1 requires lane_id")
            lane = torch.as_tensor(lane_id, dtype=torch.long, device=device).reshape(-1)
            if int(lane.numel()) != batch:
                raise ValueError(f"lane_id must contain {batch} values")
            if bool(torch.any((lane < 0) | (lane >= lane_count)).item()):
                raise ValueError(f"lane_id values must be in [0, {lane_count - 1}]")
            return self.lane_embedding(lane)

        def forward(
            self,
            x: Any,
            *,
            k_points: Any | None = None,
            lane_id: Any | None = None,
        ) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4 or int(x.shape[1]) != in_channels:
                raise ValueError(f"expected image input with {in_channels} channels")
            lane_features = self._lane_features(
                lane_id,
                batch=int(x.shape[0]),
                device=x.device,
            )
            features = self.lift(x)
            for block in self.blocks:
                features = block(features, lane_features)
            latent = functional.gelu(self.norm(features)).mean(dim=(-2, -1))
            return self.head(latent, k_points)

    return FactorizedBandOperator()


def build_tfno2d_band_v3(
    *,
    input_shape: Any,
    output_shape: Any,
    width: int = 64,
    depth: int = 6,
    modes: Any = (12, 12),
    factorization_rank: float = 0.75,
    norm_groups: int = 8,
    channel_expansion: int = 2,
    local_kernel: int = 3,
    drop_path: float = 0.1,
    head_kind: str = "dense",
    k_points: Any | None = None,
    head_hidden_dim: int = 256,
    band_embedding_dim: int = 32,
    fourier_frequencies: Any = (1, 2, 4, 8),
    dropout: float = 0.05,
) -> Any:
    """Build the Tucker-factorized two-sided FNO-style Full-Wh upgrade."""

    return _build_factorized_band_operator(
        model_name="tfno2d_band_v3",
        input_shape=input_shape,
        output_shape=output_shape,
        width=width,
        depth=depth,
        modes=modes,
        factorization_rank=factorization_rank,
        norm_groups=norm_groups,
        channel_expansion=channel_expansion,
        local_kernel=local_kernel,
        drop_path=drop_path,
        head_kind=head_kind,
        k_points=k_points,
        head_hidden_dim=head_hidden_dim,
        band_embedding_dim=band_embedding_dim,
        fourier_frequencies=fourier_frequencies,
        dropout=dropout,
        lane_conditioned=False,
        num_lanes=2,
        lane_embedding_dim=16,
    )


def build_dual_lane_band_operator_v1(
    *,
    input_shape: Any,
    output_shape: Any,
    width: int = 64,
    depth: int = 6,
    modes: Any = (12, 12),
    factorization_rank: float = 0.75,
    norm_groups: int = 8,
    channel_expansion: int = 2,
    local_kernel: int = 3,
    drop_path: float = 0.1,
    num_lanes: int = 2,
    lane_embedding_dim: int = 16,
    head_kind: str = "dense",
    k_points: Any | None = None,
    head_hidden_dim: int = 256,
    band_embedding_dim: int = 32,
    fourier_frequencies: Any = (1, 2, 4, 8),
    dropout: float = 0.05,
) -> Any:
    """Build the binary/graded FiLM-conditioned shared band operator."""

    return _build_factorized_band_operator(
        model_name="dual_lane_band_operator_v1",
        input_shape=input_shape,
        output_shape=output_shape,
        width=width,
        depth=depth,
        modes=modes,
        factorization_rank=factorization_rank,
        norm_groups=norm_groups,
        channel_expansion=channel_expansion,
        local_kernel=local_kernel,
        drop_path=drop_path,
        head_kind=head_kind,
        k_points=k_points,
        head_hidden_dim=head_hidden_dim,
        band_embedding_dim=band_embedding_dim,
        fourier_frequencies=fourier_frequencies,
        dropout=dropout,
        lane_conditioned=True,
        num_lanes=num_lanes,
        lane_embedding_dim=lane_embedding_dim,
    )
