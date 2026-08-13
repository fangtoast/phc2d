"""Factorized and lane-conditioned operators for Full-Wh upgrades.

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
            # Reconstruct one Tucker mode at a time.  A single five-operand
            # einsum can choose a catastrophically large contraction path on
            # the CUDA/PyTorch version used by N40 (hundreds of GiB for the
            # default width), even though the final tensor is only a few MiB.
            weight = torch.einsum(
                "ia,abcd->ibcd",
                self.input_factor.to(dtype=dtype),
                self.core,
            )
            weight = torch.einsum(
                "ob,ibcd->iocd",
                self.output_factor.to(dtype=dtype),
                weight,
            )
            weight = torch.einsum(
                "yc,iocd->ioyd",
                self.y_factor[:keep_y].to(dtype=dtype),
                weight,
            )
            return torch.einsum(
                "xd,ioyd->ioyx",
                self.x_factor[:keep_x].to(dtype=dtype),
                weight,
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

    FactorizedBandOperator.__phc2d_architecture_module__ = "phc2d.models.full_cell_upgrades"
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
