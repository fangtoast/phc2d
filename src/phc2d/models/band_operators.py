"""Spectral operator builders for Full-Wh forward baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models._common import _numel, _require_torch, _shape_tuple


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
        raise ValueError(
            "fno_lite input_shape must be (height, width) or (channels, height, width)"
        )
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

    FnoLite.__phc2d_architecture_module__ = "phc2d.models.baselines"
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
        raise ValueError(
            "fno2d_band input_shape must be (height, width) or (channels, height, width)"
        )
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

    Fno2DBand.__phc2d_architecture_module__ = "phc2d.models.baselines"
    return Fno2DBand()
