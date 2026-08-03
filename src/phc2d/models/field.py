"""Trainable grid-to-field baselines for TE Bloch eigenfields.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Field models require optional torch.") from exc
    return torch, nn


def build_unet_field(*, input_channels: int = 4, field_channels: int = 12, width: int = 32, depth: int = 2, **_: Any) -> Any:
    """Build a schema-size-agnostic U-Net with a scalar frequency head."""
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, a: int, b: int):
            super().__init__(); self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.GELU(), nn.Conv2d(b,b,3,padding=1), nn.GELU())
        def forward(self, x): return self.net(x)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList(); channels=[]; current=int(input_channels)
            for index in range(max(1, int(depth))):
                out=int(width)*(2**index); self.enc.append(Block(current,out)); channels.append(out); current=out
            self.pool=nn.MaxPool2d(2); self.bottleneck=Block(current,current*2); bottleneck_channels=current*2
            self.up=nn.ModuleList(); self.dec=nn.ModuleList(); current*=2
            for skip in reversed(channels):
                self.up.append(nn.ConvTranspose2d(current,skip,2,stride=2)); self.dec.append(Block(skip*2,skip)); current=skip
            self.field_head=nn.Conv2d(current,int(field_channels),1); self.freq_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(bottleneck_channels,1))
        def forward(self,x):
            skips=[]
            for block in self.enc: x=block(x); skips.append(x); x=self.pool(x)
            x=self.bottleneck(x); freq=self.freq_head(x).squeeze(-1)
            for up,dec,skip in zip(self.up,self.dec,reversed(skips)):
                x=up(x)
                if x.shape[-2:] != skip.shape[-2:]: x=torch.nn.functional.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
                x=dec(torch.cat((x,skip),dim=1))
            return self.field_head(x),freq
    return Model()


def build_fno_field(*, input_channels: int = 4, field_channels: int = 12, width: int = 32, depth: int = 4, modes: tuple[int, int] = (12, 12), **_: Any) -> Any:
    """Build a compact FNO-style field operator with a frequency head."""
    torch, nn = _torch()
    class Spectral(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.randn(width,width,int(modes[0]),int(modes[1]),dtype=torch.cfloat)*0.02)
        def forward(self,x):
            b,c,h,w=x.shape; fx=torch.fft.rfft2(x,norm='ortho'); out=torch.zeros_like(fx); mh=min(self.weight.shape[2],h); mw=min(self.weight.shape[3],fx.shape[-1]); out[:,:,:mh,:mw]=torch.einsum('bixy,ioxy->boxy',fx[:,:,:mh,:mw],self.weight[:,:,:mh,:mw]); return torch.fft.irfft2(out,s=(h,w),norm='ortho')
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.lift=nn.Conv2d(int(input_channels),int(width),1); self.blocks=nn.ModuleList([nn.ModuleList([Spectral(),nn.Conv2d(int(width),int(width),1)]) for _ in range(max(1,int(depth)))]); self.head=nn.Conv2d(int(width),int(field_channels),1); self.freq=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(int(width),1))
        def forward(self,x):
            x=self.lift(x)
            for spectral,point in self.blocks: x=torch.nn.functional.gelu(spectral(x)+point(x))
            return self.head(x),self.freq(x).squeeze(-1)
    return Model()


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if int(channels) % candidate == 0:
            return candidate
    return 1


def build_unet_field_v2(
    *,
    grid_channels: int = 5,
    context_dim: int = 3,
    field_channels: int = 6,
    width: int = 48,
    depth: int = 3,
    use_film: bool = True,
    **_: Any,
) -> Any:
    """Build a periodic, context-conditioned residual U-Net for active TE fields."""

    torch, nn = _torch()

    class PeriodicConv(nn.Module):
        def __init__(self, input_width: int, output_width: int, kernel_size: int = 3):
            super().__init__()
            self.padding = kernel_size // 2
            self.conv = nn.Conv2d(input_width, output_width, kernel_size, padding=0)

        def forward(self, values):
            values = torch.nn.functional.pad(values, (self.padding,) * 4, mode="circular")
            return self.conv(values)

    class ResidualBlock(nn.Module):
        def __init__(self, input_width: int, output_width: int):
            super().__init__()
            self.conv1 = PeriodicConv(input_width, output_width)
            self.conv2 = PeriodicConv(output_width, output_width)
            self.norm1 = nn.GroupNorm(_group_count(output_width), output_width)
            self.norm2 = nn.GroupNorm(_group_count(output_width), output_width)
            self.skip = nn.Conv2d(input_width, output_width, 1) if input_width != output_width else nn.Identity()
            self.film = nn.Linear(int(context_dim), 2 * output_width) if use_film else None

        def forward(self, values, context):
            hidden = torch.nn.functional.gelu(self.norm1(self.conv1(values)))
            hidden = self.norm2(self.conv2(hidden))
            if self.film is not None:
                scale, shift = self.film(context).chunk(2, dim=-1)
                hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            return torch.nn.functional.gelu(hidden + self.skip(values))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [int(width) * (2**index) for index in range(max(1, int(depth)))]
            self.encoder = nn.ModuleList()
            self.down = nn.ModuleList()
            current = int(grid_channels)
            for output_width in widths:
                self.encoder.append(ResidualBlock(current, output_width))
                self.down.append(nn.Conv2d(output_width, output_width, 2, stride=2))
                current = output_width
            self.bottleneck = ResidualBlock(current, current * 2)
            bottleneck_width = current * 2
            current = bottleneck_width
            self.up_projection = nn.ModuleList()
            self.decoder = nn.ModuleList()
            for skip_width in reversed(widths):
                self.up_projection.append(nn.Conv2d(current, skip_width, 1))
                self.decoder.append(ResidualBlock(skip_width * 2, skip_width))
                current = skip_width
            self.field_head = nn.Conv2d(current, int(field_channels), 1)
            self.frequency_head = nn.Sequential(
                nn.Linear(bottleneck_width + int(context_dim), bottleneck_width),
                nn.GELU(),
                nn.Linear(bottleneck_width, 1),
            )

        def forward(self, grid, context):
            skips = []
            hidden = grid
            for block, down in zip(self.encoder, self.down):
                hidden = block(hidden, context)
                skips.append(hidden)
                hidden = down(hidden)
            hidden = self.bottleneck(hidden, context)
            pooled = torch.nn.functional.adaptive_avg_pool2d(hidden, 1).flatten(1)
            frequency = self.frequency_head(torch.cat((pooled, context), dim=1)).squeeze(-1)
            for projection, block, skip in zip(self.up_projection, self.decoder, reversed(skips)):
                hidden = torch.nn.functional.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                hidden = projection(hidden)
                hidden = block(torch.cat((hidden, skip), dim=1), context)
            return self.field_head(hidden), frequency

    return Model()


def build_fno_field_v2(
    *,
    grid_channels: int = 5,
    context_dim: int = 3,
    field_channels: int = 6,
    width: int = 48,
    depth: int = 4,
    modes: tuple[int, int] = (16, 16),
    use_film: bool = True,
    corrected_spectral_modes: bool = True,
    **_: Any,
) -> Any:
    """Build a conditioned FNO with positive and negative first-axis modes."""

    torch, nn = _torch()
    resolved_width = int(width)
    modes_x, modes_y = (int(value) for value in modes)

    class SpectralConv(nn.Module):
        def __init__(self):
            super().__init__()
            scale = 1.0 / max(1, resolved_width)
            self.weight_positive = nn.Parameter(scale * torch.randn(resolved_width, resolved_width, modes_x, modes_y, dtype=torch.cfloat))
            self.weight_negative = nn.Parameter(scale * torch.randn(resolved_width, resolved_width, modes_x, modes_y, dtype=torch.cfloat)) if corrected_spectral_modes else None

        def _multiply(self, values, weights):
            return torch.einsum("bixy,ioxy->boxy", values, weights)

        def forward(self, values):
            batch, _, nx, ny = values.shape
            transformed = torch.fft.rfft2(values, norm="ortho")
            output = torch.zeros((batch, resolved_width, nx, transformed.shape[-1]), dtype=transformed.dtype, device=values.device)
            mx = min(modes_x, max(1, nx // 2))
            my = min(modes_y, transformed.shape[-1])
            output[:, :, :mx, :my] = self._multiply(transformed[:, :, :mx, :my], self.weight_positive[:, :, :mx, :my])
            if self.weight_negative is not None:
                output[:, :, -mx:, :my] = self._multiply(transformed[:, :, -mx:, :my], self.weight_negative[:, :, :mx, :my])
            return torch.fft.irfft2(output, s=(nx, ny), norm="ortho")

    class OperatorBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectral = SpectralConv()
            self.local = nn.Conv2d(resolved_width, resolved_width, 1)
            self.norm = nn.GroupNorm(_group_count(resolved_width), resolved_width)
            self.film = nn.Linear(int(context_dim), 2 * resolved_width) if use_film else None

        def forward(self, values, context):
            hidden = self.norm(self.spectral(values) + self.local(values))
            if self.film is not None:
                scale, shift = self.film(context).chunk(2, dim=-1)
                hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            return values + torch.nn.functional.gelu(hidden)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Conv2d(int(grid_channels), resolved_width, 1)
            self.blocks = nn.ModuleList([OperatorBlock() for _ in range(max(1, int(depth)))])
            self.field_head = nn.Sequential(nn.Conv2d(resolved_width, resolved_width, 1), nn.GELU(), nn.Conv2d(resolved_width, int(field_channels), 1))
            self.frequency_head = nn.Sequential(nn.Linear(resolved_width + int(context_dim), resolved_width), nn.GELU(), nn.Linear(resolved_width, 1))

        def forward(self, grid, context):
            hidden = self.lift(grid)
            for block in self.blocks:
                hidden = block(hidden, context)
            pooled = torch.nn.functional.adaptive_avg_pool2d(hidden, 1).flatten(1)
            frequency = self.frequency_head(torch.cat((pooled, context), dim=1)).squeeze(-1)
            return self.field_head(hidden), frequency

    return Model()
