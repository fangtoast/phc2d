"""Legacy schema-size-agnostic U-Net and FNO field builders.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models._field_common import _torch


def build_unet_field(
    *, input_channels: int = 4, field_channels: int = 12, width: int = 32, depth: int = 2, **_: Any
) -> Any:
    """Build a schema-size-agnostic U-Net with a scalar frequency head."""
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, a: int, b: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU()
            )

        def forward(self, x):
            return self.net(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList()
            channels = []
            current = int(input_channels)
            for index in range(max(1, int(depth))):
                out = int(width) * (2**index)
                self.enc.append(Block(current, out))
                channels.append(out)
                current = out
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = Block(current, current * 2)
            bottleneck_channels = current * 2
            self.up = nn.ModuleList()
            self.dec = nn.ModuleList()
            current *= 2
            for skip in reversed(channels):
                self.up.append(nn.ConvTranspose2d(current, skip, 2, stride=2))
                self.dec.append(Block(skip * 2, skip))
                current = skip
            self.field_head = nn.Conv2d(current, int(field_channels), 1)
            self.freq_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bottleneck_channels, 1)
            )

        def forward(self, x):
            skips = []
            for block in self.enc:
                x = block(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            freq = self.freq_head(x).squeeze(-1)
            for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = torch.nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    )
                x = dec(torch.cat((x, skip), dim=1))
            return self.field_head(x), freq

    Model.__phc2d_architecture_module__ = "phc2d.models.field"
    return Model()


def build_fno_field(
    *,
    input_channels: int = 4,
    field_channels: int = 12,
    width: int = 32,
    depth: int = 4,
    modes: tuple[int, int] = (12, 12),
    **_: Any,
) -> Any:
    """Build a compact FNO-style field operator with a frequency head."""
    torch, nn = _torch()

    class Spectral(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(width, width, int(modes[0]), int(modes[1]), dtype=torch.cfloat) * 0.02
            )

        def forward(self, x):
            b, c, h, w = x.shape
            fx = torch.fft.rfft2(x, norm="ortho")
            out = torch.zeros_like(fx)
            mh = min(self.weight.shape[2], h)
            mw = min(self.weight.shape[3], fx.shape[-1])
            out[:, :, :mh, :mw] = torch.einsum(
                "bixy,ioxy->boxy", fx[:, :, :mh, :mw], self.weight[:, :, :mh, :mw]
            )
            return torch.fft.irfft2(out, s=(h, w), norm="ortho")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Conv2d(int(input_channels), int(width), 1)
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleList([Spectral(), nn.Conv2d(int(width), int(width), 1)])
                    for _ in range(max(1, int(depth)))
                ]
            )
            self.head = nn.Conv2d(int(width), int(field_channels), 1)
            self.freq = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(int(width), 1)
            )

        def forward(self, x):
            x = self.lift(x)
            for spectral, point in self.blocks:
                x = torch.nn.functional.gelu(spectral(x) + point(x))
            return self.head(x), self.freq(x).squeeze(-1)

    Model.__phc2d_architecture_module__ = "phc2d.models.field"
    return Model()
