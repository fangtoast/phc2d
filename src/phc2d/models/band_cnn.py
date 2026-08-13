"""Convolutional builder for the lifted-grid Full-Wh baseline.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models.torch_support import numel, require_torch, shape_tuple


def build_lifted_cnn(
    *,
    input_shape: Any,
    output_shape: Any,
    channels: int,
    activation: str = "gelu",
) -> Any:
    """Build a tiny CNN baseline for grid/image representations."""

    _, nn = require_torch("lifted_cnn")
    in_shape = shape_tuple(input_shape, name="input_shape")
    out_shape = shape_tuple(output_shape, name="output_shape")
    if len(in_shape) == 2:
        in_channels = 1
    elif len(in_shape) == 3:
        in_channels = int(in_shape[0])
    else:
        raise ValueError(
            "lifted_cnn input_shape must be (height, width) or (channels, height, width)"
        )
    width = int(channels)
    if width <= 0:
        raise ValueError("channels must be positive")
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU

    class LiftedCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = in_shape
            self.output_shape = out_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                activation_layer(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                activation_layer(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
            )
            self.head = nn.Linear(width, numel(out_shape))

        def forward(self, x: Any) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            y = self.head(self.encoder(x))
            return y.reshape(x.shape[0], *self.output_shape)

    LiftedCNN.__phc2d_architecture_module__ = "phc2d.models.baselines"
    return LiftedCNN()
