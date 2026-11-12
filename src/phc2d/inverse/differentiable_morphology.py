"""Periodic differentiable raster offsets for robust inverse design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import math
from typing import Any


def periodic_soft_offset(weight: Any, *, offset: float) -> Any:
    """Apply disk max/min pooling with the hard cell-center offset convention.

    Positive offset dilates material; negative offset erodes it. On a binary
    input this matches ``periodic_binary_offset`` on the same grid. On a soft
    field, gradients flow through the selected extrema (with subgradients at
    ties). The operation is a finite-grid approximation, not subpixel geometry.
    """
    import torch

    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or weight.shape[0] != weight.shape[1] or weight.shape[0] == 0
            or not weight.is_floating_point() or not torch.isfinite(weight).all()
            or (weight < 0).any() or (weight > 1).any() or not math.isfinite(offset)):
        raise ValueError('finite square material weights in [0,1] and a finite offset required')
    size = weight.shape[0]
    radius = abs(offset) * size
    if radius < 1.0:
        return weight
    if radius >= size / 2:
        raise ValueError('offset radius must be below half the periodic grid')
    limit = math.ceil(radius)
    shifted = [torch.roll(weight, shifts=(dy, dx), dims=(0, 1))
               for dy in range(-limit, limit + 1) for dx in range(-limit, limit + 1)
               if math.hypot(dx, dy) <= radius + 1e-12]
    stack = torch.stack(shifted)
    return torch.amax(stack, dim=0) if offset > 0 else torch.amin(stack, dim=0)
