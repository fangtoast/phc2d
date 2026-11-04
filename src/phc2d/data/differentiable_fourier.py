"""Differentiable rasterization of normalized full-cell Fourier geometry.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def fourier_material_weight(source: Any, parameters: Any, *,
                            coefficient_keys: Sequence[tuple[int, str]],
                            master_resolution: int, output_resolution: int,
                            temperature: float | None) -> Any:
    """Rasterize edits with the source normalization-grid semantics.

    The last parameter is the normalized-field threshold. A positive
    temperature gives a differentiable sigmoid; None gives the hard mask.
    Both branches use cell-centered master sampling and block averaging.
    """
    import torch

    if source.generator_family != 'unified_fourier_binary' or source.material_kind != 'binary':
        raise ValueError('expected unified Fourier binary structure')
    if not isinstance(parameters, torch.Tensor) or parameters.ndim != 1:
        raise ValueError('parameters must be a one-dimensional tensor')
    keys = tuple(coefficient_keys)
    if len(set(keys)) != len(keys) or len(parameters) != len(keys) + 1:
        raise ValueError('parameters must contain unique coefficients and a threshold')
    if not torch.isfinite(parameters).all():
        raise ValueError('parameters must be finite')
    if master_resolution <= 0 or output_resolution <= 0 or master_resolution % output_resolution:
        raise ValueError('output resolution must divide master resolution')
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('temperature must be positive and finite')
    modes = source.parameters['modes']
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(modes)
           or component not in ('cos', 'sin') for i, component in keys):
        raise ValueError('invalid coefficient key')
    edits = {key: parameters[index] for index, key in enumerate(keys)}

    def field(resolution: int) -> Any:
        points = (torch.arange(resolution, dtype=parameters.dtype, device=parameters.device) + 0.5) / resolution - 0.5
        y, x = torch.meshgrid(points, points, indexing='ij')
        result = torch.zeros_like(x)
        for index, mode in enumerate(modes):
            phase = 2 * math.pi * (int(mode['mx']) * x + int(mode['my']) * y)
            cosine = edits.get((index, 'cos'), float(mode.get('cos', 0.0)))
            sine = edits.get((index, 'sin'), float(mode.get('sin', 0.0)))
            result = result + cosine * torch.cos(phase) + sine * torch.sin(phase)
        return result

    norm = field(int(source.parameters['normalization_grid_resolution']))
    mean = norm.mean()
    std = norm.std(correction=0)
    if not torch.isfinite(std) or float(std.detach()) <= 1e-12:
        raise ValueError('edited Fourier field has zero or invalid variance')
    latent = (field(master_resolution) - mean) / std
    delta = latent - parameters[-1]
    weight = (delta >= 0).to(parameters.dtype) if temperature is None else torch.sigmoid(delta / temperature)
    factor = master_resolution // output_resolution
    return weight.reshape(output_resolution, factor, output_resolution, factor).mean(dim=(1, 3))
