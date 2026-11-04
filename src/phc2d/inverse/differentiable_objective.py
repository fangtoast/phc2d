"""Differentiable sampled bandgap objectives for inverse design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Any


def fourier_surrogate_margin(source: Any, predictor: Any, parameters: Any, *,
                             coefficient_keys: Any, master_resolution: int,
                             input_resolution: int, temperature: float,
                             lower_band: int, target_frequency: float, beta: float) -> Any:
    """Evaluate the smooth surrogate objective from analytic geometry parameters."""
    from phc2d.data.differentiable_fourier import fourier_material_weight

    raster = fourier_material_weight(source, parameters, coefficient_keys=coefficient_keys,
                                     master_resolution=master_resolution,
                                     output_resolution=input_resolution, temperature=temperature)
    bands = predictor.predict_tensor(raster.unsqueeze(0))
    return smooth_target_margin(bands, lower_band=lower_band,
                                target_frequency=target_frequency, beta=beta)[0]


def smooth_target_margin(bands: Any, *, lower_band: int, target_frequency: float,
                         beta: float) -> Any:
    """Conservative smooth sampled margin for a batch of band matrices.

    Band numbers are one-based. Exact max/min should be used to rank hard
    geometries; this smooth objective is only an optimization approximation.
    """
    import torch

    if (not math.isfinite(target_frequency) or target_frequency <= 0
            or not math.isfinite(beta) or beta <= 0):
        raise ValueError('target frequency and beta must be finite and positive')
    if (bands.ndim != 3 or bands.shape[0] == 0 or bands.shape[1] == 0
            or isinstance(lower_band, bool) or not isinstance(lower_band, int)
            or not 1 <= lower_band < bands.shape[2]):
        raise ValueError('expected batch, k and adjacent band dimensions')
    lower = torch.logsumexp(beta * bands[:, :, lower_band - 1], dim=1) / beta
    upper = -torch.logsumexp(-beta * bands[:, :, lower_band], dim=1) / beta
    margins = torch.stack((target_frequency - lower, upper - target_frequency), dim=1)
    return -2 * torch.logsumexp(-beta * margins, dim=1) / (beta * target_frequency)


def optimize_adam(objective: Any, initial: Any, *, lower: Any, upper: Any,
                  learning_rate: float, steps: int) -> list[dict[str, Any]]:
    """Maximize a scalar objective inside a box and return the full trajectory.

    The callback owns geometry, predictor and objective construction. Every
    record is a pre-update evaluation; the final projected point is included.
    """
    import torch

    if not math.isfinite(learning_rate) or learning_rate <= 0 or isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('invalid optimization settings')
    x = torch.as_tensor(initial).detach().clone()
    if not x.is_floating_point() or x.ndim != 1 or not torch.isfinite(x).all():
        raise ValueError('initial parameters must be a finite floating vector')
    lo = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    hi = torch.as_tensor(upper, dtype=x.dtype, device=x.device)
    if lo.shape != x.shape or hi.shape != x.shape or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (lo > hi).any() or (x < lo).any() or (x > hi).any():
        raise ValueError('invalid box bounds or initial point')
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad()
        value = objective(x)
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError('objective must be a finite scalar')
        grad, = torch.autograd.grad(value, x)
        if not torch.isfinite(grad).all():
            raise ValueError('objective gradient must be finite')
        history.append({'step': step, 'parameters': x.detach().cpu().tolist(),
                        'objective': float(value.detach().cpu()),
                        'gradient': grad.detach().cpu().tolist()})
        if step < steps:
            x.grad = -grad
            optimizer.step()
            with torch.no_grad():
                x.clamp_(min=lo, max=hi)
    return history


def optimize_fourier_adam(source: Any, predictor: Any, *, coefficient_keys: Any,
                          scales: Any, lower: Any, upper: Any, master_resolution: int,
                          input_resolution: int, temperature: float, lower_band: int,
                          target_frequency: float, beta: float, learning_rate: float,
                          steps: int) -> list[dict[str, Any]]:
    """Optimize normalized Fourier edits and return soft-objective evaluations.

    Selection and hard-geometry eligibility remain the responsibility of the
    existing Fourier screening interface; this function makes no MPB calls.
    """
    import torch

    keys = tuple(coefficient_keys)
    base = torch.tensor([float(source.parameters['modes'][i][component]) for i, component in keys]
                        + [float(source.parameters['threshold'])], dtype=torch.float64,
                        device=predictor.device)
    scale = torch.as_tensor(scales, dtype=base.dtype, device=base.device)
    if scale.shape != base.shape or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('one finite positive scale per parameter required')

    def objective(z: Any) -> Any:
        return fourier_surrogate_margin(source, predictor, base + scale * z,
                                        coefficient_keys=keys, master_resolution=master_resolution,
                                        input_resolution=input_resolution, temperature=temperature,
                                        lower_band=lower_band, target_frequency=target_frequency,
                                        beta=beta)

    history = optimize_adam(objective, torch.zeros_like(base), lower=lower, upper=upper,
                            learning_rate=learning_rate, steps=steps)
    for record in history:
        z = torch.as_tensor(record['parameters'], dtype=base.dtype, device=base.device)
        record['fourier_parameters'] = (base + scale * z).detach().cpu().tolist()
    return history
