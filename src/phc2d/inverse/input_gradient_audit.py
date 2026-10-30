"""Directional derivative diagnostics for a frozen band predictor.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def sampled_target_margin(predictor: Any, inputs: Any, *, lower_band: int,
                          target_frequency: float) -> Any:
    """Evaluate a differentiable sampled margin for a predictor batch."""
    import torch

    if not math.isfinite(target_frequency) or target_frequency <= 0:
        raise ValueError('invalid target frequency')
    bands = predictor.predict_tensor(inputs)
    if bands.ndim != 3 or bands.shape[0] != len(inputs) or not 1 <= lower_band < bands.shape[2]:
        raise ValueError('invalid predictor output or adjacent band pair')
    lower = torch.amax(bands[:, :, lower_band - 1], dim=1)
    upper = torch.amin(bands[:, :, lower_band], dim=1)
    return 2 * torch.minimum(target_frequency - lower, upper - target_frequency) / target_frequency


def audit_input_gradient(predictor: Any, image: Any, *, lower_band: int,
                         target_frequency: float, steps: Sequence[float], seed: int) -> dict[str, Any]:
    """Compare autograd and centered finite differences in continuous input space.

    This is a mathematical diagnostic only. The perturbed inputs need not be
    binary, feasible, in distribution, or realizable by a geometry change.
    """
    import torch

    if not math.isfinite(target_frequency) or target_frequency <= 0:
        raise ValueError('invalid target frequency')
    values = [float(step) for step in steps]
    if not values or any(not math.isfinite(step) or step <= 0 for step in values):
        raise ValueError('finite positive finite-difference steps required')
    x = torch.as_tensor(image, device=predictor.device, dtype=torch.float32).detach().clone()
    if x.ndim < 2 or x.shape[0] != 1 or not torch.isfinite(x).all():
        raise ValueError('one finite batch input is required')
    x.requires_grad_(True)

    def objective(inputs: Any) -> Any:
        return sampled_target_margin(predictor, inputs, lower_band=lower_band,
                                     target_frequency=target_frequency)[0]

    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    direction = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device)
    direction = direction / torch.linalg.vector_norm(direction)
    score = objective(x)
    gradient, = torch.autograd.grad(score, x)
    derivative = float(torch.sum(gradient * direction).detach().cpu())
    norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
    if not all(math.isfinite(v) for v in (float(score.detach().cpu()), derivative, norm)):
        raise ValueError('nonfinite gradient or objective')
    estimates = []
    with torch.no_grad():
        for step in values:
            plus = float(objective(x + step * direction).cpu())
            minus = float(objective(x - step * direction).cpu())
            fd = (plus - minus) / (2 * step)
            estimates.append({'step': step, 'central_difference': fd,
                              'absolute_error': abs(fd - derivative),
                              'relative_error': abs(fd - derivative) / max(abs(fd), abs(derivative), 1e-12)})
    return {'status': 'continuous_input_diagnostic_only', 'seed': int(seed),
            'objective': float(score.detach().cpu()), 'gradient_norm': norm,
            'autograd_directional_derivative': derivative, 'finite_differences': estimates}
