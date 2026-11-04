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
    edges = smooth_band_edges(bands, lower_band=lower_band, beta=beta)
    margins = torch.stack((target_frequency - edges[:, 0], edges[:, 1] - target_frequency), dim=1)
    return -2 * torch.logsumexp(-beta * margins, dim=1) / (beta * target_frequency)


def smooth_band_edges(bands: Any, *, lower_band: int, beta: float) -> Any:
    """Return differentiable upper/lower extrema of an adjacent band pair."""
    import torch

    if (bands.ndim != 3 or bands.shape[0] == 0 or bands.shape[1] == 0
            or isinstance(lower_band, bool) or not isinstance(lower_band, int)
            or not 1 <= lower_band < bands.shape[2] or not math.isfinite(beta) or beta <= 0):
        raise ValueError('invalid bands, band pair, or smoothing parameter')
    lower = torch.logsumexp(beta * bands[:, :, lower_band - 1], dim=1) / beta
    upper = -torch.logsumexp(-beta * bands[:, :, lower_band], dim=1) / beta
    return torch.stack((lower, upper), dim=1)


def corrected_fourier_margin(source: Any, predictor: Any, parameters: Any, normalized_point: Any,
                             correction: Any, *, coefficient_keys: Any, master_resolution: int,
                             input_resolution: int, temperature: float, lower_band: int,
                             target_frequency: float, beta: float) -> Any:
    """Use a frozen local composite-residual correction inside the soft objective."""
    import torch
    from phc2d.data.differentiable_fourier import fourier_material_weight

    if not math.isfinite(target_frequency) or target_frequency <= 0:
        raise ValueError('target frequency must be positive and finite')
    raster = fourier_material_weight(source, parameters, coefficient_keys=coefficient_keys,
                                     master_resolution=master_resolution,
                                     output_resolution=input_resolution, temperature=temperature)
    edges = smooth_band_edges(predictor.predict_tensor(raster.unsqueeze(0)),
                              lower_band=lower_band, beta=beta)[0]
    corrected = correction.correct_tensor(normalized_point, edges)
    margins = torch.stack((target_frequency - corrected[0], corrected[1] - target_frequency))
    return -2 * torch.logsumexp(-beta * margins, dim=0) / (beta * target_frequency)


def robust_fourier_margin(source: Any, predictor: Any, parameters: Any, *, coefficient_keys: Any,
                          master_resolution: int, input_resolution: int, temperature: float,
                          offsets: Any, lower_band: int, target_frequency: float, beta: float,
                          scenario_reduction: str = 'worst') -> Any:
    """Smooth target margin across periodic manufacturing scenarios."""
    import torch
    from phc2d.data.differentiable_fourier import fourier_material_weight
    from phc2d.inverse.differentiable_morphology import periodic_soft_offset

    values = tuple(float(offset) for offset in offsets)
    if (not values or 0.0 not in values or len(set(values)) != len(values)
            or any(not math.isfinite(value) for value in values)
            or not math.isfinite(target_frequency) or target_frequency <= 0
            or scenario_reduction not in ('worst', 'mean')):
        raise ValueError('finite unique manufacturing offsets including zero are required')
    weight = fourier_material_weight(source, parameters, coefficient_keys=coefficient_keys,
                                     master_resolution=master_resolution,
                                     output_resolution=master_resolution, temperature=temperature)
    if input_resolution <= 0 or master_resolution % input_resolution:
        raise ValueError('input resolution must divide the master resolution')
    factor = master_resolution // input_resolution
    scenarios = []
    for offset in values:
        shifted = periodic_soft_offset(weight, offset=offset)
        scenarios.append(shifted.reshape(input_resolution, factor, input_resolution, factor).mean(dim=(1, 3)))
    edges = smooth_band_edges(predictor.predict_tensor(torch.stack(scenarios)), lower_band=lower_band, beta=beta)
    margins = torch.stack((target_frequency - edges[:, 0], edges[:, 1] - target_frequency), dim=1)
    if scenario_reduction == 'mean':
        return (-2 * torch.logsumexp(-beta * margins, dim=1) / (beta * target_frequency)).mean()
    return -2 * torch.logsumexp(-beta * margins.flatten(), dim=0) / (beta * target_frequency)


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


def optimize_black_box(objective: Any, initial: Any, *, lower: Any, upper: Any,
                       evaluations: int, seed: int, method: str) -> list[dict[str, Any]]:
    """Run a reproducible random or differential-evolution soft-objective search."""
    import numpy as np
    import torch

    x = torch.as_tensor(initial).detach()
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if (x.ndim != 1 or not x.is_floating_point() or lo.shape != tuple(x.shape)
            or hi.shape != lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo >= hi) or not torch.isfinite(x).all()
            or np.any(x.cpu().numpy() < lo) or np.any(x.cpu().numpy() > hi)):
        raise ValueError('invalid initial point or bounds')
    if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 1:
        raise ValueError('positive evaluation budget required')
    if method not in ('random', 'de') or (method == 'de' and evaluations < 5):
        raise ValueError('invalid method or DE budget')
    rng = np.random.default_rng(seed)
    history = []

    def evaluate(vector: Any) -> float:
        if len(history) >= evaluations:
            raise RuntimeError('evaluation budget exhausted')
        vector = np.asarray(vector, dtype=float)
        with torch.no_grad():
            value = objective(torch.as_tensor(vector, dtype=x.dtype, device=x.device))
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError('objective must be a finite scalar')
        score = float(value.detach().cpu())
        history.append({'step': len(history), 'parameters': vector.tolist(), 'objective': score})
        return score

    if method == 'random':
        evaluate(x.cpu().numpy())
        for _ in range(evaluations - 1):
            evaluate(rng.uniform(lo, hi))
    else:
        population_size = min(5, evaluations)
        population = rng.uniform(lo, hi, size=(population_size, len(lo)))
        population[0] = x.cpu().numpy()
        scores = np.array([evaluate(v) for v in population])
        while len(history) < evaluations:
            for i in range(population_size):
                if len(history) == evaluations:
                    break
                pool = np.delete(np.arange(population_size), i)
                a, b, c = rng.choice(pool, size=3, replace=False)
                mutant = np.clip(population[a] + 0.8 * (population[b] - population[c]), lo, hi)
                mask = rng.random(len(lo)) < 0.7
                mask[rng.integers(len(lo))] = True
                trial = np.where(mask, mutant, population[i])
                score = evaluate(trial)
                if score > scores[i]:
                    population[i], scores[i] = trial, score
    return history


def optimize_fourier_adam(source: Any, predictor: Any, *, coefficient_keys: Any,
                          scales: Any, lower: Any, upper: Any, master_resolution: int,
                          input_resolution: int, temperature: float, lower_band: int,
                          target_frequency: float, beta: float, learning_rate: float,
                          steps: int, initial: Any = None) -> list[dict[str, Any]]:
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

    start = torch.zeros_like(base) if initial is None else torch.as_tensor(initial, dtype=base.dtype, device=base.device)
    history = optimize_adam(objective, start, lower=lower, upper=upper,
                            learning_rate=learning_rate, steps=steps)
    for record in history:
        z = torch.as_tensor(record['parameters'], dtype=base.dtype, device=base.device)
        record['fourier_parameters'] = (base + scale * z).detach().cpu().tolist()
    return history


def optimize_fourier_black_box(source: Any, predictor: Any, *, coefficient_keys: Any,
                               scales: Any, lower: Any, upper: Any, master_resolution: int,
                               input_resolution: int, temperature: float, lower_band: int,
                               target_frequency: float, beta: float, evaluations: int,
                               seed: int, method: str, initial: Any = None) -> list[dict[str, Any]]:
    """Search the same normalized geometry and smooth objective as Fourier Adam."""
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

    start = torch.zeros_like(base) if initial is None else torch.as_tensor(initial, dtype=base.dtype, device=base.device)
    history = optimize_black_box(objective, start, lower=lower, upper=upper,
                                 evaluations=evaluations, seed=seed, method=method)
    for record in history:
        z = torch.as_tensor(record['parameters'], dtype=base.dtype, device=base.device)
        record['fourier_parameters'] = (base + scale * z).detach().cpu().tolist()
    return history


def _corrected_search_setup(source: Any, predictor: Any, correction: Any, *,
                            coefficient_keys: Any, scales: Any, lower: Any, upper: Any,
                            master_resolution: int, input_resolution: int, temperature: float,
                            lower_band: int, target_frequency: float, beta: float) -> tuple[Any, ...]:
    """Construct the common corrected objective and inscribed local box."""
    import torch

    keys = tuple(coefficient_keys)
    base = torch.tensor([float(source.parameters['modes'][i][component]) for i, component in keys]
                        + [float(source.parameters['threshold'])], dtype=torch.float64,
                        device=predictor.device)
    scale = torch.as_tensor(scales, dtype=base.dtype, device=base.device)
    center = torch.as_tensor(correction.center, dtype=base.dtype, device=base.device)
    lo = torch.as_tensor(lower, dtype=base.dtype, device=base.device)
    hi = torch.as_tensor(upper, dtype=base.dtype, device=base.device)
    if (scale.shape != base.shape or center.shape != base.shape or lo.shape != base.shape
            or hi.shape != base.shape or not torch.isfinite(scale).all() or (scale <= 0).any()
            or not math.isfinite(correction.radius) or correction.radius <= 0):
        raise ValueError('invalid corrected optimization dimensions or scales')
    halfwidth = correction.radius / math.sqrt(len(base))
    lo = torch.maximum(lo, center - halfwidth)
    hi = torch.minimum(hi, center + halfwidth)

    def objective(z: Any) -> Any:
        return corrected_fourier_margin(source, predictor, base + scale*z, z, correction,
            coefficient_keys=keys, master_resolution=master_resolution, input_resolution=input_resolution,
            temperature=temperature, lower_band=lower_band, target_frequency=target_frequency, beta=beta)
    return objective, base, scale, center, lo, hi


def optimize_corrected_fourier_adam(source: Any, predictor: Any, correction: Any, *,
                                    coefficient_keys: Any, scales: Any, lower: Any, upper: Any,
                                    master_resolution: int, input_resolution: int,
                                    temperature: float, lower_band: int, target_frequency: float,
                                    beta: float, learning_rate: float, steps: int) -> list[dict[str, Any]]:
    """Optimize a frozen composite correction inside its local trust region.

    The parameter center is relative to the original source. The inscribed
    box keeps every optimizer evaluation within the fitted Euclidean radius.
    Candidate screening and physical acceptance are separate responsibilities.
    """
    import torch

    objective, base, scale, center, lo, hi = _corrected_search_setup(source, predictor, correction,
        coefficient_keys=coefficient_keys, scales=scales, lower=lower, upper=upper,
        master_resolution=master_resolution, input_resolution=input_resolution, temperature=temperature,
        lower_band=lower_band, target_frequency=target_frequency, beta=beta)

    history = optimize_adam(objective, center, lower=lo, upper=hi,
                            learning_rate=learning_rate, steps=steps)
    for row in history:
        z = torch.as_tensor(row['parameters'], dtype=base.dtype, device=base.device)
        row['fourier_parameters'] = (base + scale*z).detach().cpu().tolist()
    return history


def optimize_corrected_fourier_black_box(source: Any, predictor: Any, correction: Any, *,
                                         coefficient_keys: Any, scales: Any, lower: Any, upper: Any,
                                         master_resolution: int, input_resolution: int,
                                         temperature: float, lower_band: int, target_frequency: float,
                                         beta: float, evaluations: int, seed: int, method: str) -> list[dict[str, Any]]:
    """Search the same locally corrected objective without using its gradients."""
    import torch

    objective, base, scale, center, lo, hi = _corrected_search_setup(source, predictor, correction,
        coefficient_keys=coefficient_keys, scales=scales, lower=lower, upper=upper,
        master_resolution=master_resolution, input_resolution=input_resolution, temperature=temperature,
        lower_band=lower_band, target_frequency=target_frequency, beta=beta)

    history = optimize_black_box(objective, center, lower=lo.cpu().numpy(), upper=hi.cpu().numpy(),
                                 evaluations=evaluations, seed=seed, method=method)
    for row in history:
        z = torch.as_tensor(row['parameters'], dtype=base.dtype, device=base.device)
        row['fourier_parameters'] = (base + scale*z).detach().cpu().tolist()
    return history


def optimize_robust_fourier_adam(source: Any, predictor: Any, *, coefficient_keys: Any,
                                 scales: Any, lower: Any, upper: Any, master_resolution: int,
                                 input_resolution: int, temperature: float, offsets: Any,
                                 lower_band: int, target_frequency: float, beta: float,
                                 learning_rate: float, steps: int, initial: Any = None,
                                 scenario_reduction: str = 'worst') -> list[dict[str, Any]]:
    """Optimize the shared target margin across differentiable periodic offsets."""
    import torch

    keys = tuple(coefficient_keys)
    base = torch.tensor([float(source.parameters['modes'][i][component]) for i, component in keys]
                        + [float(source.parameters['threshold'])], dtype=torch.float64,
                        device=predictor.device)
    scale = torch.as_tensor(scales, dtype=base.dtype, device=base.device)
    if scale.shape != base.shape or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('one finite positive scale per parameter required')

    def objective(z: Any) -> Any:
        return robust_fourier_margin(source, predictor, base + scale*z,
            coefficient_keys=keys, master_resolution=master_resolution, input_resolution=input_resolution,
            temperature=temperature, offsets=offsets, lower_band=lower_band,
            target_frequency=target_frequency, beta=beta, scenario_reduction=scenario_reduction)

    start = torch.zeros_like(base) if initial is None else torch.as_tensor(initial, dtype=base.dtype, device=base.device)
    history = optimize_adam(objective, start, lower=lower, upper=upper,
                            learning_rate=learning_rate, steps=steps)
    for row in history:
        z = torch.as_tensor(row['parameters'], dtype=base.dtype, device=base.device)
        row['fourier_parameters'] = (base + scale*z).detach().cpu().tolist()
    return history
