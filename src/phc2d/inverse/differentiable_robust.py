"""Run paired nominal and worst-scenario differentiable design shards.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from phc2d.inverse.differentiable_objective import optimize_fourier_adam, optimize_robust_fourier_adam
from phc2d.inverse.differentiable_selection import rank_hard_trajectory, rank_robust_trajectory


def run_robust_shard(config: Mapping[str, Any], source: Any, predictor: Any, *, mode: str) -> dict[str, Any]:
    """Optimize from a shared initial point and retain the registered selection rule."""
    if mode not in ('nominal', 'worst'):
        raise ValueError('robust design mode must be nominal or worst')
    offsets = tuple(float(value) for value in config['offsets'])
    if len(offsets) != 3 or 0.0 not in offsets or not any(x < 0 for x in offsets) or not any(x > 0 for x in offsets):
        raise ValueError('nominal and both signed manufacturing offsets are required')
    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    shared = dict(coefficient_keys=keys, scales=config['scales'], lower=config['lower'], upper=config['upper'],
        master_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        temperature=config['temperature'], lower_band=config['lower_band'],
        target_frequency=config['target_frequency'], beta=config['beta'],
        learning_rate=config['learning_rate'], steps=config['evaluations'] - 1, initial=config['initial'])
    start = time.monotonic()
    if mode == 'nominal':
        history = optimize_fourier_adam(source, predictor, **shared)
    else:
        history = optimize_robust_fourier_adam(source, predictor, offsets=offsets, **shared)
    screening = dict(coefficient_keys=keys, max_mode_radius=config['max_mode_radius'],
        raster_method='master_block_mean_v1', source_resolution=config['source_resolution'],
        input_resolution=config['input_resolution'], material_fraction_bounds=tuple(config['material_fraction_bounds']),
        lower_band=config['lower_band'], target_frequency=config['target_frequency'])
    if mode == 'nominal':
        selected = rank_hard_trajectory(source, predictor, history, **screening)
        diagnostic = rank_robust_trajectory(source, predictor, history, offsets=offsets, **screening)
    else:
        selected = rank_robust_trajectory(source, predictor, history, offsets=offsets, **screening)
        diagnostic = None
    return {'scope': 'surrogate_only', 'mode': mode, 'history': history, 'selection': selected,
            'robust_diagnostic': diagnostic, 'offsets': list(offsets), 'elapsed_s': time.monotonic() - start,
            'objective_evaluations': len(history), 'scenario_inputs_per_objective': 1 if mode == 'nominal' else len(offsets),
            'selection_accounting': 'nominal_and_scenario_screens_recorded_separately'}
