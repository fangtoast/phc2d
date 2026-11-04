"""Prepare the registered local observation stencil for one feedback seed.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.differentiable_soft_audit import audit_locked_soft


def prepare_feedback_stencil(source: Any, predictor: Any, *, base: Mapping[str, Any],
                             campaign: Mapping[str, Any], initial: list[float]) -> dict[str, Any]:
    """Evaluate fixed nominal stencil geometries without any MPB observation."""
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = np.asarray([source.parameters['modes'][i][component] for i, component in keys]
                           + [source.parameters['threshold']], dtype=np.float64)
    center = np.asarray(initial, dtype=np.float64)
    scale, lo, hi = (np.asarray(base[key], dtype=np.float64) for key in ('scales', 'lower', 'upper'))
    if (center.shape != reference.shape or scale.shape != center.shape or lo.shape != center.shape
            or hi.shape != center.shape or not all(np.isfinite(x).all() for x in (center, scale, lo, hi))
            or np.any(scale <= 0) or np.any(center < lo) or np.any(center > hi)):
        raise ValueError('invalid feedback seed or parameter contract')
    offsets = [np.asarray(row, dtype=np.float64) for row in campaign['initial_offsets']]
    if (len(offsets) != 5 or any(row.shape != center.shape or not np.isfinite(row).all() for row in offsets)
            or len({tuple(row) for row in offsets}) != len(offsets)
            or not np.all(offsets[0] == 0)):
        raise ValueError('invalid preregistered feedback stencil')
    history, indices, records = [], [], [None] * len(offsets)
    for index, offset in enumerate(offsets):
        point = center + offset
        label = f'offset_{index}'
        if np.any(point < lo) or np.any(point > hi):
            records[index] = {'label': label, 'point': point.tolist(), 'candidate': None,
                              'status': 'outside_parameter_bounds'}
            continue
        indices.append(index)
        history.append({'parameters': point.tolist(), 'fourier_parameters': (reference + scale * point).tolist()})
    if history:
        screening = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
            max_mode_radius=base['max_mode_radius'], raster_method='master_block_mean_v1',
            source_resolution=base['source_resolution'], input_resolution=base['input_resolution'],
            material_fraction_bounds=tuple(base['material_fraction_bounds']),
            lower_band=base['lower_band'], target_frequency=base['target_frequency'])
        for index, candidate, item in zip(indices, screening['candidates'], history):
            records[index] = {'label': f'offset_{index}', 'point': item['parameters'], 'candidate': candidate}
    soft = audit_locked_soft({'records': records}, predictor, base)
    return {'scope': 'feedback_initial_stencil_surrogate_only', 'source_parameter_hash': source.parameter_hash,
            'initial': center.tolist(), 'records': records, 'soft_audit': soft}
