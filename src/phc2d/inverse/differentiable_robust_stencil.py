"""Predict frozen local feedback stencils under manufacturing scenarios.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


def predict_stencil_scenarios(stencil: Mapping[str, Any], source: Any, predictor: Any, *,
                              base: Mapping[str, Any], offsets: Sequence[float]) -> dict[str, Any]:
    """Save hard and smooth scenario spectra before obtaining physical feedback."""
    import torch
    from phc2d.data.differentiable_fourier import fourier_material_weight
    from phc2d.inverse.differentiable_morphology import periodic_soft_offset
    from phc2d.inverse.differentiable_objective import smooth_band_edges

    values = tuple(float(value) for value in offsets)
    if (len(values) != 3 or len(set(values)) != 3 or 0.0 not in values or not np.isfinite(values).all()
            or not any(x < 0 for x in values) or not any(x > 0 for x in values)
            or stencil.get('source_parameter_hash') != source.parameter_hash):
        raise ValueError('invalid stencil source or manufacturing scenarios')
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = torch.tensor([source.parameters['modes'][i][component] for i, component in keys]
                             + [source.parameters['threshold']], dtype=torch.float64, device=predictor.device)
    scales = torch.as_tensor(base['scales'], dtype=reference.dtype, device=reference.device)
    resolution, inputs = base['source_resolution'], base['input_resolution']
    if scales.shape != reference.shape or inputs <= 0 or resolution % inputs:
        raise ValueError('invalid stencil geometry resolution or scale')
    factor, pair, beta = resolution // inputs, base['lower_band'], float(base['beta'])
    rows = []
    for record in stencil['records']:
        candidate = record['candidate']
        if candidate is None or candidate['status'] not in ('surrogate_only', 'duplicate_geometry'):
            rows.append({'label': record['label'], 'status': record.get('status',
                         'invalid_candidate' if candidate is None else candidate['status'])})
            continue
        point = torch.as_tensor(record['point'], dtype=reference.dtype, device=reference.device)
        if point.shape != reference.shape or not torch.isfinite(point).all():
            raise ValueError('invalid stencil normalized point')
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        grids = periodic_perturbation_grids(structure, resolution=resolution, offsets=values)
        nominal_hash = hashlib.sha256(np.ascontiguousarray(grids[values.index(0.0)], dtype=np.uint8).tobytes()).hexdigest()
        if structure.parameter_hash != candidate['parameter_hash'] or nominal_hash != candidate['master_grid_sha256']:
            raise ValueError('stencil candidate geometry identity mismatch')
        hard = predict_perturbed_gap(grids, predictor, input_resolution=inputs, lower_band=pair,
                                     target_frequency=base['target_frequency'])
        item = {'label': record['label'], 'point': record['point'], 'parameter_hash': candidate['parameter_hash'],
                'hard': hard}
        if hard['status'] != 'surrogate_only':
            rows.append({**item, 'status': hard['status']})
            continue
        with torch.no_grad():
            weight = fourier_material_weight(source, reference + scales*point, coefficient_keys=keys,
                master_resolution=resolution, output_resolution=resolution, temperature=base['temperature'])
            scenarios = [periodic_soft_offset(weight, offset=value).reshape(inputs, factor, inputs, factor).mean(dim=(1, 3))
                         for value in values]
            bands = predictor.predict_tensor(torch.stack(scenarios))
            smooth = smooth_band_edges(bands, lower_band=pair, beta=beta).detach().cpu().numpy()
            raw = bands.detach().cpu().numpy()
        if not np.isfinite(raw).all() or not np.isfinite(smooth).all():
            raise ValueError('nonfinite soft stencil spectrum')
        hard_bands = np.asarray(hard['bands'], dtype=np.float64)
        exact = np.stack((raw[:, :, pair-1].max(axis=1), raw[:, :, pair].min(axis=1)), axis=1)
        hard_edges = np.stack((hard_bands[:, :, pair-1].max(axis=1), hard_bands[:, :, pair].min(axis=1)), axis=1)
        rows.append({**item, 'status': 'computed', 'soft_bands': raw.tolist(),
                     'soft_smooth_edges': smooth.tolist(), 'soft_exact_edges': exact.tolist(),
                     'hard_edges': hard_edges.tolist()})
    return {'scope': 'robust_stencil_surrogate_only', 'source_parameter_hash': source.parameter_hash,
            'initial': stencil['initial'], 'offsets': list(values), 'records': rows}
