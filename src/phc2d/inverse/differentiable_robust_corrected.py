"""Generate robust local proposals from scenario-specific physical feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from phc2d.inverse.differentiable_objective import optimize_corrected_robust_adam
from phc2d.inverse.differentiable_robust_feedback import fit_scenario_corrections
from phc2d.inverse.differentiable_selection import rank_robust_trajectory


def propose_corrected_robust_design(source: Any, predictor: Any, observations: Sequence[Mapping[str, Any]], *,
                                    config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit and freeze a separate band-edge residual for each manufacturing scenario."""
    if (predictor.checkpoint_sha256 != config['checkpoint_sha256']
            or any(row['observation']['surrogate_contract_sha256'] != config['surrogate_contract_sha256']
                   for row in observations)):
        raise ValueError('robust feedback surrogate identity mismatch')
    offsets = tuple(float(value) for value in config['offsets'])
    center, radius = config['center'], float(config['radius'])
    corrections = fit_scenario_corrections(observations, offsets=offsets, center=center,
                                           radius=radius, ridge=config['ridge'])
    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    history = optimize_corrected_robust_adam(source, predictor, corrections, coefficient_keys=keys,
        scales=config['scales'], lower=config['lower'], upper=config['upper'],
        master_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        temperature=config['temperature'], offsets=offsets, lower_band=config['lower_band'],
        target_frequency=config['target_frequency'], beta=config['beta'], learning_rate=config['learning_rate'],
        steps=config['evaluations'] - 1, scenario_reduction=config.get('scenario_reduction', 'worst'))
    screening = rank_robust_trajectory(source, predictor, history, offsets=offsets,
        score_reduction=config.get('scenario_reduction', 'worst'), coefficient_keys=keys,
        max_mode_radius=config['max_mode_radius'], raster_method='master_block_mean_v1',
        source_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        material_fraction_bounds=tuple(config['material_fraction_bounds']), lower_band=config['lower_band'],
        target_frequency=config['target_frequency'])
    eligible = [row['index'] for row in screening['candidates'] if row['status'] == 'surrogate_only']
    best = max(eligible, key=lambda index: (history[index]['objective'], -index)) if eligible else None
    prediction = None
    if best is not None:
        import torch
        from phc2d.data.differentiable_fourier import fourier_material_weight
        from phc2d.inverse.differentiable_morphology import periodic_soft_offset
        from phc2d.inverse.differentiable_objective import smooth_band_edges

        entry = history[best]
        parameters = torch.as_tensor(entry['fourier_parameters'], dtype=torch.float64, device=predictor.device)
        point = torch.as_tensor(entry['parameters'], dtype=torch.float64, device=predictor.device)
        size, output = config['source_resolution'], config['input_resolution']
        factor = size // output
        with torch.no_grad():
            weight = fourier_material_weight(source, parameters, coefficient_keys=keys, master_resolution=size,
                                             output_resolution=size, temperature=config['temperature'])
            scenarios = [periodic_soft_offset(weight, offset=value).reshape(output, factor, output, factor).mean(dim=(1, 3))
                         for value in offsets]
            soft_bands = predictor.predict_tensor(torch.stack(scenarios))
            raw_edges = smooth_band_edges(soft_bands, lower_band=config['lower_band'], beta=config['beta'])
            corrected_edges = torch.stack([correction.correct_tensor(point, raw_edges[i])
                                           for i, correction in enumerate(corrections)])
        prediction = {'point': entry['parameters'], 'soft_bands': soft_bands.detach().cpu().tolist(),
                      'soft_smooth_edges': raw_edges.detach().cpu().tolist(),
                      'corrected_smooth_edges': corrected_edges.detach().cpu().tolist(),
                      'hard_scenarios': screening['candidates'][best]['robust_prediction'],
                      'extra_forward_calls': 1}
    digest = hashlib.sha256(json.dumps(list(observations), sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf8')).hexdigest()
    return {'scope': 'feedback_surrogate_proposal_only', 'status': 'proposal_ready' if best is not None else 'no_valid_candidate',
            'selection_rule': 'maximum_corrected_soft_objective_among_valid_hard_scenarios',
            'offsets': list(offsets), 'observation_sha256': digest, 'history': history,
            'hard_screening': screening, 'selected_index': best, 'prefit_prediction': prediction,
            'corrections': [{'center': item.center.tolist(), 'radius': item.radius,
                'coefficients': item.coefficients.tolist(), 'sample_count': item.sample_count,
                'comparison_contract_sha256': item.comparison_contract_sha256,
                'surrogate_contract_sha256': item.surrogate_contract_sha256} for item in corrections]}
