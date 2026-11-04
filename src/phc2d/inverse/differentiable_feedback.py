"""Generate locally corrected Fourier proposals from physical observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from phc2d.inverse.differentiable_objective import optimize_corrected_fourier_adam
from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.local_edge_correction import fit_composite_edge_correction


def propose_corrected_design(source: Any, predictor: Any, observations: Sequence[Mapping[str, Any]], *,
                             config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit one frozen local correction and propose an eligible hard geometry.

    The caller provides authenticated, already-gated physical observations from
    this feedback run. This function never reads final physical test results and
    never invokes the solver. Ranking uses corrected soft objectives among hard
    geometries that pass the unchanged GL validity screen.
    """
    correction = fit_composite_edge_correction(observations, center=config['center'],
        radius=config['radius'], ridge=config['ridge'])
    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    history = optimize_corrected_fourier_adam(source, predictor, correction,
        coefficient_keys=keys, scales=config['scales'], lower=config['lower'], upper=config['upper'],
        master_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        temperature=config['temperature'], lower_band=config['lower_band'],
        target_frequency=config['target_frequency'], beta=config['beta'],
        learning_rate=config['learning_rate'], steps=config['steps'])
    screening = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
        max_mode_radius=config['max_mode_radius'], raster_method='master_block_mean_v1',
        source_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        material_fraction_bounds=tuple(config['material_fraction_bounds']),
        lower_band=config['lower_band'], target_frequency=config['target_frequency'])
    eligible = [row['index'] for row in screening['candidates'] if row['status'] == 'surrogate_only']
    best = max(eligible, key=lambda i: (history[i]['objective'], -i)) if eligible else None
    observations_raw = json.dumps(list(observations), sort_keys=True, separators=(',', ':'),
                                  allow_nan=False).encode('utf8')
    return {'status': 'proposal_ready' if best is not None else 'no_valid_candidate',
            'evidence_scope': 'feedback_surrogate_proposal_only',
            'observation_sha256': hashlib.sha256(observations_raw).hexdigest(),
            'correction': {'center': correction.center.tolist(), 'radius': correction.radius,
                'coefficients': correction.coefficients.tolist(), 'sample_count': correction.sample_count,
                'comparison_contract_sha256': correction.comparison_contract_sha256,
                'surrogate_contract_sha256': correction.surrogate_contract_sha256},
            'history': history, 'hard_screening': screening, 'selected_index': best}
