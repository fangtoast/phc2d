"""Generate locally corrected Fourier proposals from physical observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.inverse.differentiable_objective import (optimize_corrected_fourier_adam,
                                                     optimize_corrected_fourier_black_box)
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
    surrogate_contract = config['surrogate_contract_sha256']
    physical_contract = config['comparison_contract_sha256']
    if (predictor.checkpoint_sha256 != config['checkpoint_sha256']
            or not surrogate_contract or not physical_contract
            or any(row.get('surrogate_contract_sha256') != surrogate_contract
                   or row.get('comparison_contract_sha256') != physical_contract for row in observations)):
        raise ValueError('observation and configured feedback contracts differ')
    correction = fit_composite_edge_correction(observations, center=config['center'],
        radius=config['radius'], ridge=config['ridge'])
    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    shared = dict(
        coefficient_keys=keys, scales=config['scales'], lower=config['lower'], upper=config['upper'],
        master_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        temperature=config['temperature'], lower_band=config['lower_band'],
        target_frequency=config['target_frequency'], beta=config['beta'])
    method = config.get('method', 'adam')
    if method == 'adam':
        history = optimize_corrected_fourier_adam(source, predictor, correction, **shared,
            learning_rate=config['learning_rate'], steps=config['steps'])
    elif method in ('de', 'random'):
        history = optimize_corrected_fourier_black_box(source, predictor, correction, **shared,
            evaluations=config['steps'] + 1, seed=config['seed'], method=method)
    else:
        raise ValueError('unknown corrected proposal method')
    screening = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
        max_mode_radius=config['max_mode_radius'], raster_method='master_block_mean_v1',
        source_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        material_fraction_bounds=tuple(config['material_fraction_bounds']),
        lower_band=config['lower_band'], target_frequency=config['target_frequency'])
    eligible = [row['index'] for row in screening['candidates'] if row['status'] == 'surrogate_only']
    best = max(eligible, key=lambda i: (history[i]['objective'], -i)) if eligible else None
    observations_raw = json.dumps(list(observations), sort_keys=True, separators=(',', ':'),
                                  allow_nan=False).encode('utf8')
    return {'status': 'proposal_ready' if best is not None else 'no_valid_candidate', 'method': method,
            'evidence_scope': 'feedback_surrogate_proposal_only',
            'observation_sha256': hashlib.sha256(observations_raw).hexdigest(),
            'correction': {'center': correction.center.tolist(), 'radius': correction.radius,
                'coefficients': correction.coefficients.tolist(), 'sample_count': correction.sample_count,
                'comparison_contract_sha256': correction.comparison_contract_sha256,
                'surrogate_contract_sha256': correction.surrogate_contract_sha256},
            'history': history, 'hard_screening': screening, 'selected_index': best}


def main() -> None:
    """Run one frozen feedback version from an explicitly configured observation batch."""
    from phc2d.data.full_cell_spec import FullCellStructureSpec
    from phc2d.inverse.audit_refine import read_checked_json
    from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
    from phc2d.inverse.differentiable_runtime import configure_design_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'checkpoint', 'observations', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config_raw = Path(args.config).read_bytes()
    config = json.loads(config_raw)
    runtime_info = configure_design_runtime(config['runtime'], device=args.device)
    source_raw, _ = read_checked_json(args.source, config['source_sha256'])
    source = FullCellStructureSpec.from_mapping(source_raw)
    predictor = FrozenBandAdapter(args.checkpoint, sha256=config['checkpoint_sha256'], device=args.device)
    if (predictor.model_name != 'global_local_band_operator_v1'
            or predictor.k_points_sha256 != config['k_points_sha256']
            or predictor.input_shape != (config['input_resolution'],)*2):
        raise ValueError('feedback checkpoint contract mismatch')
    observation_payload, observation_digest = read_checked_json(args.observations, config['observations_sha256'])
    observations = observation_payload['observations']
    result = propose_corrected_design(source, predictor, observations, config=config)
    result.update(config_sha256=hashlib.sha256(config_raw).hexdigest(),
                  observation_file_sha256=observation_digest,
                  source_sha256=config['source_sha256'], checkpoint_sha256=predictor.checkpoint_sha256,
                  runtime=runtime_info, code_commit=os.environ.get('PHC2D_DEPLOYED_GIT_COMMIT'), device=args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
