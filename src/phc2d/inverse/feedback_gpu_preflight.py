"""One frozen-model robust proposal for cross-server feedback preflight.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.feedback_handoff import build_robust_solver_manifest, validate_robust_observation
from phc2d.inverse.feedback_journal import create_robust_round
from phc2d.inverse.feedback_state import propose_feedback, start_feedback
from phc2d.inverse.fourier_feedback_proposal import propose_fourier_candidate
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.local_edge_correction import fit_local_edge_correction
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def run(*, manifest_path: str | Path, manifest_sha256: str, protocol_path: str | Path,
        protocol_sha256: str, anchor_path: str | Path, anchor_sha256: str, case_index: int,
        design_path: str | Path, design_sha256: str, checkpoint: str | Path,
        checkpoint_sha256: str, device: str, output_root: str | Path) -> Path:
    """Create one physics-calibrated candidate, with all input identities frozen.

    This single-anchor diagnostic is not a formal optimization campaign. Its
    caller supplies an already verified checkpoint/dataset raster contract.
    """
    manifest, manifest_digest = read_checked_json(manifest_path, manifest_sha256)
    protocol, protocol_digest = read_checked_json(protocol_path, protocol_sha256)
    anchor, anchor_digest = read_checked_json(anchor_path, anchor_sha256)
    design, design_digest = read_checked_json(design_path, design_sha256)
    if not design.get('expected_dataset_sha256') or manifest.get('dataset_sha256') != design['expected_dataset_sha256']:
        raise ValueError('dataset contract mismatch')
    observation = validate_robust_observation(manifest_bytes=Path(manifest_path).read_bytes(),
        expected_manifest_sha256=manifest_sha256, protocol_bytes=Path(protocol_path).read_bytes(),
        protocol_sha256=protocol_sha256, case_index=case_index, result=anchor)
    source_row = manifest['rows'][case_index]
    structure = FullCellStructureSpec.from_mapping(source_row['structure'])
    if source_row['label'] != 'anchor' or not all(observation['row'][key] is True for key in
            ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')):
        raise ValueError('a numerically stable anchor is required')
    if int(design['source_resolution']) != int(protocol['raster_resolution']):
        raise ValueError('proxy and physical master raster resolutions differ')
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    points = np.ascontiguousarray(manifest['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
        raise ValueError('checkpoint k-point identity mismatch')
    if predictor.input_shape != (int(design['input_resolution']),) * 2:
        raise ValueError('checkpoint input shape mismatch')
    keys = tuple((int(item[0]), str(item[1])) for item in design['coefficient_keys'])
    if (len(set(keys)) != len(keys) or any(index < 0 or index >= len(structure.parameters['modes'])
                                          or component not in ('cos', 'sin') for index, component in keys)):
        raise ValueError('design coefficient keys must be unique and valid')
    center = np.asarray([float(structure.parameters['modes'][index][component]) for index, component in keys]
                        + [float(structure.parameters['threshold'])], dtype=np.float64)
    dimension = len(center)
    scales = np.asarray(design['scales'], dtype=np.float64)
    if scales.shape != (dimension,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('one positive scale per design coordinate is required')
    radius = float(design['radius'])
    state = start_feedback(incumbent_hash=structure.parameter_hash, incumbent_parameters=[0.] * dimension,
        incumbent_observation=observation, target_frequency=manifest['target_frequency'], radius=radius,
        min_radius=float(design['min_radius']), max_radius=float(design['max_radius']), candidate_budget=1)
    offsets = tuple(protocol['offsets'])
    grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
    predicted = predict_perturbed_gap(grids, predictor, input_resolution=design['input_resolution'],
        lower_band=manifest['lower_band'], target_frequency=manifest['target_frequency'])
    observed_grids = [row['provenance']['evaluated_grid_sha256'] for row in anchor['observations']]
    if predicted['status'] != 'surrogate_only' or predicted['grid_sha256'] != observed_grids:
        raise ValueError('anchor surrogate prediction or perturbation grid identity mismatch')
    physical_row = observation['row']
    correction = fit_local_edge_correction([{'parameter_hash':structure.parameter_hash, 'parameters':[0.] * dimension,
        'predicted_edges':[predicted['gap']['lower_edge'],predicted['gap']['upper_edge']],
        'physical_edges':[physical_row['refined_lower_edge_frequency'],physical_row['refined_upper_edge_frequency']],
        'numerical_gate_passed':True, 'comparison_contract_sha256':observation['provenance']['comparison_contract_sha256'],
        'surrogate_contract_sha256':checkpoint_sha256}], center=[0.] * dimension, radius=radius, ridge=float(design['ridge']))
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        proposal = propose_fourier_candidate(state, structure, predictor, coefficient_keys=keys,
            global_bounds=tuple((float(c-s),float(c+s)) for c,s in zip(center,scales)), origin=center, scales=scales,
            max_mode_radius=float(design['max_mode_radius']), seed=int(design['seed']),
            population_size=int(design['population_size']), surrogate_budget=int(design['surrogate_budget']),
            raster_method='master_block_mean_v1', source_resolution=int(design['source_resolution']),
            input_resolution=int(design['input_resolution']), material_fraction_bounds=tuple(design['material_fraction_bounds']),
            lower_band=manifest['lower_band'], surrogate_contract_sha256=checkpoint_sha256,
            perturbation_offsets=offsets, geometry_constraints=design.get('geometry_constraints'),
            edge_correction=correction)
        payload: dict[str, Any] = {'status':proposal['status'], 'scope':'exploratory_single_anchor_engineering_preflight',
            'manifest_sha256':manifest_digest, 'protocol_sha256':protocol_digest, 'anchor_sha256':anchor_digest,
            'design_sha256':design_digest, 'checkpoint_sha256':checkpoint_sha256, 'case_index':case_index,
            'source_index':source_row['source_index'], 'anchor_predicted_gap':predicted['gap'],
            'anchor_physical_edges':[physical_row['refined_lower_edge_frequency'],physical_row['refined_upper_edge_frequency']],
            'correction_coefficients':correction.coefficients.tolist(), 'proposal':proposal,
            'search_elapsed_s':time.monotonic()-started}
        if proposal['proposal'] is not None:
            pending = propose_feedback(state, **proposal['proposal'])
            candidate = FullCellStructureSpec.from_mapping(proposal['structure'])
            solver_manifest = build_robust_solver_manifest(pending, candidate, k_points=manifest['k_points'],
                config=FullCellMpbBandSolverConfig(**manifest['solver_config']),
                lower_band=manifest['lower_band'], source_index=source_row['source_index'])
            manifest_raw = json.dumps(solver_manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
            (output/'solver_manifest.json').write_bytes(manifest_raw)
            (output/'pending_state.json').write_text(json.dumps(asdict(pending),indent=2,allow_nan=False),encoding='utf8')
            protocol_raw = Path(protocol_path).read_bytes()
            if hashlib.sha256(protocol_raw).hexdigest() != protocol_digest:
                raise ValueError('protocol changed during proposal')
            create_robust_round(output/'journal', pending=pending, manifest_bytes=manifest_raw,
                                protocol_bytes=protocol_raw)
            payload['solver_manifest_sha256'] = hashlib.sha256(manifest_raw).hexdigest()
        (output/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__, 'message':str(exc),
            'elapsed_s':time.monotonic()-started},indent=2),encoding='utf8')
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare one physics-calibrated robust proposal')
    for name in ('manifest-path','manifest-sha256','protocol-path','protocol-sha256','anchor-path','anchor-sha256',
                 'design-path','design-sha256','checkpoint','checkpoint-sha256','device','output-root'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--case-index', required=True, type=int)
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == '__main__':
    main()
