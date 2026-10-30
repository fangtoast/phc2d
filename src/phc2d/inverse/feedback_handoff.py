"""Identity-checked handoff from feedback proposals to solver results.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.feedback_state import FeedbackState, finish_feedback
from phc2d.physics.bz_gap_refinement import adjacent_gap_records
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def build_nominal_solver_manifest(state: FeedbackState, structure: FullCellStructureSpec, *,
                                  k_points: Any, config: FullCellMpbBandSolverConfig,
                                  lower_band: int, source_index: int) -> dict[str, Any]:
    """Create the immutable one-case contract consumed by the audit solver runner."""
    if state.pending is None or structure.parameter_hash != state.pending.parameter_hash:
        raise ValueError('structure does not match pending candidate')
    points = np.asarray(k_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('k points must be a finite nonempty coordinate matrix')
    if (not isinstance(lower_band, int) or isinstance(lower_band, bool)
            or not 1 <= lower_band < config.num_bands or not isinstance(source_index, int)
            or isinstance(source_index, bool) or source_index < 0):
        raise ValueError('invalid band pair or source identity')
    return {'status': 'feedback_solver_request', 'k_points': points.tolist(), 'lower_band': lower_band,
            'target_frequency': state.target_frequency, 'solver_config': asdict(config),
            'surrogate_contract_sha256': state.pending.surrogate_contract_sha256,
            'rows': [{'source_index': source_index, 'label': 'candidate',
                      'parameter_hash': structure.parameter_hash, 'structure': structure.to_mapping()}]}


def consume_nominal_solver_result(state: FeedbackState, *, manifest_bytes: bytes,
                                  expected_manifest_sha256: str, refinement_sha256: str,
                                  refinement_bytes: bytes, result: Mapping[str, Any]) -> FeedbackState:
    """Accept one solver record only after matching request, geometry and fidelity."""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError('solver manifest digest mismatch')
    if hashlib.sha256(refinement_bytes).hexdigest() != refinement_sha256:
        raise ValueError('refinement digest mismatch')
    manifest = json.loads(manifest_bytes)
    refinement = json.loads(refinement_bytes)
    if state.pending is None or manifest.get('status') != 'feedback_solver_request' or len(manifest['rows']) != 1:
        raise ValueError('invalid pending solver request')
    request = manifest['rows'][0]
    structure = FullCellStructureSpec.from_mapping(request['structure'])
    if (structure.parameter_hash != state.pending.parameter_hash or request['parameter_hash'] != structure.parameter_hash
            or manifest['surrogate_contract_sha256'] != state.pending.surrogate_contract_sha256
            or manifest['target_frequency'] != state.target_frequency):
        raise ValueError('pending candidate or prediction contract mismatch')
    if (result.get('manifest_sha256') != digest or result.get('refinement_sha256') != refinement_sha256
            or result.get('case_index') != 0 or result.get('source_index') != request['source_index']
            or result.get('label') != 'candidate' or result.get('parameter_hash') != structure.parameter_hash):
        raise ValueError('solver result identity mismatch')
    provenance = result['provenance']
    if (provenance.get('parameter_hash') != structure.parameter_hash
            or provenance.get('evaluated_geometry') != 'analytic'
            or provenance.get('initial_solver', {}).get('geometry_hash') != structure.geometry_hash
            or provenance.get('verification_solver', {}).get('geometry_hash') != structure.geometry_hash):
        raise ValueError('solver geometry identity mismatch')
    if len(result['rows']) != 1 or result['rows'][0]['lower_band_one_based'] != manifest['lower_band']:
        raise ValueError('solver band-pair mismatch')
    initial_points = np.asarray(manifest['k_points'], dtype=np.float64)
    refined_points = np.asarray(result['arrays']['k_points'], dtype=np.float64)
    if (refined_points.ndim != 2 or len(refined_points) < len(initial_points)
            or not np.array_equal(refined_points[:len(initial_points)], initial_points)):
        raise ValueError('initial k-point identity mismatch')
    bands = np.asarray(result['arrays']['bands'], dtype=np.float64)
    verify_points = np.asarray(result['arrays']['verification_k_points'], dtype=np.float64)
    verify_bands = np.asarray(result['arrays']['verification_bands'], dtype=np.float64)
    band_count = manifest['solver_config']['num_bands']
    if (bands.shape != (len(refined_points), band_count) or verify_points.ndim != 2
            or verify_points.shape[1] != 2 or verify_bands.shape != (len(verify_points), band_count)
            or not np.isfinite(bands).all() or not np.isfinite(verify_bands).all()):
        raise ValueError('solver spectral arrays invalid')
    band = manifest['lower_band']
    gap = adjacent_gap_records(bands, lower_band_min_one_based=band, lower_band_max_one_based=band)[0]
    row = result['rows'][0]
    for name in ('lower_edge_frequency', 'upper_edge_frequency'):
        if not np.isclose(float(row['refined_' + name]), float(gap[name]), rtol=0, atol=1e-10):
            raise ValueError('solver edge does not match spectral arrays')
    verification_lookup = {tuple(point): index for index, point in enumerate(verify_points)}
    if len(verification_lookup) != len(verify_points):
        raise ValueError('duplicate verification k points')
    changes = []
    for index_key, band_index in (('lower_edge_k_index', band-1), ('upper_edge_k_index', band)):
        index = int(gap[index_key])
        location = verification_lookup.get(tuple(refined_points[index]))
        if location is None:
            raise ValueError('verification edge k point missing')
        changes.append(abs(float(bands[index, band_index]) - float(verify_bands[location, band_index])))
    for name, value in zip(('lower_resolution_change', 'upper_resolution_change'), changes):
        if not np.isclose(float(row[name]), value, rtol=0, atol=1e-10):
            raise ValueError('solver resolution change mismatch')
    expected_resolution_stable = max(changes) <= float(refinement['resolution_tolerance'])
    history = result['history']
    if not history:
        raise ValueError('refinement history missing')
    expected_edge_stable = float(history[-1]['max_edge_frequency_change']) <= float(refinement['edge_tolerance'])
    expected_converged = (len(history) >= int(refinement['min_levels'])
                          and int(history[-1]['stable_streak']) >= int(refinement['required_stable_levels']))
    if (row.get('resolution_stable') is not expected_resolution_stable
            or row.get('last_refinement_stable') is not expected_edge_stable
            or row.get('adaptive_refinement_converged') is not expected_converged
            or provenance.get('verification_resolution') != refinement['verification_resolution']):
        raise ValueError('solver numerical gate mismatch')
    for key in ('resolution', 'polarization', 'num_bands', 'mesh_size', 'band_guard_count',
                'band_order_policy', 'eigensolver_tolerance', 'epsilon_air', 'epsilon_material'):
        expected = manifest['solver_config'].get(key)
        if provenance['initial_solver'].get(key) != expected:
            raise ValueError(f'solver {key} mismatch')
        verification_expected = refinement['verification_resolution'] if key == 'resolution' else expected
        if provenance['verification_solver'].get(key) != verification_expected:
            raise ValueError(f'verification solver {key} mismatch')
    for key in ('solver_backend', 'solver_version'):
        if (not provenance['initial_solver'].get(key)
                or provenance['initial_solver'][key] != provenance['verification_solver'].get(key)):
            raise ValueError(f'solver {key} identity mismatch')
    observation = {'row': result['rows'][0], 'provenance': provenance}
    return finish_feedback(state, observation=observation)
