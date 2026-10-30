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
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids
from phc2d.inverse.refinement_feedback import aggregate_refined_perturbations
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


def _validate_spectral_trace(row: Mapping[str, Any], arrays: Mapping[str, Any],
                             history: Any, provenance: Mapping[str, Any],
                             manifest: Mapping[str, Any], refinement: Mapping[str, Any]) -> None:
    """Recompute band edges, numerical gates and solver identity."""
    if row['lower_band_one_based'] != manifest['lower_band']:
        raise ValueError('solver band-pair mismatch')
    initial_points = np.asarray(manifest['k_points'], dtype=np.float64)
    refined_points = np.asarray(arrays['k_points'], dtype=np.float64)
    if (refined_points.ndim != 2 or len(refined_points) < len(initial_points)
            or not np.array_equal(refined_points[:len(initial_points)], initial_points)):
        raise ValueError('initial k-point identity mismatch')
    bands = np.asarray(arrays['bands'], dtype=np.float64)
    verify_points = np.asarray(arrays['verification_k_points'], dtype=np.float64)
    verify_bands = np.asarray(arrays['verification_bands'], dtype=np.float64)
    band_count = manifest['solver_config']['num_bands']
    if (bands.shape != (len(refined_points), band_count) or verify_points.ndim != 2
            or verify_points.shape[1] != 2 or verify_bands.shape != (len(verify_points), band_count)
            or not np.isfinite(refined_points).all() or not np.isfinite(verify_points).all()
            or not np.isfinite(bands).all() or not np.isfinite(verify_bands).all()
            or np.any(bands < 0) or np.any(verify_bands < 0)
            or np.any(np.diff(bands, axis=1) < 0) or np.any(np.diff(verify_bands, axis=1) < 0)):
        raise ValueError('solver spectral arrays invalid')
    band = manifest['lower_band']
    gap = adjacent_gap_records(bands, lower_band_min_one_based=band, lower_band_max_one_based=band)[0]
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
    if not history or len(history) > int(refinement['max_levels']):
        raise ValueError('refinement history missing')
    def gap_at(count: int) -> Mapping[str, Any]:
        return adjacent_gap_records(bands[:count], lower_band_min_one_based=band,
                                    lower_band_max_one_based=band)[0]
    previous = gap_at(len(initial_points))
    count, streak, edge_change = len(initial_points), 0, float('inf')
    for level, item in enumerate(history, start=1):
        new_points = item.get('new_k_points')
        if (not isinstance(new_points, int) or isinstance(new_points, bool) or new_points < 0
                or item.get('stage') != f'refinement_{level}' or item.get('total_k_points') != count + new_points
                or count + new_points > len(refined_points)
                or not np.isclose(float(item['step']), float(refinement['initial_spacing']) / (2 ** level), rtol=0, atol=1e-12)):
            raise ValueError('refinement history structure mismatch')
        count += new_points
        current = gap_at(count)
        edge_change = max(abs(float(current[name]) - float(previous[name]))
                          for name in ('lower_edge_frequency', 'upper_edge_frequency'))
        streak = streak + 1 if edge_change <= float(refinement['edge_tolerance']) else 0
        if (not np.isclose(float(item['max_edge_frequency_change']), edge_change, rtol=0, atol=1e-10)
                or item['stable_streak'] != streak):
            raise ValueError('refinement history edge or stability mismatch')
        if (level < len(history) and level >= int(refinement['min_levels'])
                and streak >= int(refinement['required_stable_levels'])):
            raise ValueError('refinement continued after convergence')
        previous = current
    if count != len(refined_points):
        raise ValueError('refinement history does not cover all k points')
    expected_edge_stable = edge_change <= float(refinement['edge_tolerance'])
    expected_converged = len(history) >= int(refinement['min_levels']) and streak >= int(refinement['required_stable_levels'])
    if (row.get('resolution_stable') is not expected_resolution_stable
            or row.get('last_refinement_stable') is not expected_edge_stable
            or row.get('adaptive_refinement_converged') is not expected_converged
            or provenance.get('verification_resolution') != refinement['verification_resolution']):
        raise ValueError('solver numerical gate mismatch')
    resolved_solver_config = asdict(FullCellMpbBandSolverConfig(**manifest['solver_config']))
    for key in ('resolution', 'polarization', 'num_bands', 'mesh_size', 'band_guard_count',
                'band_order_policy', 'eigensolver_tolerance', 'epsilon_air', 'epsilon_material'):
        expected = resolved_solver_config.get(key)
        if provenance['initial_solver'].get(key) != expected:
            raise ValueError(f'solver {key} mismatch')
        verification_expected = refinement['verification_resolution'] if key == 'resolution' else expected
        if provenance['verification_solver'].get(key) != verification_expected:
            raise ValueError(f'verification solver {key} mismatch')
    for key in ('solver_backend', 'solver_version'):
        if (not provenance['initial_solver'].get(key)
                or provenance['initial_solver'][key] != provenance['verification_solver'].get(key)):
            raise ValueError(f'solver {key} identity mismatch')


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
    if len(result['rows']) != 1:
        raise ValueError('solver band-pair mismatch')
    _validate_spectral_trace(result['rows'][0], result['arrays'], result['history'],
                             provenance, manifest, refinement)
    incumbent_solver = state.incumbent_observation['provenance'].get('initial_solver', {})
    for key in ('solver_backend', 'solver_version'):
        if not incumbent_solver.get(key) or incumbent_solver[key] != provenance['initial_solver'].get(key):
            raise ValueError('incumbent and candidate solver identity mismatch')
    observation = {'row': result['rows'][0], 'provenance': provenance}
    return finish_feedback(state, observation=observation)


def build_robust_solver_manifest(state: FeedbackState, structure: FullCellStructureSpec, *,
                                 k_points: Any, config: FullCellMpbBandSolverConfig,
                                 lower_band: int, source_index: int) -> dict[str, Any]:
    """Create a candidate contract for the offset solver runner."""
    manifest = build_nominal_solver_manifest(state, structure, k_points=k_points, config=config,
                                             lower_band=lower_band, source_index=source_index)
    manifest['status'] = 'feedback_robust_solver_request'
    return manifest


def validate_robust_observation(*, manifest_bytes: bytes, expected_manifest_sha256: str,
                                protocol_bytes: bytes, protocol_sha256: str,
                                case_index: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute one immutable offset observation, including solver and gate identity."""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_manifest_sha256 or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256:
        raise ValueError('manifest or protocol digest mismatch')
    manifest, protocol = json.loads(manifest_bytes), json.loads(protocol_bytes)
    rows = manifest['rows']
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 0 <= case_index < len(rows):
        raise ValueError('case index out of bounds')
    request = rows[case_index]
    structure = FullCellStructureSpec.from_mapping(request['structure'])
    if structure.parameter_hash != request['parameter_hash']:
        raise ValueError('candidate parameter identity mismatch')
    if (result.get('manifest_sha256') != digest or result.get('protocol_sha256') != protocol_sha256
            or result.get('case_index') != case_index or result.get('source_index') != request['source_index']
            or result.get('label') != request['label'] or result.get('parameter_hash') != structure.parameter_hash):
        raise ValueError('solver result identity mismatch')
    offsets = tuple(protocol['offsets'])
    grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
    observations = result['observations']
    if len(observations) != len(offsets) or [o['offset'] for o in observations] != list(offsets):
        raise ValueError('solver offset identity mismatch')
    for grid, trace in zip(grids, observations):
        provenance = trace['provenance']
        grid_hash = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
        if (provenance.get('parameter_hash') != structure.parameter_hash
                or provenance.get('evaluated_geometry') != 'periodic_binary_pixel_cells_v1'
                or provenance.get('evaluated_grid_sha256') != grid_hash
                or provenance.get('initial_solver', {}).get('grid_sha256') != grid_hash
                or provenance.get('verification_solver', {}).get('grid_sha256') != grid_hash):
            raise ValueError('solver offset grid identity mismatch')
        _validate_spectral_trace(trace['row'], trace['arrays'], trace['history'], provenance,
                                 manifest, protocol['refinement'])
    aggregate = aggregate_refined_perturbations(observations, expected_offsets=offsets)
    if aggregate != result['aggregate']:
        raise ValueError('solver common-gap aggregate mismatch')
    identities = [{'solver_backend': o['provenance']['initial_solver']['solver_backend'],
                   'solver_version': o['provenance']['initial_solver']['solver_version']} for o in observations]
    if any(identity != identities[0] for identity in identities):
        raise ValueError('offset solver identities differ')
    return {'row': aggregate['row'], 'provenance': {**aggregate['provenance'], 'solver_identity': identities[0]}}


def consume_robust_solver_result(state: FeedbackState, *, manifest_bytes: bytes,
                                 expected_manifest_sha256: str, protocol_bytes: bytes,
                                 protocol_sha256: str, result: Mapping[str, Any]) -> FeedbackState:
    """Verify every offset raster and numerical gate before applying feedback."""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_manifest_sha256 or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256:
        raise ValueError('manifest or protocol digest mismatch')
    manifest, protocol = json.loads(manifest_bytes), json.loads(protocol_bytes)
    if state.pending is None or manifest.get('status') != 'feedback_robust_solver_request' or len(manifest['rows']) != 1:
        raise ValueError('invalid pending robust solver request')
    request = manifest['rows'][0]
    structure = FullCellStructureSpec.from_mapping(request['structure'])
    if (structure.parameter_hash != state.pending.parameter_hash or request['parameter_hash'] != structure.parameter_hash
            or manifest['surrogate_contract_sha256'] != state.pending.surrogate_contract_sha256
            or manifest['target_frequency'] != state.target_frequency):
        raise ValueError('pending candidate or prediction contract mismatch')
    observation = validate_robust_observation(manifest_bytes=manifest_bytes,
        expected_manifest_sha256=expected_manifest_sha256, protocol_bytes=protocol_bytes,
        protocol_sha256=protocol_sha256, case_index=0, result=result)
    if state.incumbent_observation['provenance'].get('solver_identity') != observation['provenance']['solver_identity']:
        raise ValueError('incumbent and candidate solver identity mismatch')
    return finish_feedback(state, observation=observation)
