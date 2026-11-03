"""Checks for immutable cross-server feedback solver handoff.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_refine import run_case
from phc2d.inverse.audit_perturb_refine import run_case as run_offset_case
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_handoff import (build_nominal_solver_manifest, consume_nominal_solver_result,
                                            build_robust_solver_manifest, consume_robust_solver_result,
                                            validate_robust_observation,
                                            validate_nominal_observation_from_robust)
from phc2d.inverse.feedback_state import propose_feedback, start_feedback
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def test_real_runner_output_can_be_reconciled_into_feedback_state(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    points = np.array([[0., 0.], [.25, 0.]])
    def solver(item, *, k_points, config):
        delta = item.parameters['threshold'] - source.parameters['threshold']
        metadata = {'geometry_hash': item.geometry_hash, 'resolution': config.resolution,
                    'polarization': config.polarization, 'num_bands': config.num_bands,
                    'mesh_size': config.mesh_size, 'band_guard_count': config.band_guard_count,
                    'band_order_policy': config.band_order_policy, 'eigensolver_tolerance': config.eigensolver_tolerance,
                    'epsilon_air': config.epsilon_air, 'epsilon_material': config.epsilon_material,
                    'solver_backend': 'mock', 'solver_version': '1'}
        return np.tile([.2 + delta, .7 - delta], (len(k_points), 1)), metadata
    anchor = evaluate_feedback_candidate(source, k_points=points, config=config, lower_band=1,
                                          refinement=options, solver=solver)['observation']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[0.],
        incumbent_observation=anchor, target_frequency=.5, radius=.1, min_radius=.01,
        max_radius=.2, candidate_budget=1)
    candidate = update_fourier_binary(source, sample_id='candidate', coefficients={},
                                     threshold=source.parameters['threshold']-.01, max_mode_radius=1)
    pending = propose_feedback(state, parameter_hash=candidate.parameter_hash, parameters=[-.01],
                               predicted_center=.8, predicted_candidate=.84, surrogate_contract_sha256='model')
    manifest = build_nominal_solver_manifest(pending, candidate, k_points=points, config=config,
                                             lower_band=1, source_index=3)
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest), encoding='utf8')
    op = tmp_path / 'refinement.json'
    op.write_text(json.dumps(options), encoding='utf8')
    mh, oh = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (mp, op))
    output = run_case(mp, expected_sha256=mh, refinement_path=op, expected_refinement_sha256=oh,
                      case_index=0, output_root=tmp_path/'cases', solver=solver)
    result = json.loads((output/'results.json').read_text())
    final = consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
        expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)
    assert final.incumbent_hash == candidate.parameter_hash
    assert final.history[-1]['accepted']
    result['arrays']['k_points'][0][0] = .125
    with pytest.raises(ValueError, match='k-point identity'):
        consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
            expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)
    result['arrays']['k_points'][0][0] = 0.
    result['rows'][0]['resolution_stable'] = False
    with pytest.raises(ValueError, match='numerical gate'):
        consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
            expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)
    result['rows'][0]['resolution_stable'] = True
    result['provenance']['initial_solver']['solver_version'] = '2'
    result['provenance']['verification_solver']['solver_version'] = '2'
    with pytest.raises(ValueError, match='incumbent and candidate solver identity'):
        consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
            expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)
    result = json.loads((output/'results.json').read_text())
    result['history'][-1]['max_edge_frequency_change'] += .1
    with pytest.raises(ValueError, match='history edge'):
        consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
            expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)
    result = json.loads((output/'results.json').read_text())
    result['arrays']['bands'][0][0] = float('nan')
    with pytest.raises(ValueError, match='spectral arrays'):
        consume_nominal_solver_result(pending, manifest_bytes=mp.read_bytes(),
            expected_manifest_sha256=mh, refinement_sha256=oh, refinement_bytes=op.read_bytes(), result=result)


def test_robust_handoff_recomputes_offset_grids_and_common_gap(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    points = np.array([[0., 0.], [.25, 0.]])
    offsets = (-.0625, 0., .0625)
    def solver(grid, *, k_points, config):
        fill = float(grid.mean())
        metadata = {'grid_sha256': hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest(),
                    'resolution': config.resolution, 'polarization': config.polarization,
                    'num_bands': config.num_bands, 'mesh_size': config.mesh_size,
                    'band_guard_count': config.band_guard_count, 'band_order_policy': config.band_order_policy,
                    'eigensolver_tolerance': config.eigensolver_tolerance, 'epsilon_air': config.epsilon_air,
                    'epsilon_material': config.epsilon_material, 'solver_backend': 'mock', 'solver_version': '1'}
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), metadata
    anchor = evaluate_feedback_candidate(source, k_points=points, config=config, lower_band=1,
        refinement=options, offsets=offsets, raster_resolution=16, solver=solver)['observation']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[0.],
        incumbent_observation=anchor, target_frequency=.5, radius=.2, min_radius=.01,
        max_radius=.4, candidate_budget=1)
    candidate = update_fourier_binary(source, sample_id='candidate', coefficients={},
                                     threshold=source.parameters['threshold']+.1, max_mode_radius=1)
    pending = propose_feedback(state, parameter_hash=candidate.parameter_hash, parameters=[.1],
                               predicted_center=.4, predicted_candidate=.5, surrogate_contract_sha256='model')
    manifest = build_robust_solver_manifest(pending, candidate, k_points=points, config=config,
                                            lower_band=1, source_index=3)
    # Historical manifests can omit the solver's default band-order policy.
    del manifest['solver_config']['band_order_policy']
    protocol = {'offsets': offsets, 'raster_resolution': 16, 'refinement': options}
    mp, pp = tmp_path/'manifest.json', tmp_path/'protocol.json'
    mp.write_text(json.dumps(manifest), encoding='utf8')
    pp.write_text(json.dumps(protocol), encoding='utf8')
    mh, ph = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (mp, pp))
    output = run_offset_case(mp, expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                             case_index=0, output_root=tmp_path/'cases', solver=solver)
    result = json.loads((output/'results.json').read_text())
    validated = validate_robust_observation(manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
        protocol_bytes=pp.read_bytes(), protocol_sha256=ph, case_index=0, result=result)
    assert validated['provenance']['parameter_hash'] == candidate.parameter_hash
    assert validated['provenance']['solver_identity']['solver_backend'] == 'mock'
    nominal = validate_nominal_observation_from_robust(
        manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
        protocol_bytes=pp.read_bytes(), protocol_sha256=ph, case_index=0, result=result)
    assert nominal['provenance']['offsets'] == [0.0]
    assert nominal['provenance']['comparison_contract_sha256'] != validated['provenance']['comparison_contract_sha256']
    assert nominal['row']['refined_lower_edge_frequency'] == result['observations'][1]['row']['refined_lower_edge_frequency']
    assert nominal['row']['refined_upper_edge_frequency'] == result['observations'][1]['row']['refined_upper_edge_frequency']
    nominal_protocol = tmp_path / 'nominal_protocol.json'
    nominal_protocol.write_text(json.dumps({**protocol, 'offsets': [0.0]}), encoding='utf8')
    nominal_ph = hashlib.sha256(nominal_protocol.read_bytes()).hexdigest()
    nominal_output = run_offset_case(mp, expected_sha256=mh, protocol_path=nominal_protocol,
        expected_protocol_sha256=nominal_ph, case_index=0, output_root=tmp_path/'nominal_cases', solver=solver)
    single = validate_robust_observation(manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
        protocol_bytes=nominal_protocol.read_bytes(), protocol_sha256=nominal_ph, case_index=0,
        result=json.loads((nominal_output/'results.json').read_text()))
    assert nominal == single
    final = consume_robust_solver_result(pending, manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
                                         protocol_bytes=pp.read_bytes(), protocol_sha256=ph, result=result)
    assert final.incumbent_hash == candidate.parameter_hash
    result['observations'][0]['provenance']['evaluated_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='grid identity'):
        validate_nominal_observation_from_robust(manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
            protocol_bytes=pp.read_bytes(), protocol_sha256=ph, case_index=0, result=result)
    with pytest.raises(ValueError, match='grid identity'):
        consume_robust_solver_result(pending, manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
                                     protocol_bytes=pp.read_bytes(), protocol_sha256=ph, result=result)
    result = json.loads((output/'results.json').read_text())
    for trace in result['observations']:
        trace['provenance']['initial_solver']['solver_version'] = '2'
        trace['provenance']['verification_solver']['solver_version'] = '2'
    with pytest.raises(ValueError, match='incumbent and candidate solver identity'):
        consume_robust_solver_result(pending, manifest_bytes=mp.read_bytes(), expected_manifest_sha256=mh,
                                     protocol_bytes=pp.read_bytes(), protocol_sha256=ph, result=result)
