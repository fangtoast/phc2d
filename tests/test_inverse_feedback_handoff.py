"""Checks for immutable cross-server feedback solver handoff.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import base64
from dataclasses import asdict

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_refine import run_case
from phc2d.inverse.audit_perturb_refine import run_case as run_offset_case
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_batch_dispatch import (FeedbackBatchState, prepare_worker_batch,
    mark_batch_submission_unknown, confirm_worker_batch, reconcile_worker_batch, settle_worker_batch,
    release_unstarted_worker_batch, restore_feedback_batch_state)
from phc2d.inverse.feedback_handoff import (build_nominal_solver_manifest, consume_nominal_solver_result,
                                            build_robust_solver_manifest, consume_robust_solver_result,
                                            validate_robust_observation, consume_robust_worker_group,
                                            validate_nominal_observation_from_robust)
from phc2d.inverse.feedback_state import propose_feedback, start_feedback
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger, reserve_resources, settle_slurm_worker_group
from phc2d.inverse.feedback_journal import create_batch_journal, append_batch_event, verify_batch_journal
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
    other = update_fourier_binary(source, sample_id='other', coefficients={},
                                  threshold=source.parameters['threshold']-.1, max_mode_radius=1)
    other_pending = propose_feedback(state, parameter_hash=other.parameter_hash, parameters=[-.1],
        predicted_center=.4, predicted_candidate=.5, surrogate_contract_sha256='model')
    other_manifest = build_robust_solver_manifest(other_pending, other, k_points=points, config=config,
                                                  lower_band=1, source_index=3)
    other_raw = json.dumps(other_manifest).encode()
    def worker(exit_code, manifest_raw, manifest_hash, result_raw, reason):
        return {'exit_code': exit_code, 'manifest_bytes': manifest_raw, 'manifest_sha256': manifest_hash,
                'protocol_bytes': pp.read_bytes(), 'protocol_sha256': ph,
                'result_bytes': result_raw, 'failure_reason': reason}
    states = {'success': pending, 'failure': other_pending}
    records = {'success': worker(0, mp.read_bytes(), mh, (output/'results.json').read_bytes(), None),
               'failure': worker(137, other_raw, hashlib.sha256(other_raw).hexdigest(), None, 'worker OOM')}
    updated, outcomes = consume_robust_worker_group(states, records)
    shared_manifest = {**manifest, 'rows': [other_manifest['rows'][0], manifest['rows'][0]]}
    shared_path = tmp_path/'shared_manifest.json'
    shared_path.write_text(json.dumps(shared_manifest), encoding='utf8')
    shared_raw = shared_path.read_bytes()
    shared_hash = hashlib.sha256(shared_raw).hexdigest()
    shared_output = run_offset_case(shared_path, expected_sha256=shared_hash, protocol_path=pp,
        expected_protocol_sha256=ph, case_index=1, output_root=tmp_path/'shared_cases', solver=solver)
    shared_records = {'success': worker(0, shared_raw, shared_hash, (shared_output/'results.json').read_bytes(), None),
                      'failure': worker(137, shared_raw, shared_hash, None, 'worker OOM')}
    shared_updated, shared_outcomes = consume_robust_worker_group(states, shared_records)
    assert shared_updated == updated
    assert shared_outcomes['success']['candidate_sha256'] == candidate.parameter_hash
    wrong_result = json.loads(shared_records['success']['result_bytes'])
    wrong_result['case_index'] = 0
    wrong_records = {**shared_records, 'success': {**shared_records['success'],
                                                    'result_bytes': json.dumps(wrong_result).encode()}}
    with pytest.raises(ValueError, match='solver result identity mismatch'):
        consume_robust_worker_group(states, wrong_records)
    duplicate_manifest = {**shared_manifest, 'rows': shared_manifest['rows'] + [shared_manifest['rows'][1]]}
    duplicate_raw = json.dumps(duplicate_manifest).encode()
    duplicate_records = {**shared_records, 'success': {**shared_records['success'], 'manifest_bytes': duplicate_raw,
        'manifest_sha256': hashlib.sha256(duplicate_raw).hexdigest()}}
    with pytest.raises(ValueError, match='worker request does not match'):
        consume_robust_worker_group(states, duplicate_records)
    assert updated['success'].remaining_evaluations == 0
    assert updated['failure'].history[-1]['status'] == 'solver_failure'
    assert updated['failure'].remaining_evaluations == 0
    assert outcomes['success']['result_sha256'] == hashlib.sha256((output/'results.json').read_bytes()).hexdigest()
    assert outcomes['failure']['exit_code'] == 137 and outcomes['failure']['result_sha256'] is None
    ledger = reserve_resources(ResourceLedger(ResourceAmount(40, 0)), task_id='batch', amount=ResourceAmount(40, 0))
    ledger, accounting = settle_slurm_worker_group(ledger, task_id='batch', cluster_id='m9', job_id='100',
        accounting_bytes=b'100|FAILED|10|4|cpu=4,node=1\n', expected_gpus=0,
        expected_workers={name: item.pending.parameter_hash for name, item in states.items()}, worker_outcomes=outcomes)
    assert ledger.totals()['actual'] == ResourceAmount(40, 0) and accounting['failed_workers'] == ['failure']
    batch_state = FeedbackBatchState({'success': state, 'failure': state}, {},
                                      ResourceLedger(ResourceAmount(40, 0)), max_inflight=2, max_attempts=4)
    worker_lanes = {'success': 'success', 'failure': 'failure'}
    proposals = {'success': asdict(pending.pending), 'failure': asdict(other_pending.pending)}
    batch_state = prepare_worker_batch(batch_state, batch_id='batch', task_id='task', cluster_id='m9',
                                      proposals=proposals, workers=worker_lanes, reservation=ResourceAmount(40, 0))
    batch_state = mark_batch_submission_unknown(batch_state, batch_id='batch', reason='submitted, outcome uncertain')
    restored = restore_feedback_batch_state(json.loads(json.dumps(asdict(batch_state))))
    assert restored == batch_state and restored.ledger.totals()['reserved'] == ResourceAmount(40, 0)
    with pytest.raises(ValueError, match='no-future-execution'):
        release_unstarted_worker_batch(restored, batch_id='batch', confirmed_not_started=True,
            confirmed_no_future_execution=False, confirmation_source='scheduler')
    batch_state = confirm_worker_batch(restored, batch_id='batch', job_id='100', confirmation_source='scheduler')
    unknown_records = dict(records)
    unknown_records['failure'] = {**records['failure'], 'exit_code': None,
                                  'failure_reason': 'terminal job; worker exit unavailable'}
    with pytest.raises(ValueError, match='terminal accounting required'):
        reconcile_worker_batch(batch_state, batch_id='batch', results_by_worker=unknown_records)
    with pytest.raises(ValueError, match='not terminal'):
        reconcile_worker_batch(batch_state, batch_id='batch', results_by_worker=unknown_records,
            terminal_accounting_bytes=b'100|RUNNING|10|4|cpu=4,node=1\n', expected_gpus=0)
    unknown_batch = reconcile_worker_batch(batch_state, batch_id='batch', results_by_worker=unknown_records,
        terminal_accounting_bytes=b'100|FAILED|10|4|cpu=4,node=1\n', expected_gpus=0)
    assert unknown_batch.lanes['failure'].history[-1]['status'] == 'worker_result_unknown'
    unknown_batch = restore_feedback_batch_state(json.loads(json.dumps(asdict(unknown_batch))))
    with pytest.raises(ValueError, match='terminal accounting differs'):
        settle_worker_batch(unknown_batch, batch_id='batch',
            accounting_bytes=b'100|FAILED|11|4|cpu=4,node=1\n', expected_gpus=0)
    unknown_batch, unknown_audit = settle_worker_batch(unknown_batch, batch_id='batch',
        accounting_bytes=b'100|FAILED|10|4|cpu=4,node=1\n', expected_gpus=0)
    assert unknown_audit['unknown_workers'] == ['failure'] and unknown_audit['failed_workers'] == []
    assert unknown_batch.lanes['failure'].remaining_evaluations == 0
    batch_state = reconcile_worker_batch(batch_state, batch_id='batch', results_by_worker=records)
    assert batch_state.lanes['success'] == updated['success']
    assert batch_state.lanes['failure'] == updated['failure']
    batch_state = restore_feedback_batch_state(json.loads(json.dumps(asdict(batch_state))))
    batch_state, batch_audit = settle_worker_batch(batch_state, batch_id='batch',
        accounting_bytes=b'100|FAILED|10|4|cpu=4,node=1\n', expected_gpus=0)
    assert batch_state.ledger.totals()['actual'] == ResourceAmount(40, 0)
    assert batch_audit['failed_workers'] == ['failure'] and batch_state.attempts == 2
    initial_batch = FeedbackBatchState({'success': state, 'failure': state}, {},
        ResourceLedger(ResourceAmount(40, 0)), max_inflight=2, max_attempts=4)
    journal = tmp_path/'batch-journal'
    initial_hash = create_batch_journal(journal, initial=initial_batch)
    head = initial_hash
    def add(operation, payload):
        nonlocal head
        next_state, head = append_batch_event(journal, initial_sha256=initial_hash, expected_head=head,
                                               batch_id='batch', operation=operation, payload=payload)
        return next_state
    add('prepare', {'task_id': 'task', 'cluster_id': 'm9', 'workers': worker_lanes, 'proposals': proposals,
                    'reservation': {'cpu_core_seconds': 40, 'gpu_seconds': 0}})
    add('submission_unknown', {'reason': 'persisted before scheduler'})
    unknown, count = verify_batch_journal(journal, initial_sha256=initial_hash, expected_head=head)
    assert count == 2 and unknown.batches['batch'].phase == 'submission_unknown'
    with pytest.raises(ValueError, match='head differs'):
        append_batch_event(journal, initial_sha256=initial_hash, expected_head=initial_hash,
            batch_id='batch', operation='confirm', payload={'job_id': '100', 'confirmation_source': 'scheduler'})
    add('confirm', {'job_id': '100', 'confirmation_source': 'scheduler'})
    def encoded(raw):
        return base64.b64encode(raw).decode('ascii')
    worker_payload = {name: {'exit_code': item['exit_code'], 'manifest': encoded(item['manifest_bytes']),
        'manifest_sha256': item['manifest_sha256'], 'protocol': encoded(item['protocol_bytes']),
        'protocol_sha256': item['protocol_sha256'],
        'result': None if item['result_bytes'] is None else encoded(item['result_bytes']),
        'failure_reason': item['failure_reason']} for name, item in records.items()}
    corrupted_payload = dict(worker_payload)
    corrupted_payload['success'] = {**worker_payload['success'], 'result': 'not-base64!'}
    with pytest.raises(ValueError, match='invalid batch event'):
        add('reconcile', {'workers': corrupted_payload})
    with pytest.raises(ValueError, match='invalid batch event'):
        add('reconcile', {'workers': {'success': worker_payload['success']}})
    add('reconcile', {'workers': worker_payload})
    completed = add('settle', {'accounting': encoded(b'100|FAILED|10|4|cpu=4,node=1\n'), 'expected_gpus': 0})
    verified, count = verify_batch_journal(journal, initial_sha256=initial_hash, expected_head=head)
    assert verified == completed and count == 5 and verified.ledger.totals()['actual'] == ResourceAmount(40, 0)
    last = journal/'events'/'00000005.json'
    saved = last.read_bytes()
    last.unlink()
    with pytest.raises(ValueError, match='head differs'):
        verify_batch_journal(journal, initial_sha256=initial_hash, expected_head=head)
    last.write_bytes(saved)
    journal_unknown = tmp_path/'batch-journal-unknown'
    unknown_initial_hash = create_batch_journal(journal_unknown, initial=initial_batch)
    unknown_head = unknown_initial_hash
    def add_unknown(operation, payload):
        nonlocal unknown_head
        next_state, unknown_head = append_batch_event(journal_unknown, initial_sha256=unknown_initial_hash,
            expected_head=unknown_head, batch_id='batch', operation=operation, payload=payload)
        return next_state
    add_unknown('prepare', {'task_id': 'task', 'cluster_id': 'm9', 'workers': worker_lanes, 'proposals': proposals,
                            'reservation': {'cpu_core_seconds': 40, 'gpu_seconds': 0}})
    add_unknown('submission_unknown', {'reason': 'persisted before scheduler'})
    add_unknown('confirm', {'job_id': '100', 'confirmation_source': 'scheduler'})
    unknown_payload = dict(worker_payload)
    unknown_payload['failure'] = {**worker_payload['failure'], 'exit_code': None,
                                  'failure_reason': 'terminal job; worker exit unavailable'}
    with pytest.raises(ValueError, match='invalid batch event'):
        add_unknown('reconcile', {'workers': unknown_payload})
    add_unknown('reconcile', {'workers': unknown_payload,
        'terminal_accounting': encoded(b'100|FAILED|10|4|cpu=4,node=1\n'), 'expected_gpus': 0})
    completed_unknown = add_unknown('settle', {'accounting': encoded(b'100|FAILED|10|4|cpu=4,node=1\n'),
                                                 'expected_gpus': 0})
    replayed, _ = verify_batch_journal(journal_unknown, initial_sha256=unknown_initial_hash, expected_head=unknown_head)
    assert replayed == completed_unknown and replayed.lanes['failure'].history[-1]['status'] == 'worker_result_unknown'
    assert states['success'].pending is not None and states['failure'].pending is not None
    with pytest.raises(ValueError, match='roster'):
        consume_robust_worker_group(states, {'success': records['success']})
    bad = dict(records)
    bad['failure'] = {**records['failure'], 'manifest_sha256': '0'*64}
    with pytest.raises(ValueError, match='digest mismatch'):
        consume_robust_worker_group(states, bad)
    with pytest.raises(ValueError, match='duplicate worker candidate'):
        consume_robust_worker_group({'success': pending, 'failure': pending}, records)
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
