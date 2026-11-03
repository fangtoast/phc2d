"""Shared feedback allocation budget across independent lanes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
import base64
import hashlib
from dataclasses import asdict

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.feedback_dispatch import (FeedbackCampaignState, apply_campaign_event,
                                             restore_feedback_campaign)
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_handoff import build_robust_solver_manifest
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.feedback_state import start_feedback
from phc2d.inverse.feedback_journal import (append_campaign_event, create_campaign_journal,
                                             replay_campaign_journal, verify_campaign_journal)
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def initial_lane(identity):
    observation = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
        'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
        'resolution_stable': True, 'adaptive_refinement_converged': True},
        'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'solver'}}
    return start_feedback(incumbent_hash=identity, incumbent_parameters=[0.], incumbent_observation=observation,
        target_frequency=.5, radius=.5, min_radius=.05, max_radius=1., candidate_budget=2)


def event(state, lane, operation, **payload):
    return apply_campaign_event(state, lane=lane, operation=operation, payload=payload)


def proposal(candidate):
    return {'parameter_hash': candidate, 'parameters': [.1], 'predicted_center': .4,
            'predicted_candidate': .5, 'surrogate_contract_sha256': 'checkpoint'}


def test_shared_budget_unknown_submission_recovery_and_failure_settlement():
    state = FeedbackCampaignState({name: initial_lane(name) for name in ('a', 'b', 'c')}, {},
                                   ResourceLedger(ResourceAmount(80, 0)), max_inflight=2, max_attempts=3)
    state = event(state, 'a', 'begin', proposal=proposal('a1'), task_id='task-a', cluster_id='m9',
                  reservation=ResourceAmount(40, 0))
    state = event(state, 'a', 'submission_unknown', reason='submit outcome unknown')
    with pytest.raises(ValueError, match='previous dispatch'):
        event(state, 'a', 'begin', proposal=proposal('a2'), task_id='retry', cluster_id='m9',
              reservation=ResourceAmount(0, 0))
    state = event(state, 'b', 'begin', proposal=proposal('b1'), task_id='task-b', cluster_id='m9',
                  reservation=ResourceAmount(40, 0))
    state = restore_feedback_campaign(json.loads(json.dumps(asdict(state))))
    assert state.ledger.totals()['reserved'] == ResourceAmount(80, 0)
    with pytest.raises(ValueError, match='concurrency'):
        event(state, 'c', 'begin', proposal=proposal('c1'), task_id='task-c', cluster_id='m9',
              reservation=ResourceAmount(1, 0))
    state = event(state, 'a', 'confirm', job_id='100', confirmation_source='scheduler')
    state = event(state, 'b', 'submission_unknown', reason='persisted before submission')
    with pytest.raises(ValueError, match='duplicate'):
        event(state, 'b', 'confirm', job_id='100', confirmation_source='scheduler')
    state = event(state, 'b', 'confirm', job_id='101', confirmation_source='scheduler')
    state = event(state, 'a', 'failure', failure='OUT_OF_MEMORY')
    state = event(state, 'a', 'settle', accounting_bytes=b'100|OUT_OF_MEMORY|10|3|cpu=3,node=1\n',
                  allocated_gpus=0)
    assert state.lanes['a'].remaining_evaluations == 1
    assert state.ledger.totals()['actual'] == ResourceAmount(30, 0)
    assert state.ledger.totals()['reserved'] == ResourceAmount(40, 0)
    with pytest.raises(ValueError, match='budget'):
        event(state, 'c', 'begin', proposal=proposal('c1'), task_id='task-c', cluster_id='m9',
              reservation=ResourceAmount(11, 0))
    state = event(state, 'c', 'begin', proposal=proposal('c1'), task_id='task-c', cluster_id='m9',
                  reservation=ResourceAmount(10, 0))
    assert restore_feedback_campaign(json.loads(json.dumps(asdict(state)))) == state
    state = event(state, 'b', 'failure', failure='FAILED')
    state = event(state, 'b', 'settle', accounting_bytes=b'101|FAILED|10|3|cpu=3,node=1\n',
                  allocated_gpus=0)
    with pytest.raises(ValueError, match='attempt limit'):
        event(state, 'a', 'begin', proposal=proposal('a2'), task_id='task-a2', cluster_id='m9',
              reservation=ResourceAmount(1, 0))
    corrupted = json.loads(json.dumps(asdict(state)))
    corrupted['dispatches']['c']['task_id'] = 'task-b'
    with pytest.raises(ValueError):
        restore_feedback_campaign(corrupted)


def test_campaign_journal_replays_interleaved_lanes_and_rejects_stale_heads(tmp_path):
    initial = FeedbackCampaignState({name: initial_lane(name) for name in ('a', 'b')}, {},
                                    ResourceLedger(ResourceAmount(80, 0)), max_inflight=2, max_attempts=3)
    journal = tmp_path/'campaign'
    initial_sha = create_campaign_journal(journal, initial=initial)
    head = initial_sha
    guard = journal/'.writer.lock'
    guard.write_text('other writer', encoding='ascii')
    with pytest.raises(ValueError, match='writer guard'):
        append_campaign_event(journal, initial_sha256=initial_sha, expected_head=head,
            lane='a', operation='submission_unknown', payload={'reason': 'not dispatched'})
    assert guard.read_text(encoding='ascii') == 'other writer'
    assert not list((journal/'events').iterdir())
    guard.unlink()
    def append(lane, operation, payload):
        nonlocal head
        state, head = append_campaign_event(journal, initial_sha256=initial_sha, expected_head=head,
                                            lane=lane, operation=operation, payload=payload)
        return state
    for lane in ('a', 'b'):
        append(lane, 'begin', {'proposal': proposal(lane+'1'), 'task_id': 'task-'+lane, 'cluster_id': 'm9',
            'reservation': {'cpu_core_seconds': 40, 'gpu_seconds': 0}})
        append(lane, 'submission_unknown', {'reason': 'persisted before submit'})
    with pytest.raises(ValueError, match='head differs'):
        append_campaign_event(journal, initial_sha256=initial_sha, expected_head=initial_sha,
            lane='a', operation='confirm', payload={'job_id': '100', 'confirmation_source': 'scheduler'})
    assert not guard.exists()
    append('a', 'confirm', {'job_id': '100', 'confirmation_source': 'scheduler'})
    with pytest.raises(ValueError, match='duplicate'):
        append('b', 'confirm', {'job_id': '100', 'confirmation_source': 'scheduler'})
    append('b', 'confirm', {'job_id': '101', 'confirmation_source': 'scheduler'})
    append('a', 'failure', {'failure': 'FAILED'})
    accounting = base64.b64encode(b'100|FAILED|10|3|cpu=3,node=1\n').decode()
    latest = append('a', 'settle', {'accounting': accounting, 'allocated_gpus': 0})
    restored, restored_head, count = replay_campaign_journal(journal, initial_sha256=initial_sha)
    assert restored == latest and restored_head == head and count == 8
    assert restored.ledger.totals()['actual'] == ResourceAmount(30, 0)
    assert restored.ledger.totals()['reserved'] == ResourceAmount(40, 0)
    assert restored.dispatches['b'].job_id == '101'
    assert verify_campaign_journal(journal, initial_sha256=initial_sha, expected_head=head) == (latest, 8)
    last = journal/'events'/'00000008.json'
    saved = last.read_bytes()
    last.unlink()
    with pytest.raises(ValueError, match='head differs'):
        verify_campaign_journal(journal, initial_sha256=initial_sha, expected_head=head)
    last.write_bytes(saved)
    path = journal/'events'/'00000004.json'
    record = json.loads(path.read_bytes())
    record['lane'] = 'a'
    path.write_text(json.dumps(record), encoding='utf8')
    with pytest.raises(ValueError):
        replay_campaign_journal(journal, initial_sha256=initial_sha)


def test_campaign_journal_consumes_checked_nominal_solver_result_and_other_lane_failure(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
        'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1, 'resolution_tolerance': 1e-6}
    points = np.array([[0., 0.], [.25, 0.]])
    def solver(grid, *, k_points, config):
        fill = float(grid.mean())
        provenance = {'grid_sha256': hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest(),
            'resolution': config.resolution, 'polarization': config.polarization, 'num_bands': config.num_bands,
            'mesh_size': config.mesh_size, 'band_guard_count': config.band_guard_count,
            'band_order_policy': config.band_order_policy, 'eigensolver_tolerance': config.eigensolver_tolerance,
            'epsilon_air': config.epsilon_air, 'epsilon_material': config.epsilon_material,
            'solver_backend': 'mock', 'solver_version': '1'}
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), provenance
    anchor = evaluate_feedback_candidate(source, k_points=points, config=config, lower_band=1,
        refinement=options, offsets=(0.,), raster_resolution=16, solver=solver)['observation']
    def lane():
        return start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[0.],
            incumbent_observation=anchor, target_frequency=.5, radius=.2, min_radius=.01,
            max_radius=.4, candidate_budget=2)
    initial = FeedbackCampaignState({'a': lane(), 'b': lane()}, {}, ResourceLedger(ResourceAmount(80, 0)), 2, 4)
    candidate = update_fourier_binary(source, sample_id='candidate', coefficients={},
        threshold=source.parameters['threshold']+.1, max_mode_radius=1)
    journal = tmp_path/'campaign'
    initial_sha = create_campaign_journal(journal, initial=initial)
    head = initial_sha
    def append(lane_id, operation, payload):
        nonlocal head
        state, head = append_campaign_event(journal, initial_sha256=initial_sha, expected_head=head,
            lane=lane_id, operation=operation, payload=payload)
        return state
    for lane_id, hash_value, job in (('a', candidate.parameter_hash, '100'), ('b', 'other', '101')):
        append(lane_id, 'begin', {'proposal': proposal(hash_value), 'task_id': 'task-'+lane_id, 'cluster_id': 'm9',
            'reservation': {'cpu_core_seconds': 40, 'gpu_seconds': 0}})
        append(lane_id, 'submission_unknown', {'reason': 'persisted before scheduler call'})
        state = append(lane_id, 'confirm', {'job_id': job, 'confirmation_source': 'scheduler'})
    request = build_robust_solver_manifest(state.lanes['a'], candidate, k_points=points, config=config,
                                            lower_band=1, source_index=3)
    request_raw = json.dumps(request).encode()
    protocol_raw = json.dumps({'offsets': [0.], 'raster_resolution': 16, 'refinement': options}).encode()
    request_sha, protocol_sha = hashlib.sha256(request_raw).hexdigest(), hashlib.sha256(protocol_raw).hexdigest()
    request_path, protocol_path = tmp_path/'manifest.json', tmp_path/'protocol.json'
    request_path.write_bytes(request_raw)
    protocol_path.write_bytes(protocol_raw)
    run = run_case(request_path, expected_sha256=request_sha, protocol_path=protocol_path,
                   expected_protocol_sha256=protocol_sha, case_index=0, output_root=tmp_path/'solver', solver=solver)
    result_raw = (run/'results.json').read_bytes()
    enc = lambda raw: base64.b64encode(raw).decode('ascii')
    result_payload = {'manifest': enc(request_raw), 'manifest_sha256': request_sha,
                      'protocol': enc(protocol_raw), 'protocol_sha256': protocol_sha, 'result': enc(result_raw)}
    with pytest.raises(ValueError, match='invalid dispatch event') as error:
        append('b', 'result', result_payload)
    assert 'candidate' in str(error.value.__cause__)
    state = append('a', 'result', result_payload)
    assert state.lanes['a'].remaining_evaluations == 1
    assert state.lanes['a'].history[-1]['status'] == 'compared'
    append('a', 'settle', {'accounting': enc(b'100|COMPLETED|10|3|cpu=3,node=1\n'), 'allocated_gpus': 0})
    append('b', 'failure', {'failure': 'TIMEOUT'})
    state = append('b', 'settle', {'accounting': enc(b'101|TIMEOUT|10|3|cpu=3,node=1\n'), 'allocated_gpus': 0})
    assert state.lanes['b'].remaining_evaluations == 1
    assert state.lanes['b'].history[-1]['status'] == 'solver_failure'
    assert state.ledger.totals()['actual'] == ResourceAmount(60, 0)
    restored, _ = verify_campaign_journal(journal, initial_sha256=initial_sha, expected_head=head)
    assert restored == state
    second = update_fourier_binary(source, sample_id='second', coefficients={},
        threshold=source.parameters['threshold']+.05, max_mode_radius=1)
    second_proposal = {**proposal(second.parameter_hash), 'parameters': [.05]}
    append('a', 'begin', {'proposal': second_proposal, 'task_id': 'task-a2', 'cluster_id': 'm9',
        'reservation': {'cpu_core_seconds': 20, 'gpu_seconds': 0}})
    append('a', 'submission_unknown', {'reason': 'persisted before scheduler call'})
    state = append('a', 'confirm', {'job_id': '102', 'confirmation_source': 'scheduler'})
    request2 = build_robust_solver_manifest(state.lanes['a'], second, k_points=points, config=config,
                                             lower_band=1, source_index=3)
    request2_raw = json.dumps(request2).encode()
    request2_sha = hashlib.sha256(request2_raw).hexdigest()
    request2_path = tmp_path/'manifest2.json'
    request2_path.write_bytes(request2_raw)
    run2 = run_case(request2_path, expected_sha256=request2_sha, protocol_path=protocol_path,
                    expected_protocol_sha256=protocol_sha, case_index=0, output_root=tmp_path/'solver2', solver=solver)
    result2 = {'manifest': enc(request2_raw), 'manifest_sha256': request2_sha,
               'protocol': enc(protocol_raw), 'protocol_sha256': protocol_sha,
               'result': enc((run2/'results.json').read_bytes())}
    state = append('a', 'result', result2)
    assert state.lanes['a'].remaining_evaluations == 0
    assert len(state.lanes['a'].history) == 2
    assert all(row['status'] == 'compared' for row in state.lanes['a'].history)
    state = append('a', 'settle', {'accounting': enc(b'102|COMPLETED|5|3|cpu=3,node=1\n'), 'allocated_gpus': 0})
    assert state.ledger.totals()['actual'] == ResourceAmount(75, 0)
    restored, _ = verify_campaign_journal(journal, initial_sha256=initial_sha, expected_head=head)
    assert restored == state
