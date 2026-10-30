"""Durable feedback handoff, replay and corruption checks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_handoff import build_robust_solver_manifest
from phc2d.inverse.feedback_journal import (create_robust_round, read_robust_round,
                                            reconcile_failed_robust_round, reconcile_robust_round,
                                            verify_robust_journal_chain)
from phc2d.inverse.feedback_state import propose_feedback, start_feedback
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def setup_attempt(tmp_path):
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
        return np.tile([.2 + .1 * fill, .8 - .1 * fill], (len(k_points), 1)), metadata

    anchor = evaluate_feedback_candidate(source, k_points=points, config=config, lower_band=1,
        refinement=options, offsets=offsets, raster_resolution=16, solver=solver)['observation']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[0.],
        incumbent_observation=anchor, target_frequency=.5, radius=.2, min_radius=.01,
        max_radius=.4, candidate_budget=2)
    candidate = update_fourier_binary(source, sample_id='candidate', coefficients={},
                                     threshold=source.parameters['threshold'] + .1, max_mode_radius=1)
    pending = propose_feedback(state, parameter_hash=candidate.parameter_hash, parameters=[.1],
        predicted_center=.4, predicted_candidate=.5, surrogate_contract_sha256='model')
    manifest = build_robust_solver_manifest(pending, candidate, k_points=points, config=config,
                                            lower_band=1, source_index=3)
    protocol = {'offsets': offsets, 'raster_resolution': 16, 'refinement': options}
    mp, pp = tmp_path / 'manifest.json', tmp_path / 'protocol.json'
    mp.write_text(json.dumps(manifest), encoding='utf8')
    pp.write_text(json.dumps(protocol), encoding='utf8')
    mh, ph = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (mp, pp))
    output = run_case(mp, expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                      case_index=0, output_root=tmp_path / 'cases', solver=solver)
    return pending, mp.read_bytes(), pp.read_bytes(), (output / 'results.json').read_bytes(), solver


def test_journal_recovers_pending_and_consumes_identical_result_once(tmp_path):
    pending, manifest, protocol, raw, _ = setup_attempt(tmp_path)
    journal = create_robust_round(tmp_path / 'round', pending=pending,
                                  manifest_bytes=manifest, protocol_bytes=protocol)
    restored, status = read_robust_round(journal)
    assert status == 'pending_reconciliation' and restored == pending
    final = reconcile_robust_round(journal, solver_result_bytes=raw)
    assert final.remaining_evaluations == 1 and final.pending is None
    assert reconcile_robust_round(journal, solver_result_bytes=raw) == final
    restored, status = read_robust_round(journal)
    assert status == 'completed' and restored == final
    altered = json.loads(raw)
    altered['elapsed_s'] = float(altered['elapsed_s']) + 1.
    with pytest.raises(ValueError, match='conflicting'):
        reconcile_robust_round(journal, solver_result_bytes=json.dumps(altered).encode())
    with pytest.raises(FileExistsError):
        create_robust_round(journal, pending=pending, manifest_bytes=manifest, protocol_bytes=protocol)


def test_journal_rejects_changed_request_and_forged_decision(tmp_path):
    pending, manifest, protocol, raw, _ = setup_attempt(tmp_path)
    journal = create_robust_round(tmp_path / 'round', pending=pending,
                                  manifest_bytes=manifest, protocol_bytes=protocol)
    reconcile_robust_round(journal, solver_result_bytes=raw)
    receipt_path = journal / 'receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    receipt['completed_state']['remaining_evaluations'] = 2
    receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    with pytest.raises(ValueError, match='decision'):
        read_robust_round(journal)
    request_path = journal / 'request.json'
    request = json.loads(request_path.read_bytes())
    request['manifest_sha256'] = '0' * 64
    request_path.write_text(json.dumps(request), encoding='utf8')
    with pytest.raises(ValueError, match='digest'):
        read_robust_round(journal)


def test_incomplete_receipt_fails_closed_instead_of_repeating_attempt(tmp_path):
    pending, manifest, protocol, raw, _ = setup_attempt(tmp_path)
    journal = create_robust_round(tmp_path / 'round', pending=pending,
                                  manifest_bytes=manifest, protocol_bytes=protocol)
    (journal / 'receipt.json').write_bytes(b'{')
    with pytest.raises(json.JSONDecodeError):
        read_robust_round(journal)
    with pytest.raises(ValueError, match='incomplete'):
        reconcile_robust_round(journal, solver_result_bytes=raw)


def test_terminal_failure_is_replayable_and_cannot_be_replaced(tmp_path):
    from dataclasses import asdict
    from phc2d.inverse.feedback_dispatch import (FeedbackDispatchState, begin_dispatch,
        mark_submission_unknown, confirm_submission, record_terminal_failure, settle_dispatch)
    from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger
    pending, manifest, protocol, raw, _ = setup_attempt(tmp_path)
    initial = replace(pending, pending=None)
    journal = create_robust_round(tmp_path / 'round', pending=pending,
                                  manifest_bytes=manifest, protocol_bytes=protocol)
    accounting = b'10_0|OUT_OF_MEMORY|10|3|cpu=3,node=1\n'
    with pytest.raises(ValueError, match='not a confirmed'):
        reconcile_failed_robust_round(journal, accounting_bytes=b'10_0|COMPLETED|10|3|cpu=3,node=1\n',
                                      cluster='m9', job='10_0', allocated_gpus=0)
    final = reconcile_failed_robust_round(journal, accounting_bytes=accounting,
                                          cluster='m9', job='10_0', allocated_gpus=0)
    assert final.incumbent_hash == initial.incumbent_hash and final.remaining_evaluations == 1
    assert final.history[-1]['status'] == 'solver_failure'
    assert reconcile_failed_robust_round(journal, accounting_bytes=accounting,
        cluster='m9', job='10_0', allocated_gpus=0) == final
    assert read_robust_round(journal) == (final, 'completed')
    assert verify_robust_journal_chain(initial, [journal]) == (final, 'ready')
    dispatch = begin_dispatch(FeedbackDispatchState(initial, ResourceLedger(ResourceAmount(100, 0))),
        proposal=asdict(pending.pending), task_id='attempt', cluster_id='m9', reservation=ResourceAmount(40, 0))
    dispatch = mark_submission_unknown(dispatch, reason='persisted before submit')
    dispatch = confirm_submission(dispatch, job_id='10_0', confirmation_source='scheduler')
    dispatch = record_terminal_failure(dispatch, failure='OUT_OF_MEMORY')
    assert dispatch.feedback == final
    settled = settle_dispatch(dispatch, accounting_bytes=accounting, allocated_gpus=0)
    assert settled.ledger.totals()['actual'] == ResourceAmount(30, 0)
    with pytest.raises(ValueError, match='conflicting'):
        reconcile_robust_round(journal, solver_result_bytes=raw)
    receipt_path = journal / 'receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    receipt['accounting_sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    with pytest.raises(ValueError, match='digest'):
        read_robust_round(journal)


def test_two_rounds_resume_without_reusing_candidate_or_budget(tmp_path):
    pending, manifest, protocol, raw, solver = setup_attempt(tmp_path)
    initial = replace(pending, pending=None)
    first = create_robust_round(tmp_path / 'round1', pending=pending,
                                manifest_bytes=manifest, protocol_bytes=protocol)
    state = reconcile_robust_round(first, solver_result_bytes=raw)
    assert state.history[-1]['accepted'] and state.remaining_evaluations == 1
    original_manifest = json.loads(manifest)
    source = FullCellStructureSpec.from_mapping(original_manifest['rows'][0]['structure'])
    candidate = update_fourier_binary(source, sample_id='second', coefficients={},
                                     threshold=source.parameters['threshold'] + .03, max_mode_radius=1)
    with pytest.raises(ValueError, match='already evaluated'):
        propose_feedback(state, parameter_hash=source.parameter_hash, parameters=[.1],
                         predicted_center=.5, predicted_candidate=.6, surrogate_contract_sha256='model')
    pending2 = propose_feedback(state, parameter_hash=candidate.parameter_hash, parameters=[.13],
                               predicted_center=.5, predicted_candidate=.6, surrogate_contract_sha256='model')
    config = FullCellMpbBandSolverConfig(**original_manifest['solver_config'])
    manifest2 = build_robust_solver_manifest(pending2, candidate, k_points=original_manifest['k_points'],
                                             config=config, lower_band=1, source_index=3)
    raw2 = json.dumps(manifest2).encode()
    round2 = create_robust_round(tmp_path / 'round2', pending=pending2,
                                 manifest_bytes=raw2, protocol_bytes=protocol)
    resumed, status = read_robust_round(round2)
    assert status == 'pending_reconciliation' and resumed.remaining_evaluations == 1
    chain_state, chain_status = verify_robust_journal_chain(initial, [first, round2])
    assert chain_status == 'pending_reconciliation' and chain_state == resumed
    with pytest.raises(ValueError, match='twice'):
        verify_robust_journal_chain(initial, [first, first])
    with pytest.raises(ValueError, match='previous verified state'):
        verify_robust_journal_chain(initial, [round2, first])
    receipt = first / 'receipt.json'
    parked = first / 'receipt.saved'
    receipt.rename(parked)
    with pytest.raises(ValueError, match='unfinished round must be last'):
        verify_robust_journal_chain(initial, [first, round2])
    parked.rename(receipt)
    manifest_path = tmp_path / 'manifest2.json'
    manifest_path.write_bytes(raw2)
    output = run_case(manifest_path, expected_sha256=hashlib.sha256(raw2).hexdigest(),
        protocol_path=tmp_path / 'protocol.json', expected_protocol_sha256=hashlib.sha256(protocol).hexdigest(),
        case_index=0, output_root=tmp_path / 'cases2', solver=solver)
    final = reconcile_robust_round(round2, solver_result_bytes=(output / 'results.json').read_bytes())
    assert final.remaining_evaluations == 0 and len(final.history) == 2 and final.pending is None
    verified, status = verify_robust_journal_chain(initial, [first, round2])
    assert status == 'budget_exhausted' and verified == final
