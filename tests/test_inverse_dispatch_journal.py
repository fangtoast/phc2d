"""Replay feedback dispatch transitions and retain scheduler accounting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import base64
import json

import pytest

from phc2d.inverse.feedback_dispatch import FeedbackDispatchState
from phc2d.inverse.feedback_journal import (append_dispatch_event, create_dispatch_journal,
                                            replay_dispatch_journal)
from phc2d.inverse.feedback_state import finish_feedback, start_feedback
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger


def _observation(identity, lower=.4, upper=.6):
    return {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': lower,
                    'refined_upper_edge_frequency': upper, 'last_refinement_stable': True,
                    'resolution_stable': True, 'adaptive_refinement_converged': True},
            'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'solver'}}


def _initial():
    feedback = start_feedback(incumbent_hash='a', incumbent_parameters=[0.],
        incumbent_observation=_observation('a'), target_frequency=.5, radius=.5,
        min_radius=.05, max_radius=1., candidate_budget=3)
    return FeedbackDispatchState(feedback, ResourceLedger(ResourceAmount(200, 0)))


def _begin(identity, task):
    return {'proposal': {'parameter_hash': identity, 'parameters': [.1], 'predicted_center': .4,
                         'predicted_candidate': .5, 'surrogate_contract_sha256': 'checkpoint'},
            'task_id': task, 'cluster_id': 'm9',
            'reservation': {'cpu_core_seconds': 20, 'gpu_seconds': 0}}


def _b64(value):
    return base64.b64encode(value).decode('ascii')


def test_replay_two_attempts_preserves_failures_results_and_cost(tmp_path, monkeypatch):
    from phc2d.inverse import feedback_dispatch

    def consume(feedback, **kwargs):
        assert kwargs['result'] == {'example': 1}
        return finish_feedback(feedback, observation=_observation('c', .35, .65))

    monkeypatch.setattr(feedback_dispatch, 'consume_robust_solver_result', consume)
    folder = tmp_path / 'dispatch'
    genesis = head = create_dispatch_journal(folder, initial=_initial())

    def append(operation, payload):
        nonlocal head
        state, head = append_dispatch_event(folder, initial_sha256=genesis, expected_head=head,
                                             operation=operation, payload=payload)
        assert replay_dispatch_journal(folder, initial_sha256=genesis)[0] == state
        return state

    append('begin', _begin('b', 'first'))
    append('submission_unknown', {'reason': 'persisted before submission'})
    append('confirm', {'job_id': '10', 'confirmation_source': 'scheduler'})
    append('failure', {'failure': 'TIMEOUT'})
    append('settle', {'accounting': _b64(b'10|TIMEOUT|10|3|cpu=3,node=1\n'), 'allocated_gpus': 0})
    append('begin', _begin('c', 'second'))
    append('submission_unknown', {'reason': 'persisted before submission'})
    append('confirm', {'job_id': '11', 'confirmation_source': 'scheduler'})
    append('result', {'manifest': _b64(b'{}'), 'manifest_sha256': 'x', 'protocol': _b64(b'{}'),
                      'protocol_sha256': 'y', 'result': _b64(b'{"example": 1}')})
    final = append('settle', {'accounting': _b64(b'11|COMPLETED|5|3|cpu=3,node=1\n'), 'allocated_gpus': 0})
    replayed, replayed_head, count = replay_dispatch_journal(folder, initial_sha256=genesis)
    assert replayed == final and replayed_head == head and count == 10
    assert final.feedback.incumbent_hash == 'c' and final.feedback.remaining_evaluations == 1
    assert final.ledger.totals()['actual'] == ResourceAmount(45, 0)
    assert [record.allocation_ids for record in final.ledger.records] == [('m9:10',), ('m9:11',)]


def test_stale_head_and_tampering_fail_closed(tmp_path):
    folder = tmp_path / 'dispatch'
    genesis = create_dispatch_journal(folder, initial=_initial())
    _, head = append_dispatch_event(folder, initial_sha256=genesis, expected_head=genesis,
                                    operation='begin', payload=_begin('b', 'first'))
    with pytest.raises(ValueError, match='head differs'):
        append_dispatch_event(folder, initial_sha256=genesis, expected_head=genesis,
                              operation='submission_unknown', payload={'reason': 'stale'})
    with pytest.raises(ValueError, match='initial hash'):
        replay_dispatch_journal(folder, initial_sha256='0' * 64)
    event_path = folder / 'events' / '00000001.json'
    event = json.loads(event_path.read_bytes())
    event['payload']['task_id'] = 'tampered'
    event_path.write_text(json.dumps(event), encoding='utf8')
    with pytest.raises(ValueError, match='transition differs'):
        replay_dispatch_journal(folder, initial_sha256=genesis)
    assert head != genesis


def test_invalid_transition_does_not_append_and_missing_event_is_rejected(tmp_path):
    folder = tmp_path / 'dispatch'
    genesis = head = create_dispatch_journal(folder, initial=_initial())
    _, head = append_dispatch_event(folder, initial_sha256=genesis, expected_head=head,
                                    operation='begin', payload=_begin('b', 'first'))
    with pytest.raises(ValueError):
        append_dispatch_event(folder, initial_sha256=genesis, expected_head=head,
                              operation='confirm', payload={'job_id': '10', 'confirmation_source': 'scheduler'})
    assert len(list((folder / 'events').iterdir())) == 1
    _, head = append_dispatch_event(folder, initial_sha256=genesis, expected_head=head,
                                    operation='submission_unknown', payload={'reason': 'persisted'})
    (folder / 'events' / '00000001.json').unlink()
    with pytest.raises(ValueError, match='sequence differs'):
        replay_dispatch_journal(folder, initial_sha256=genesis)


def test_cli_inspects_dispatch_only_with_trusted_head(tmp_path, capsys):
    from phc2d.cli.main import main

    folder = tmp_path / 'dispatch'
    genesis = create_dispatch_journal(folder, initial=_initial())
    _, head = append_dispatch_event(folder, initial_sha256=genesis, expected_head=genesis,
                                    operation='begin', payload=_begin('b', 'first'))
    args = ['inverse', 'feedback-inspect', '--dispatch-directory', str(folder),
            '--initial-sha256', genesis, '--head-sha256', head]
    main(args)
    summary = json.loads(capsys.readouterr().out)
    assert summary['phase'] == 'prepared' and summary['pending_hash'] == 'b'
    assert summary['event_count'] == 1 and summary['reserved']['cpu_core_seconds'] == 20
    assert summary['evidence_scope'] == 'journal_replay_only_initial_and_scheduler_evidence_not_independently_verified'
    with pytest.raises(ValueError, match='head differs'):
        main(args[:-1] + [genesis])
    with pytest.raises(SystemExit, match='requires'):
        main(['inverse', 'feedback-inspect', '--dispatch-directory', str(folder), '--initial-sha256', genesis])
