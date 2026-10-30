"""Exercise persisted feedback dispatch, ambiguity and resource accounting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import json
from dataclasses import asdict

import pytest

from phc2d.inverse.feedback_dispatch import (
    FeedbackDispatchState, begin_dispatch, confirm_submission, mark_submission_unknown,
    record_robust_result, record_terminal_failure, resolve_unstarted,
    restore_feedback_dispatch, settle_dispatch,
)
from phc2d.inverse.feedback_state import finish_feedback, start_feedback
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger


def observation(identity, lower=.4, upper=.6):
    return {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': lower,
                    'refined_upper_edge_frequency': upper, 'last_refinement_stable': True,
                    'resolution_stable': True, 'adaptive_refinement_converged': True},
            'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'solver'}}


def prepared(limit=100):
    feedback = start_feedback(incumbent_hash='a', incumbent_parameters=[0.],
        incumbent_observation=observation('a'), target_frequency=.5, radius=.5,
        min_radius=.05, max_radius=1., candidate_budget=3)
    return begin_dispatch(FeedbackDispatchState(feedback, ResourceLedger(ResourceAmount(limit, 0))),
        proposal={'parameter_hash': 'b', 'parameters': [.1], 'predicted_center': .4,
                  'predicted_candidate': .5, 'surrogate_contract_sha256': 'checkpoint'},
        task_id='attempt', cluster_id='m9', reservation=ResourceAmount(20, 0))


def restored(state):
    return restore_feedback_dispatch(json.loads(json.dumps(asdict(state))))


def test_ambiguous_submission_requires_confirmed_nonexecution_and_no_future_execution():
    state = restored(mark_submission_unknown(prepared(), reason='connection lost'))
    with pytest.raises(ValueError):
        resolve_unstarted(state, confirmed_not_started=True, confirmed_no_future_execution=False,
                          confirmation_source='scheduler')
    with pytest.raises(ValueError):
        resolve_unstarted(state, confirmed_not_started=False, confirmed_no_future_execution=True,
                          confirmation_source='scheduler')
    released = resolve_unstarted(state, confirmed_not_started=True, confirmed_no_future_execution=True,
                                 confirmation_source='scheduler')
    assert released.feedback.pending is None
    assert released.feedback.remaining_evaluations == 3
    assert released.ledger.totals()['reserved'] == ResourceAmount(0, 0)
    assert released.ledger.records[0].source == 'scheduler'


def test_failure_is_counted_once_and_must_match_exact_scheduler_record():
    with pytest.raises(ValueError):
        confirm_submission(prepared(), job_id='10_0', confirmation_source='scheduler')
    state = mark_submission_unknown(prepared(), reason='persisted before submit')
    state = confirm_submission(state, job_id='10_0', confirmation_source='scheduler')
    with pytest.raises(ValueError):
        confirm_submission(state, job_id='10_0', confirmation_source='scheduler')
    state = restored(record_terminal_failure(state, failure='OUT_OF_MEMORY', reason='scheduler OOM'))
    assert state.feedback.incumbent_hash == 'a' and state.feedback.remaining_evaluations == 2
    with pytest.raises(ValueError):
        settle_dispatch(state, accounting_bytes=b'10_0|COMPLETED|10|3|cpu=3,node=1\n', allocated_gpus=0)
    with pytest.raises(ValueError):
        settle_dispatch(state, accounting_bytes=b'11_0|OUT_OF_MEMORY|10|3|cpu=3,node=1\n', allocated_gpus=0)
    done = settle_dispatch(state, accounting_bytes=(b'10_0|OUT_OF_MEMORY|10|3|cpu=3,node=1\n'
        b'10_0.batch|OUT_OF_MEMORY|10|3|cpu=3,node=1\n'), allocated_gpus=0)
    assert done.dispatch is None and done.ledger.totals()['actual'] == ResourceAmount(30, 0)
    assert done.ledger.records[0].allocation_ids == ('m9:10_0',)
    assert restored(done) == done


def test_result_path_and_budget_overrun_block_followup(monkeypatch):
    from phc2d.inverse import feedback_dispatch
    def consume(feedback, **kwargs):
        assert kwargs['result'] == {'example': 1}
        return finish_feedback(feedback, observation=observation('b', .35, .65))
    monkeypatch.setattr(feedback_dispatch, 'consume_robust_solver_result', consume)
    state = mark_submission_unknown(prepared(limit=20), reason='persisted before submit')
    state = confirm_submission(state, job_id='10', confirmation_source='scheduler')
    state = record_robust_result(state, manifest_bytes=b'{}', manifest_sha256='x',
                                 protocol_bytes=b'{}', protocol_sha256='y', result_bytes=b'{"example": 1}')
    assert restored(state) == state and state.feedback.incumbent_hash == 'b'
    state = settle_dispatch(state, accounting_bytes=b'10|COMPLETED|10|3|cpu=3,node=1\n', allocated_gpus=0)
    assert state.ledger.status() == 'budget_exceeded'
    with pytest.raises(ValueError, match='budget'):
        begin_dispatch(state, proposal={}, task_id='next', cluster_id='m9', reservation=ResourceAmount(0, 0))
