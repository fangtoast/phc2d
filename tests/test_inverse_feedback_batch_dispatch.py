"""Multi-worker feedback reservations and recovery contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
from dataclasses import asdict

import pytest

from phc2d.inverse.feedback_batch_dispatch import (FeedbackBatchState, confirm_worker_batch,
    mark_batch_submission_unknown, prepare_worker_batch, release_unstarted_worker_batch,
    restore_feedback_batch_state)
from phc2d.inverse.feedback_state import start_feedback
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger


def lane(identity):
    observation = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
        'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
        'resolution_stable': True, 'adaptive_refinement_converged': True},
        'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'solver'}}
    return start_feedback(incumbent_hash=identity, incumbent_parameters=[0.], incumbent_observation=observation,
        target_frequency=.5, radius=.5, min_radius=.05, max_radius=1., candidate_budget=2)


def proposal(identity):
    return {'parameter_hash': identity, 'parameters': [.1], 'predicted_center': .4,
            'predicted_candidate': .5, 'surrogate_contract_sha256': 'model'}


def prepare(state, name, workers, proposals, amount=40):
    return prepare_worker_batch(state, batch_id=name, task_id='task-'+name, cluster_id='m9',
                                workers=workers, proposals=proposals, reservation=ResourceAmount(amount, 0))


def test_batch_unknown_submission_recovers_without_duplicate_or_unconfirmed_release():
    state = FeedbackBatchState({'a': lane('a'), 'b': lane('b')}, {}, ResourceLedger(ResourceAmount(80, 0)), 2, 4)
    state = prepare(state, 'first', {'w1': 'a', 'w2': 'b'},
                    {'w1': proposal('1'*64), 'w2': proposal('2'*64)})
    assert state.attempts == 2 and state.ledger.totals()['reserved'] == ResourceAmount(40, 0)
    state = mark_batch_submission_unknown(state, batch_id='first', reason='submission outcome unknown')
    state = restore_feedback_batch_state(json.loads(json.dumps(asdict(state))))
    with pytest.raises(ValueError, match='batch'):
        prepare(state, 'first', {'w1': 'a'}, {'w1': proposal('3'*64)})
    with pytest.raises(ValueError, match='no-future-execution'):
        release_unstarted_worker_batch(state, batch_id='first', confirmed_not_started=True,
            confirmed_no_future_execution=False, confirmation_source='scheduler')
    state = release_unstarted_worker_batch(state, batch_id='first', confirmed_not_started=True,
        confirmed_no_future_execution=True, confirmation_source='scheduler')
    assert not state.batches and state.ledger.totals()['reserved'] == ResourceAmount(0, 0)
    assert state.attempts == 2 and state.lanes['a'].remaining_evaluations == 2
    state = prepare(state, 'second', {'w1': 'a', 'w2': 'b'},
                    {'w1': proposal('3'*64), 'w2': proposal('4'*64)})
    assert state.attempts == 4
    with pytest.raises(ValueError, match='attempt budget'):
        prepare(state, 'third', {'w1': 'a'}, {'w1': proposal('5'*64)}, amount=0)


def test_batch_job_and_reservation_cannot_be_reused_between_active_groups():
    state = FeedbackBatchState({'a': lane('a'), 'b': lane('b')}, {}, ResourceLedger(ResourceAmount(80, 0)), 2, 3)
    state = prepare(state, 'first', {'w': 'a'}, {'w': proposal('1'*64)})
    state = mark_batch_submission_unknown(state, batch_id='first', reason='submitted')
    state = confirm_worker_batch(state, batch_id='first', job_id='100', confirmation_source='scheduler')
    with pytest.raises(ValueError, match='duplicate'):
        prepare(state, 'duplicate', {'w': 'b'}, {'w': proposal('1'*64)})
    state = prepare(state, 'second', {'w': 'b'}, {'w': proposal('2'*64)})
    state = mark_batch_submission_unknown(state, batch_id='second', reason='submitted')
    with pytest.raises(ValueError, match='duplicate'):
        confirm_worker_batch(state, batch_id='second', job_id='100', confirmation_source='scheduler')
    corrupt = json.loads(json.dumps(asdict(state)))
    corrupt['batches']['second']['task_id'] = 'task-first'
    with pytest.raises(ValueError, match='duplicate'):
        restore_feedback_batch_state(corrupt)
