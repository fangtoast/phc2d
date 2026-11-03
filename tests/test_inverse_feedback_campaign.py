"""Shared feedback allocation budget across independent lanes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
from dataclasses import asdict

import pytest

from phc2d.inverse.feedback_dispatch import (FeedbackCampaignState, apply_campaign_event,
                                             restore_feedback_campaign)
from phc2d.inverse.feedback_state import start_feedback
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger


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
