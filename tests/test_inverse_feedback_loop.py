"""Tests for recoverable solver-gated feedback orchestration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.feedback_loop import SolverAttemptFailed, run_feedback_loop
from phc2d.inverse.feedback_state import start_feedback


def observation(identity, lower, upper):
    return {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': lower,
                    'refined_upper_edge_frequency': upper, 'last_refinement_stable': True,
                    'resolution_stable': True, 'adaptive_refinement_converged': True},
            'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'contract'}}


def initial():
    return start_feedback(incumbent_hash='a', incumbent_parameters=[0],
                          incumbent_observation=observation('a', .4, .6), target_frequency=.5,
                          radius=.5, min_radius=.05, max_radius=1., candidate_budget=2)


def proposal(state):
    identity = 'b' if state.remaining_evaluations == 2 else 'c'
    return {'parameter_hash': identity, 'parameters': [.1], 'predicted_center': .4,
            'predicted_candidate': .5, 'surrogate_contract_sha256': 'checkpoint'}


def test_records_pending_before_solver_and_consumes_failure():
    saved = []
    def evaluate(state):
        assert saved[-1].pending == state.pending
        if state.pending.parameter_hash == 'c':
            raise SolverAttemptFailed('timeout')
        return observation('b', .35, .65)
    result, status = run_feedback_loop(initial(), propose=proposal, evaluate=evaluate, persist=saved.append)
    assert status == 'budget_exhausted' and result.remaining_evaluations == 0
    assert result.incumbent_hash == 'b'
    assert [x['status'] for x in result.history] == ['compared', 'solver_failure']
    assert len(saved) == 4 and saved[-1].pending is None


def test_crash_leaves_pending_for_manual_reconciliation_without_retry():
    saved = []
    calls = []
    def evaluate(state):
        calls.append(state.pending.parameter_hash)
        raise RuntimeError('unexpected crash')
    with pytest.raises(RuntimeError, match='unexpected crash'):
        run_feedback_loop(initial(), propose=proposal, evaluate=evaluate, persist=saved.append)
    pending = saved[-1]
    recovered, status = run_feedback_loop(pending, propose=proposal, evaluate=evaluate, persist=saved.append)
    assert status == 'pending_reconciliation' and recovered == pending
    assert calls == ['b']


def test_no_candidate_is_not_a_solver_attempt():
    result, status = run_feedback_loop(initial(), propose=lambda s: None,
                                       evaluate=lambda s: None, persist=lambda s: None)
    assert status == 'proposal_exhausted' and result.remaining_evaluations == 2
