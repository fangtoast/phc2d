"""Pure feedback state transitions preserve verified incumbents and budgets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from dataclasses import asdict

import pytest

from phc2d.inverse.feedback_state import finish_feedback, propose_feedback, restore_feedback, start_feedback


def observation(identity, lower, upper, *, stable=True, contract='solver'):
    return {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': lower,
                    'refined_upper_edge_frequency': upper, 'last_refinement_stable': stable,
                    'resolution_stable': stable, 'adaptive_refinement_converged': stable},
            'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': contract}}


def state(budget=3):
    return start_feedback(incumbent_hash='a', incumbent_parameters=[0.],
                          incumbent_observation=observation('a', .4, .6), target_frequency=.5,
                          radius=.5, min_radius=.05, max_radius=1., candidate_budget=budget)


def proposal(current, identity, position=.1):
    return propose_feedback(current, parameter_hash=identity, parameters=[position],
                            predicted_center=.4, predicted_candidate=.5,
                            surrogate_contract_sha256='checkpoint')


def test_accept_only_verified_improvement_and_resume_from_serializable_state():
    initial = state()
    pending = proposal(initial, 'b')
    result = finish_feedback(pending, observation=observation('b', .35, .65))
    assert result.incumbent_hash == 'b'
    assert result.history[-1]['accepted']
    assert result.remaining_evaluations == 2 and result.pending is None
    assert result.radius == 1.
    assert restore_feedback(asdict(result)) == result
    assert initial.incumbent_hash == 'a' and initial.remaining_evaluations == 3


def test_rejects_false_improvement_and_preserves_original_incumbent():
    result = finish_feedback(proposal(state(), 'b'), observation=observation('b', .45, .55))
    assert result.incumbent_hash == 'a'
    assert not result.history[-1]['accepted']
    assert result.radius == .25
    with pytest.raises(ValueError, match='already evaluated'):
        proposal(result, 'b')


def test_failure_and_numerical_gate_consume_attempt_without_replacing_incumbent():
    result = finish_feedback(proposal(state(), 'b'), observation=observation('b', .35, .65, stable=False))
    assert result.incumbent_hash == 'a' and result.remaining_evaluations == 2
    assert result.history[-1]['status'] == 'numerical_gate_failed'
    result = finish_feedback(proposal(result, 'c'), failure_reason='timeout')
    assert result.incumbent_hash == 'a' and result.remaining_evaluations == 1
    assert result.history[-1]['status'] == 'solver_failure'


def test_unknown_worker_blocks_next_round_after_restart():
    pending = proposal(state(), 'b')
    unknown = finish_feedback(pending, failure_reason='worker exit missing', failure_status='worker_result_unknown')
    assert unknown.incumbent_hash == 'a' and unknown.remaining_evaluations == 2
    restored = restore_feedback(asdict(unknown))
    with pytest.raises(ValueError, match='unknown worker result'):
        proposal(restored, 'c')


def test_invalid_candidate_or_protocol_cannot_be_silently_consumed():
    current = state(budget=1)
    with pytest.raises(ValueError, match='outside'):
        proposal(current, 'b', position=1.)
    pending = proposal(current, 'b')
    restored = restore_feedback(asdict(pending))
    assert restored.pending == pending.pending and restored.remaining_evaluations == 1
    invalid = asdict(pending)
    invalid['seen_hashes'] = ('wrong', 'extra')
    with pytest.raises(ValueError, match='history or identities'):
        restore_feedback(invalid)
    with pytest.raises(ValueError, match='contracts'):
        finish_feedback(pending, observation=observation('b', .35, .65, contract='other'))
    with pytest.raises(ValueError, match='identity'):
        finish_feedback(pending, observation=observation('wrong', .35, .65))
    assert pending.remaining_evaluations == 1 and pending.pending is not None
    finished = finish_feedback(pending, failure_reason='invalid solver record')
    with pytest.raises(ValueError, match='budget exhausted'):
        proposal(finished, 'c')


def test_strict_physical_policy_persists_and_ignores_predicted_gain():
    initial = start_feedback(incumbent_hash='a', incumbent_parameters=[0.],
        incumbent_observation=observation('a', .4, .6), target_frequency=.5,
        radius=.2, min_radius=.01, max_radius=.2, candidate_budget=2,
        acceptance_policy='strict_physical_improvement')
    pending = propose_feedback(initial, parameter_hash='b', parameters=[.1], predicted_center=.5,
                               predicted_candidate=.4, surrogate_contract_sha256='checkpoint')
    accepted = finish_feedback(restore_feedback(asdict(pending)), observation=observation('b', .35, .65))
    assert accepted.incumbent_hash == 'b' and accepted.radius == .2
    assert accepted.acceptance_policy == 'strict_physical_improvement'
    rejected = finish_feedback(proposal(accepted, 'c', position=.15), observation=observation('c', .4, .6))
    assert rejected.incumbent_hash == 'b' and rejected.radius == .1
    assert restore_feedback(asdict(rejected)) == rejected


def test_cached_repeat_consumes_round_without_a_new_physical_observation():
    initial = start_feedback(incumbent_hash='a', incumbent_parameters=[0.],
        incumbent_observation=observation('a', .4, .6), target_frequency=.5,
        radius=.2, min_radius=.025, max_radius=.2, candidate_budget=3,
        acceptance_policy='strict_physical_improvement')
    pending = propose_feedback(initial, parameter_hash='a', parameters=[0.], predicted_center=.2,
        predicted_candidate=.2, surrogate_contract_sha256='checkpoint', cached=True)
    done = finish_feedback(pending, observation=observation('a', .4, .6))
    assert done.radius == .1 and done.remaining_evaluations == 2
    assert done.history[-1]['cached'] is True and done.history[-1]['accepted'] is False
    assert restore_feedback(asdict(done)) == done
