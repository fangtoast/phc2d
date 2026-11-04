"""Freeze feedback incumbents from authenticated observations and candidate locks."""
from dataclasses import asdict

import pytest

from phc2d.inverse.differentiable_feedback_final import freeze_final_feedback
from phc2d.inverse.feedback_state import finish_feedback, propose_feedback, start_feedback


def test_final_feedback_resolves_incumbent_and_does_not_reselect():
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
        'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
        'resolution_stable': True, 'adaptive_refinement_converged': True},
        'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0., 0.], incumbent_observation=physical,
        target_frequency=.5, radius=.2, min_radius=.025, max_radius=.2, candidate_budget=3,
        acceptance_policy='strict_physical_improvement')
    for _ in range(3):
        state = propose_feedback(state, parameter_hash='a', parameters=[0., 0.], predicted_center=.1,
                                 predicted_candidate=.1, surrogate_contract_sha256='model', cached=True)
        state = finish_feedback(state, observation=physical)
    arm = {'seed_id': 'source_0008_initial_0_adam', 'arm': 'adam_residual', 'status': 'completed',
           'state': asdict(state), 'physical_observations': {'a': physical}}
    arms = {'scope': 'matched_feedback_arms_initialized', 'completed_rounds': 3, 'selection_lock_sha256': 'seed',
            'previous_round_lock_sha256': 'r2', 'arms': [arm]}
    design = {'scope': 'preregistered_surrogate_design', 'base_config_sha256': 'base', 'sources': [{'source_id': 8}]}
    design_lock = {'manifest_sha256': 'design', 'k_points': [[0., 0.]]}
    candidate = {'status': 'surrogate_only', 'parameter_hash': 'a', 'master_grid_sha256': 'grid',
                 'bands': [[.4, .6]], 'predicted_gap': {'target_margin': .4}}
    locks = [{'purpose': 'feedback_initial_stencil', 'records': [{'trial_id': arm['seed_id'],
        'point': [0., 0.], 'candidate': candidate}]}] + [{'round_index': i, 'records': []} for i in range(3)]
    args = dict(arms=arms, design=design, design_lock=design_lock, prior_locks=locks, design_sha256='design',
                arms_sha256='arms', prior_lock_sha256=['seed', 'r0', 'r1', 'r2'])
    synthetic, locked = freeze_final_feedback(**args)
    assert synthetic['trials'][0]['trial_id'] == arm['seed_id'] + '_adam_residual'
    assert locked['records'][0]['candidate']['parameter_hash'] == 'a'
    with pytest.raises(ValueError, match='identity'):
        freeze_final_feedback(**{**args, 'prior_lock_sha256': ['seed', 'r0', 'r1', 'wrong']})
