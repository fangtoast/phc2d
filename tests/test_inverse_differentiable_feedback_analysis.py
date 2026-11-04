"""Check paired physics-feedback metrics use only prefit predictions and locked baselines."""
import copy
from dataclasses import asdict

import pytest

from phc2d.inverse.differentiable_feedback_analysis import analyze_feedback
from phc2d.inverse.feedback_state import finish_feedback, propose_feedback, start_feedback


def test_feedback_analysis_reproduces_pairing_interaction_and_prefit_errors():
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
        'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
        'resolution_stable': True, 'adaptive_refinement_converged': True},
        'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0., 0.], incumbent_observation=physical,
        target_frequency=.5, radius=.2, min_radius=.025, max_radius=.2, candidate_budget=3,
        acceptance_policy='strict_physical_improvement')
    records = []
    for _ in range(3):
        state = propose_feedback(state, parameter_hash='a', parameters=[0., 0.], predicted_center=.1,
                                 predicted_candidate=.1, surrogate_contract_sha256='model', cached=True)
        state = finish_feedback(state, observation=physical)
        records.append({'completion': {'state': asdict(state), 'proposal_sha256': 'proposal', 'accepted': False,
            'cached': True, 'prefit_edge_error': [.01, -.01]}, 'composite': {'physical_edges': [.4, .6]},
            'request': {'frozen': {'proposal_sha256': 'proposal',
                'prefit_prediction': {'soft_smooth_edges': [.42, .58], 'corrected_smooth_edges': [.41, .59]}}}})
    seed = 'source_0008_initial_0_adam'
    names = ('adam_trajectory', 'adam_residual', 'de_residual')
    arms = {'scope': 'matched_feedback_arms_initialized', 'completed_rounds': 3, 'arms': [
        {'seed_id': seed, 'arm': name, 'status': 'completed', 'state': asdict(state),
         'physical_observations': {'a': physical}} for name in names]}
    baseline = {'roles': [{'label': seed+'/top1', 'status': 'numerically_stable', 'physical_refined_target_margin': .3},
        {'label': 'source_0008_initial_0_de/top1', 'status': 'numerically_stable', 'physical_refined_target_margin': .2}]}
    report = analyze_feedback(arms, baseline, {f'{seed}_{name}': records for name in names})
    assert report['complete_2x2_count'] == 1
    assert report['seeds'][0]['interaction'] == pytest.approx(-.1)
    round0 = report['seeds'][0]['arms']['adam_residual']['rounds'][0]
    assert round0['raw_edge_error'] == pytest.approx([.02, -.02])
    assert round0['corrected_edge_error'] == pytest.approx([.01, -.01])
    bad = copy.deepcopy(records); bad[0]['completion']['prefit_edge_error'][0] += .1
    with pytest.raises(ValueError, match='prefit'):
        analyze_feedback(arms, baseline, {f'{seed}_{name}': bad for name in names})
