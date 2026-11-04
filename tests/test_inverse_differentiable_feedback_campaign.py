"""Verify matched feedback arms share the same frozen seed evidence."""
import copy
import pytest
from phc2d.inverse.differentiable_feedback_campaign import initialize_feedback_arms


def test_initialize_three_arms_with_shared_samples_and_budget():
    rows = []
    for i in range(5):
        point = [i / 100, 0.0]
        composite = {'parameter_hash': str(i), 'parameters': point, 'numerical_gate_passed': True, 'physical_edges': [.4, .6],
                     'comparison_contract_sha256': 'physical', 'surrogate_contract_sha256': 'model'}
        physical = {'status': 'numerically_stable', 'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
                    'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
                    'resolution_stable': True, 'adaptive_refinement_converged': True},
                    'provenance': {'parameter_hash': str(i), 'comparison_contract_sha256': 'physical'}}
        rows.append({'trial_id': 'seed', 'role': f'offset_{i}', 'status': 'observed',
                     'point': point, 'composite': composite, 'physical': physical})
    observations = {'scope': 'feedback_initial_observations', 'selection_lock_sha256': 'lock',
                    'physical_manifest_sha256': 'manifest', 'surrogate_contract_sha256': 'model', 'records': rows}
    campaign = {'rounds': 3, 'evaluations_per_round': 20, 'max_observation_roles': 8,
                'acceptance': 'strict_physical_improvement', 'initial_trust_radius': .2,
                'rejection_radius_factor': .5, 'arms': ['adam_trajectory', 'adam_residual', 'de_residual']}
    result = initialize_feedback_arms(observations, campaign=campaign, target_frequency=.5)
    assert len(result['arms']) == 3
    assert [arm['apply_correction'] for arm in result['arms']] == [False, True, True]
    assert all(len(arm['observations']) == 5 and arm['state']['remaining_evaluations'] == 3 for arm in result['arms'])
    bad = copy.deepcopy(observations); bad['records'][1]['composite']['surrogate_contract_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        initialize_feedback_arms(bad, campaign=campaign, target_frequency=.5)
