"""Test that new physical data cannot alter a frozen prefit prediction."""
import copy
import hashlib
import json
import pytest

from phc2d.inverse.differentiable_feedback_round import (complete_round, freeze_round,
    save_round_completion, save_round_request)
from phc2d.inverse.feedback_state import start_feedback


def test_feedback_round_freezes_predictions_before_observation(tmp_path):
    def physical(identity, lower, upper):
        return {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': lower,
                        'refined_upper_edge_frequency': upper, 'last_refinement_stable': True,
                        'resolution_stable': True, 'adaptive_refinement_converged': True},
                'provenance': {'parameter_hash': identity, 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0.0, 0.0],
        incumbent_observation=physical('a', .4, .6), target_frequency=.5, radius=.2,
        min_radius=.025, max_radius=.2, candidate_budget=3, acceptance_policy='strict_physical_improvement')
    bank = [{'parameter_hash': 'a'}]
    digest = hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    prediction = {'point': [0.1, 0.0], 'hard_edges': [.38, .62], 'soft_exact_edges': [.37, .63],
                  'soft_smooth_edges': [.36, .64], 'corrected_smooth_edges': [.35, .65]}
    candidate = {'status': 'surrogate_only', 'parameter_hash': 'b', 'index': 19}
    history = [{'parameters': [0.0, 0.0], 'objective': .1}] * 19 + [
        {'parameters': [0.1, 0.0], 'objective': .2}]
    proposal = {'status': 'proposal_ready', 'observation_sha256': digest, 'selected_index': 19,
                'history': history, 'hard_screening': {'candidates': [candidate] * 20},
                'prefit_prediction': prediction, 'correction': {'center': [0.0, 0.0],
                'radius': .2, 'surrogate_contract_sha256': 'model', 'comparison_contract_sha256': 'physical'}}
    frozen = freeze_round(state, proposal, bank)
    composite = {'parameter_hash': 'b', 'parameters': [.1, 0.0], 'numerical_gate_passed': True,
                 'surrogate_contract_sha256': 'model', 'comparison_contract_sha256': 'physical',
                 'hard_edges': [.38, .62], 'soft_exact_edges': [.37, .63],
                 'predicted_edges': [.36, .64], 'physical_edges': [.34, .66]}
    completed = complete_round(frozen, physical=physical('b', .34, .66), composite=composite)
    assert completed['accepted'] is True
    assert completed['prefit_edge_error'] == pytest.approx([.01, -.01])
    assert completed['state']['remaining_evaluations'] == 2
    directory = tmp_path/'round_0'
    save_round_request(directory, frozen=frozen, observations=bank)
    stored = save_round_completion(directory, physical=physical('b', .34, .66), composite=composite)
    assert stored == save_round_completion(directory, physical=physical('b', .34, .66), composite=composite)
    assert len(stored['next_observations']) == 2
    bad = copy.deepcopy(composite); bad['predicted_edges'][0] += .01
    with pytest.raises(ValueError, match='prefit'):
        complete_round(frozen, physical=physical('b', .34, .66), composite=bad)
    bad_frozen = copy.deepcopy(frozen); bad_frozen['prefit_prediction']['corrected_smooth_edges'][0] += .01
    with pytest.raises(ValueError, match='integrity'):
        complete_round(bad_frozen, physical=physical('b', .34, .66), composite=composite)
    with pytest.raises(ValueError, match='observation identity'):
        freeze_round(state, proposal, bank + [{'parameter_hash': 'future'}])
