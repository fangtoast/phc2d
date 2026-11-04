"""Verify matched feedback arms share the same frozen seed evidence."""
import copy
import pytest
from phc2d.inverse.differentiable_feedback_campaign import (initialize_feedback_arms,
    prepare_round_configuration, run_feedback_arm_round)


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


def test_round_configuration_uses_current_state_and_shared_budget():
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
                'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
                'resolution_stable': True, 'adaptive_refinement_converged': True},
                'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    from dataclasses import asdict
    from phc2d.inverse.feedback_state import start_feedback
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[.15, 0],
        incumbent_observation=physical, target_frequency=.5, radius=.2, min_radius=.025,
        max_radius=.2, candidate_budget=3, acceptance_policy='strict_physical_improvement')
    arm = {'status': 'ready', 'state': asdict(state), 'method': 'adam',
           'apply_correction': False, 'seed_id': 'source_0008_initial_1_adam',
           'surrogate_contract_sha256': 'model', 'observations': [
               {'comparison_contract_sha256': 'physical', 'surrogate_contract_sha256': 'model'}]}
    campaign = {'rounds': 3, 'evaluations_per_round': 20, 'ridge': 1e-6}
    source = {'source_id': 8, 'source_sha256': 'source'}
    config = prepare_round_configuration(arm, base={'seed': 42}, campaign=campaign,
        source=source, round_index=0, observations_sha256='observations')
    assert config['center'] == [.15, 0] and config['radius'] == .2
    assert config['steps'] == 19 and config['seed'] == 42 and config['source_sha256'] == 'source'
    with pytest.raises(ValueError, match='round state'):
        prepare_round_configuration(arm, base={'seed': 42}, campaign=campaign,
            source=source, round_index=1, observations_sha256='observations')
    with pytest.raises(ValueError, match='source identity'):
        prepare_round_configuration(arm, base={'seed': 42}, campaign=campaign,
            source={'source_id': 491}, round_index=0, observations_sha256='observations')


def test_round_runner_freezes_proposal_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from phc2d.inverse.feedback_state import start_feedback
    import phc2d.inverse.differentiable_feedback as proposal_module
    import phc2d.inverse.differentiable_feedback_round as round_module
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
                'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
                'resolution_stable': True, 'adaptive_refinement_converged': True},
                'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0, 0],
        incumbent_observation=physical, target_frequency=.5, radius=.2, min_radius=.025,
        max_radius=.2, candidate_budget=3, acceptance_policy='strict_physical_improvement')
    arm = {'status': 'ready', 'state': asdict(state), 'method': 'adam', 'apply_correction': False,
           'seed_id': 'source_0008_initial_0_adam', 'surrogate_contract_sha256': 'model',
           'observations': [{'comparison_contract_sha256': 'physical', 'surrogate_contract_sha256': 'model'}]}
    proposal = {'status': 'proposal_ready', 'history': [1, 2]}
    monkeypatch.setattr(proposal_module, 'propose_corrected_design', lambda *a, **k: proposal)
    monkeypatch.setattr(round_module, 'freeze_round', lambda *a: {'frozen_sha256': 'f', 'cached_observation': None})
    def save(path, **kwargs):
        path.mkdir()
        (path / 'request.json').write_text(json.dumps(kwargs))
    monkeypatch.setattr(round_module, 'save_round_request', save)
    output = tmp_path / 'round'
    kwargs = dict(base={'seed': 42}, campaign={'rounds': 3, 'evaluations_per_round': 20, 'ridge': 1e-6},
        source_meta={'source_id': 8, 'source_sha256': 'source'}, source=None, predictor=None, round_index=0, output=output)
    assert run_feedback_arm_round(arm, **kwargs)['status'] == 'frozen'
    assert json.loads((output / 'proposal.json').read_text()) == proposal
    assert json.loads((output / 'config.json').read_text())['steps'] == 19
    assert run_feedback_arm_round(arm, **kwargs)['status'] == 'frozen_reused'
    (output / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='identity mismatch'):
        run_feedback_arm_round(arm, **kwargs)
