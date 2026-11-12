"""Exercise a complete corrected surrogate proposal from physical observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.composite_feedback import composite_edge_observation
from phc2d.inverse.differentiable_feedback import propose_corrected_design


def test_corrected_proposal_fits_observations_and_screens_hard_structures():
    class Predictor:
        device = torch.device('cpu')
        checkpoint_sha256 = 'checkpoint'
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            low = 0.3 + 0.1 * value
            return torch.stack((low, low + 0.3), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    observations = [composite_edge_observation(parameter_hash=str(i), parameters=point,
        physical_edges=[0.36, 0.66], hard_edges=[0.35, 0.67], soft_exact_edges=[0.34, 0.68],
        soft_smooth_edges=[0.33, 0.69], comparison_contract_sha256='physical',
        surrogate_contract_sha256='model', numerical_gate_passed=True)
        for i, point in enumerate(([0., 0.], [0.1, 0.], [0., 0.1]))]
    config = {'center': [0., 0.], 'radius': 0.4, 'ridge': 1e-5, 'coefficient_keys': [[0, 'cos']],
        'scales': [0.2, 0.1], 'lower': [-0.5, -0.5], 'upper': [0.5, 0.5],
        'source_resolution': 64, 'input_resolution': 16, 'temperature': 0.2, 'lower_band': 1,
        'target_frequency': 0.5, 'beta': 100, 'learning_rate': 0.05, 'steps': 3,
        'max_mode_radius': 1, 'material_fraction_bounds': [0, 1],
        'surrogate_contract_sha256': 'model', 'comparison_contract_sha256': 'physical',
        'checkpoint_sha256': 'checkpoint'}
    result = propose_corrected_design(source, Predictor(), observations, config=config)
    assert result['status'] == 'proposal_ready'
    assert len(result['history']) == 4 and result['correction']['sample_count'] == 3
    assert result['hard_screening']['candidates'][result['selected_index']]['status'] == 'surrogate_only'
    assert result['prefit_prediction']['extra_forward_calls'] == 1
    assert result['prefit_prediction']['point'] == result['history'][result['selected_index']]['parameters']
    uncorrected = propose_corrected_design(source, Predictor(), observations,
                                            config={**config, 'apply_correction': False})
    assert uncorrected['correction']['enabled'] is False
    assert uncorrected['prefit_prediction']['corrected_smooth_edges'] == pytest.approx(
        uncorrected['prefit_prediction']['soft_smooth_edges'])
    de = propose_corrected_design(source, Predictor(), observations,
                                  config={**config, 'method': 'de', 'steps': 4, 'seed': 5})
    assert de['status'] == 'proposal_ready' and len(de['history']) == 5
    assert de['history'][0]['objective'] == pytest.approx(result['history'][0]['objective'])
    observations[1]['numerical_gate_passed'] = False
    with pytest.raises(ValueError, match='numerical gates'):
        propose_corrected_design(source, Predictor(), observations, config=config)
    observations[1]['numerical_gate_passed'] = True
    observations[1]['surrogate_contract_sha256'] = 'other-model'
    with pytest.raises(ValueError, match='contracts differ'):
        propose_corrected_design(source, Predictor(), observations, config=config)


def test_real_proposal_round_can_resume_from_frozen_artifacts(tmp_path):
    from dataclasses import asdict
    from phc2d.inverse.differentiable_feedback_campaign import run_feedback_arm_round
    from phc2d.inverse.feedback_state import start_feedback

    class Predictor:
        device = torch.device('cpu')
        checkpoint_sha256 = 'checkpoint'
        def predict_tensor(self, image):
            low = .3 + .1 * image.mean(dim=(1, 2))
            return torch.stack((low, low + .3), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32})
    bank = [composite_edge_observation(parameter_hash='a', parameters=[0., 0.],
        physical_edges=[.36, .66], hard_edges=[.35, .67], soft_exact_edges=[.34, .68],
        soft_smooth_edges=[.33, .69], comparison_contract_sha256='physical',
        surrogate_contract_sha256='model', numerical_gate_passed=True)]
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .36,
        'refined_upper_edge_frequency': .66, 'last_refinement_stable': True,
        'resolution_stable': True, 'adaptive_refinement_converged': True},
        'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0., 0.], incumbent_observation=physical,
        target_frequency=.5, radius=.2, min_radius=.025, max_radius=.2, candidate_budget=3,
        acceptance_policy='strict_physical_improvement')
    arm = {'seed_id': 'source_0008_initial_0_adam', 'arm': 'adam_residual', 'status': 'ready',
           'state': asdict(state), 'observations': bank, 'method': 'adam', 'apply_correction': True,
           'surrogate_contract_sha256': 'model'}
    base = {'seed': 42, 'coefficient_keys': [[0, 'cos']], 'scales': [.2, .1], 'lower': [-.5, -.5],
        'upper': [.5, .5], 'source_resolution': 64, 'input_resolution': 16, 'temperature': .2,
        'lower_band': 1, 'target_frequency': .5, 'beta': 100, 'learning_rate': .05,
        'max_mode_radius': 1, 'material_fraction_bounds': [0, 1], 'checkpoint_sha256': 'checkpoint'}
    kwargs = {'base': base, 'campaign': {'rounds': 3, 'evaluations_per_round': 20, 'ridge': 1e-5},
              'source_meta': {'source_id': 8, 'source_sha256': 'source'}, 'source': source,
              'predictor': Predictor(), 'round_index': 0, 'output': tmp_path/'round'}
    assert run_feedback_arm_round(arm, **kwargs)['status'] == 'frozen'
    assert run_feedback_arm_round(arm, **kwargs)['status'] == 'frozen_reused'
