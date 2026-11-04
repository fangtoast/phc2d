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
