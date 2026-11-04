"""Exercise the registered feedback stencil and boundary coverage."""
import pytest
torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_feedback_seed import prepare_feedback_stencil
from phc2d.inverse.differentiable_robust_stencil import predict_stencil_scenarios


def test_feedback_stencil_retains_all_roles_and_bounds():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    base = {'coefficient_keys': [[0, 'cos']], 'scales': [0.2, 0.1], 'lower': [-0.5, -0.5],
            'upper': [0.5, 0.5], 'max_mode_radius': 1.0, 'source_resolution': 64,
            'input_resolution': 16, 'material_fraction_bounds': [0, 1], 'lower_band': 1,
            'target_frequency': .5, 'temperature': .2, 'beta': 100.0}
    cfg = {'initial_offsets': [[0, 0], [.1, 0], [-.1, 0], [0, .1], [0, -.1]]}
    result = prepare_feedback_stencil(source, Predictor(), base=base, campaign=cfg, initial=[.5, 0])
    assert len(result['records']) == 5
    assert result['records'][1]['status'] == 'outside_parameter_bounds'
    assert result['records'][0]['point'] == [.5, 0]
    assert result['soft_audit']['roles'][1]['status'] == 'not_eligible'
    edges = result['records'][0]['surrogate_edges']
    assert edges['soft_smooth_edges'][0] >= edges['soft_exact_edges'][0]
    assert edges['soft_smooth_edges'][1] <= edges['soft_exact_edges'][1]
    scenarios = predict_stencil_scenarios(result, source, Predictor(), base=base, offsets=[-1/64, 0., 1/64])
    assert len(scenarios['records']) == 5 and scenarios['records'][1]['status'] == 'outside_parameter_bounds'
    row = scenarios['records'][0]
    assert row['status'] == 'computed' and len(row['hard']['bands']) == 3
    assert row['soft_smooth_edges'][1] == pytest.approx(edges['soft_smooth_edges'], abs=1e-6)
    assert row['hard_edges'][1] == pytest.approx(edges['hard_edges'], abs=1e-6)
