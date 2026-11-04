"""Check preregistered neighbor-center direction diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_expansion import combine_centers, evaluate_center


def fixture():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    config = {'coefficient_keys': [[0, 'cos']], 'scales': [0.2, 0.1], 'lower': [-0.5, -0.5],
        'upper': [0.5, 0.5], 'source_resolution': 64, 'input_resolution': 16, 'temperature': 0.2,
        'lower_band': 1, 'target_frequency': 0.5, 'beta': 100, 'seed': 10, 'max_mode_radius': 1,
        'material_fraction_bounds': [0, 1], 'finite_difference_steps': [0.01, 0.003],
        'direction_radii': [0.05, 0.1], 'gradient_fd_absolute_tolerance': 2e-5,
        'gradient_fd_relative_tolerance': 0.05, 'gradient_fd_minimum_steps': 2}
    extension = {'centers': [[0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05]],
                 'seed_offset': 100}
    return source, config, extension


def test_neighbor_center_records_nine_fixed_roles_and_fd():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1*value, 0.7 + 0.1*value), dim=1).unsqueeze(1)
    source, config, extension = fixture()
    result = evaluate_center(source, Predictor(), base_config=config, extension=extension, center_index=2)
    assert result['center'] == [0.0, 0.05]
    assert len(result['records']) == 9 and result['fd_passed']
    assert result['records'][0]['label'] == 'center'
    assert sum(row['status'] == 'surrogate_only' for row in result['records']) > 0
    with pytest.raises(ValueError, match='center index'):
        evaluate_center(source, Predictor(), base_config=config, extension=extension, center_index=4)


def test_zero_gradient_is_recorded_without_replacing_direction_roles():
    class FlatPredictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))*0
            return torch.stack((0.3 + value, 0.7 + value), dim=1).unsqueeze(1)
    source, config, extension = fixture()
    result = evaluate_center(source, FlatPredictor(), base_config=config, extension=extension, center_index=0)
    assert result['gradient_status'] == 'zero_or_nonfinite'
    assert len(result['records']) == 9
    assert sum(row['status'] == 'direction_unavailable' for row in result['records']) == 4
    assert result['fd_passed'] is False


def test_extension_lock_preserves_all_four_centers_and_missing_directions():
    _, _, extension = fixture()
    extension['base_config_sha256'] = 'base'
    labels = ['center', 'gradient_0.05', 'negative_gradient_0.05', 'random_0.05',
              'negative_random_0.05', 'gradient_0.1', 'negative_gradient_0.1',
              'random_0.1', 'negative_random_0.1']
    identity = {'base_config_sha256': 'base', 'extension_config_sha256': 'extension',
        'source_sha256': 'source', 'source_parameter_hash': 'parameter',
        'checkpoint_sha256': 'model', 'k_points_sha256': 'k', 'k_points': [[0, 0]], 'code_commit': 'commit'}
    shards = [dict(identity, center_index=i, center=extension['centers'][i], fd_passed=(i != 1),
                   records=[{'label': label, 'candidate': None, 'status': 'direction_unavailable'} for label in labels])
              for i in range(4)]
    lock = combine_centers(shards[::-1], extension)
    assert len(lock['records']) == 36 and lock['center_fd_passed'] == [True, False, True, True]
    assert lock['records'][9]['label'] == 'center_1/center'
    shards[3]['checkpoint_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='differ'):
        combine_centers(shards, extension)
