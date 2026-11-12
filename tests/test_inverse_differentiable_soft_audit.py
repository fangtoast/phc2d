"""Check post-selection soft/hard diagnostics preserve candidate identity.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.differentiable_soft_audit import audit_locked_soft


def test_soft_audit_keeps_locked_geometry_and_records_full_spectrum():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1*value, 0.7 + 0.1*value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    params = [source.parameters['modes'][0]['cos'], source.parameters['threshold']]
    ranked = rank_hard_trajectory(source, Predictor(), [{'fourier_parameters': params}]*2,
        coefficient_keys=[(0, 'cos')], max_mode_radius=1, raster_method='master_block_mean_v1',
        source_resolution=64, input_resolution=16, material_fraction_bounds=(0, 1),
        lower_band=1, target_frequency=0.5)
    lock = {'records': [{'label': 'center', 'candidate': ranked['candidates'][0]},
                        {'label': 'duplicate', 'candidate': ranked['candidates'][1]},
                        {'label': 'missing', 'candidate': None}]}
    config = {'coefficient_keys': [[0, 'cos']], 'source_resolution': 64, 'input_resolution': 16,
              'temperature': 0.2, 'lower_band': 1, 'target_frequency': 0.5}
    report = audit_locked_soft(lock, Predictor(), config)
    assert [row['status'] for row in report['roles']] == ['computed', 'computed', 'not_eligible']
    assert report['roles'][0]['soft_exact_margin'] == pytest.approx(report['roles'][1]['soft_exact_margin'])
    assert len(report['roles'][0]['soft_bands']) == 1
    broken = copy.deepcopy(lock)
    broken['records'][0]['candidate']['master_grid_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='geometry'):
        audit_locked_soft(broken, Predictor(), config)
