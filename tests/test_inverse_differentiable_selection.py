"""Check hard selection and duplicate input handling.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_selection import rank_hard_trajectory


def test_selection_records_duplicate_and_full_prediction():
    class Predictor:
        def predict_tensor(self, inputs):
            return torch.tensor([[[0.4, 0.6], [0.41, 0.61]]], device=inputs.device)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    params = [source.parameters['modes'][0]['cos'], source.parameters['threshold']]
    history = [{'fourier_parameters': params}, {'fourier_parameters': params}]
    result = rank_hard_trajectory(source, Predictor(), history, coefficient_keys=[(0, 'cos')],
        max_mode_radius=1, raster_method='master_block_mean_v1', source_resolution=64,
        input_resolution=16, material_fraction_bounds=(0, 1), lower_band=1, target_frequency=0.5)
    assert result['selected_index'] == 0
    assert len(result['candidates'][0]['bands']) == 2
    assert result['candidates'][1]['status'] == 'duplicate_geometry'
    assert result['candidates'][1]['duplicate_of'] == 0


def test_duplicate_does_not_turn_a_rejected_candidate_into_an_eligible_one():
    class BadPredictor:
        def predict_tensor(self, inputs):
            return torch.tensor([[[0.6, 0.4]]], device=inputs.device)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    params = [source.parameters['modes'][0]['cos'], source.parameters['threshold']]
    result = rank_hard_trajectory(source, BadPredictor(), [{'fourier_parameters': params}]*2,
        coefficient_keys=[(0, 'cos')], max_mode_radius=1, raster_method='master_block_mean_v1',
        source_resolution=64, input_resolution=16, material_fraction_bounds=(0, 1),
        lower_band=1, target_frequency=0.5)
    assert result['candidates'][0]['status'] == 'band_order_violation'
    assert result['candidates'][1]['status'] == 'duplicate_geometry_rejected'
    assert result['selected_index'] is None
