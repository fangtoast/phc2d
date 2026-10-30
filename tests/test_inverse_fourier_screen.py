"""Tests for analytic candidate screening with a frozen predictor.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_candidate


class Predictor:
    def __init__(self, bands):
        self.bands = bands
        self.inputs = []

    def predict_tensor(self, x):
        self.inputs.append(x)
        return torch.tensor([self.bands], dtype=torch.float32)


def screen(predictor, bounds=(0, 1)):
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    return screen_fourier_candidate(s, predictor, raster_method='master_block_mean_v1',
                                    source_resolution=16, input_resolution=8,
                                    material_fraction_bounds=bounds, lower_band=1,
                                    target_frequency=0.5)


def test_nominal_prediction_returns_surrogate_only_gap():
    p = Predictor([[0.2, 0.8], [0.3, 0.7]])
    result = screen(p)
    assert result.status == 'surrogate_only'
    assert result.predicted_gap['contains_target']
    assert result.predicted_gap['lower_edge'] == pytest.approx(0.3)
    assert tuple(p.inputs[0].shape) == (1, 8, 8)


def test_illegal_fraction_does_not_call_predictor():
    p = Predictor([[0.2, 0.8]])
    assert screen(p, bounds=(0, 0)).status == 'material_fraction_rejected'
    assert not p.inputs


def test_band_order_violation_is_not_silently_sorted():
    result = screen(Predictor([[0.6, 0.5]]))
    assert result.status == 'band_order_violation'
    assert result.predicted_gap is None


def test_negative_unrelated_band_does_not_reject_target_pair():
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    result = screen_fourier_candidate(s, Predictor([[-0.001, 0.2, 0.8]]),
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=2, target_frequency=0.5)
    assert result.status == 'surrogate_only'
