"""Directional derivative diagnostic tests."""

import pytest
import hashlib
import json

import numpy as np

torch = pytest.importorskip('torch')

from phc2d.inverse.input_gradient_audit import audit_input_gradient
from phc2d.inverse.audit_gradient_cases import run
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary


class LinearPredictor:
    device = torch.device('cpu')

    def predict_tensor(self, inputs):
        v = inputs.reshape(inputs.shape[0], -1).sum(dim=1)
        return torch.stack((.2 + .01 * v, .8 + .02 * v), dim=1).unsqueeze(1)


def test_input_gradient_matches_directional_finite_difference():
    result = audit_input_gradient(LinearPredictor(), torch.ones(1, 2, 2), lower_band=1,
                                  target_frequency=.5, steps=[.01, .1], seed=42)
    assert result['status'] == 'continuous_input_diagnostic_only'
    assert result['gradient_norm'] > 0
    assert max(x['absolute_error'] for x in result['finite_differences']) < 1e-5


def test_input_gradient_rejects_nonpositive_step():
    with pytest.raises(ValueError, match='positive'):
        audit_input_gradient(LinearPredictor(), torch.ones(1, 2, 2), lower_band=1,
                             target_frequency=.5, steps=[0], seed=42)


def test_manifest_runner_checks_raster_and_preserves_identity(tmp_path, monkeypatch):
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    grid = np.ascontiguousarray(s.rasterize((16, 16)), dtype=np.uint8)
    points = [[0., 0.]]
    k_hash = hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()
    manifest = {'rows': [{'label': 'anchor', 'source_index': 3, 'structure': s.to_mapping(),
                          'parameter_hash': s.parameter_hash,
                          'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}],
                'k_points': points, 'raster_resolution': 16, 'lower_band': 1, 'target_frequency': .5}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf8')
    class Predictor(LinearPredictor):
        input_shape = (2, 2)
        k_points_sha256 = k_hash
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr('phc2d.inverse.audit_gradient_cases.FrozenBandAdapter', Predictor)
    output = tmp_path / 'output.json'
    run(path, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checkpoint='unused',
        checkpoint_sha256='sha', device='cpu', output=output)
    result = json.loads(output.read_text())
    assert result['anchor_count'] == 1
    assert result['rows'][0]['parameter_hash'] == s.parameter_hash
