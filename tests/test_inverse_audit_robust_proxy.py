"""Tests for fixed-candidate sampled robust proxy diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_robust_proxy import run, validate_paired_proxy, validate_frozen_proxy_predictions
from phc2d.inverse.perturbation_screen import compare_predictor_devices


def test_device_parity_checks_full_spectra_identity_and_nonfinite():
    class Predictor:
        input_shape = (2, 2)
        output_shape = (1, 2)
        checkpoint_sha256 = 'checkpoint'
        k_points_sha256 = 'points'
        def __init__(self, value):
            self.value = value
        def predict_tensor(self, x):
            return torch.tensor([[[.2, self.value]]] * len(x))
    grids = (np.zeros((4, 4), dtype=np.uint8),)
    reference = Predictor(.7)
    passing = compare_predictor_devices(grids, Predictor(.700001), reference,
                                        input_resolution=2, absolute_tolerance=1e-5)
    assert passing['passed'] and passing['max_absolute_difference'] > 0
    assert not compare_predictor_devices(grids, Predictor(.8), reference,
                                         input_resolution=2, absolute_tolerance=1e-5)['passed']
    invalid = compare_predictor_devices(grids, Predictor(float('nan')), reference,
                                        input_resolution=2, absolute_tolerance=1e-5)
    assert not invalid['passed'] and invalid['prediction_nonfinite_count'] == 1
    assert invalid['max_absolute_difference'] is None
    changed = Predictor(.7)
    changed.checkpoint_sha256 = 'other'
    with pytest.raises(ValueError, match='identity'):
        compare_predictor_devices(grids, changed, reference, input_resolution=2, absolute_tolerance=1e-5)
    with pytest.raises(ValueError, match='tolerance'):
        compare_predictor_devices(grids, reference, reference, input_resolution=2, absolute_tolerance=float('nan'))


def test_robust_proxy_keeps_offset_grid_and_candidate_identities(tmp_path, monkeypatch):
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    grid = np.ascontiguousarray(structure.rasterize((16, 16)), dtype=np.uint8)
    points = [[0., 0.]]
    k_hash = hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()
    manifest = {'rows': [{'label': 'anchor', 'source_index': 3, 'structure': structure.to_mapping(),
                          'parameter_hash': structure.parameter_hash,
                          'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}],
                'k_points': points, 'lower_band': 1, 'target_frequency': .5}
    protocol = {'raster_resolution': 16, 'offsets': [-.0625, 0., .0625]}
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp, mh = save('manifest.json', manifest)
    pp, ph = save('protocol.json', protocol)
    class Predictor:
        input_shape = (2, 2)
        output_shape = (1, 2)
        checkpoint_sha256 = 'test'
        k_points_sha256 = k_hash
        def __init__(self, *args, **kwargs):
            self.device = kwargs.get('device')
        def predict_tensor(self, x):
            return torch.tensor([[[.2, .9 if self.device == 'bad' else .7]]] * len(x))
    monkeypatch.setattr('phc2d.inverse.audit_robust_proxy.FrozenBandAdapter', Predictor)
    output = tmp_path / 'scores.json'
    run(mp, manifest_sha256=mh, protocol_path=pp, protocol_sha256=ph,
        checkpoint='unused', checkpoint_sha256='test', device='cpu', output=output)
    result = json.loads(output.read_text())
    assert result['rows'][0]['predicted_margin'] == pytest.approx(.8)
    assert result['rows'][0]['predicted_gap']['lower_edge'] == pytest.approx(.2)
    assert len(result['rows'][0]['predicted_bands']) == 3
    assert len(result['rows'][0]['grid_sha256']) == 3
    assert result['rows'][0]['grid_sha256'][1] == manifest['rows'][0]['master_grid_sha256']
    frozen_args = dict(manifest_bytes=mp.read_bytes(), manifest_sha256=mh, protocol_bytes=pp.read_bytes(),
                       protocol_sha256=ph, checkpoint_sha256='test', k_points_sha256=k_hash,
                       checkpoint_output_shape=(1, 2))
    verified = validate_frozen_proxy_predictions(proxy=result, **frozen_args)
    assert verified['candidate_count'] == 1 and verified['status_counts'] == {'surrogate_only': 1}
    parity_output = tmp_path / 'parity.json'
    run(mp, manifest_sha256=mh, protocol_path=pp, protocol_sha256=ph,
        checkpoint='unused', checkpoint_sha256='test', device='cpu', output=parity_output,
        parity_reference_device='cpu', parity_absolute_tolerance=1e-6)
    assert json.loads(parity_output.read_text())['device_parity']['passed']
    rejected = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='device prediction parity failed'):
        run(mp, manifest_sha256=mh, protocol_path=pp, protocol_sha256=ph,
            checkpoint='unused', checkpoint_sha256='test', device='bad', output=rejected,
            parity_reference_device='cpu', parity_absolute_tolerance=1e-6)
    assert not rejected.exists()
    changed = json.loads(json.dumps(result))
    changed['rows'][0]['predicted_margin'] += .01
    with pytest.raises(ValueError, match='does not reproduce'):
        validate_frozen_proxy_predictions(proxy=changed, **frozen_args)
    changed = json.loads(json.dumps(result))
    changed['rows'][0]['grid_sha256'][0] = 'wrong'
    with pytest.raises(ValueError, match='grid identity'):
        validate_frozen_proxy_predictions(proxy=changed, **frozen_args)
    changed = json.loads(json.dumps(result))
    changed['rows'][0].update(status='band_order_violation', predicted_gap=None, predicted_margin=None,
                              predicted_bands=[[[.7, .2]]] * 3)
    assert validate_frozen_proxy_predictions(proxy=changed, **frozen_args)['status_counts'] == {'band_order_violation': 1}
    changed['rows'][0]['predicted_bands'] = [[[.2, .7]]] * 3
    with pytest.raises(ValueError, match='band-order failure'):
        validate_frozen_proxy_predictions(proxy=changed, **frozen_args)
    changed['rows'][0].update(status='negative_frequency', predicted_bands=[[[-.1, .7]]] * 3)
    assert validate_frozen_proxy_predictions(proxy=changed, **frozen_args)['status_counts'] == {'negative_frequency': 1}
    changed['rows'][0]['predicted_bands'] = [[[.2, .7]]] * 3
    with pytest.raises(ValueError, match='negative-frequency failure'):
        validate_frozen_proxy_predictions(proxy=changed, **frozen_args)
    changed['rows'][0].update(status='nonfinite_prediction', predicted_bands=None)
    assert validate_frozen_proxy_predictions(proxy=changed, **frozen_args)['nonfinite_failure_reason_verifiable'] is False
    row = result['rows'][0]
    physical = [{'source_index': row['source_index'], 'label': row['label'], 'parameter_hash': row['parameter_hash'],
                 'manifest_sha256': mh, 'protocol_sha256': ph,
                 'observations': [{'offset': offset, 'provenance': {'evaluated_grid_sha256': digest}}
                                  for offset, digest in zip(protocol['offsets'], row['grid_sha256'])]}]
    args = dict(manifest_sha256=mh, protocol_sha256=ph, checkpoint_sha256='test', k_points_sha256=k_hash)
    assert len(validate_paired_proxy(result, physical, **args)) == 1
    bad = dict(result, rows=[dict(row, status='band_order_violation', predicted_gap=None, predicted_margin=None)])
    with pytest.raises(ValueError, match='invalid score'):
        validate_paired_proxy(bad, physical, **args)
    assert validate_paired_proxy(bad, physical, allow_unscored=True, **args)[0]['predicted_margin'] is None
    invalid = dict(result, rows=[dict(row, status='band_order_violation', predicted_gap=None, predicted_margin=.1)])
    with pytest.raises(ValueError, match='invalid score'):
        validate_paired_proxy(invalid, physical, allow_unscored=True, **args)
    physical[0]['observations'][0]['provenance']['evaluated_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='grid identity'):
        validate_paired_proxy(result, physical, **args)
