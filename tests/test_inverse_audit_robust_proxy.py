"""Tests for fixed-candidate sampled robust proxy diagnostics."""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_robust_proxy import run, validate_paired_proxy


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
        k_points_sha256 = k_hash
        def __init__(self, *args, **kwargs):
            pass
        def predict_tensor(self, x):
            return torch.tensor([[[.2, .7]]] * len(x))
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
    row = result['rows'][0]
    physical = [{'source_index': row['source_index'], 'label': row['label'], 'parameter_hash': row['parameter_hash'],
                 'manifest_sha256': mh, 'protocol_sha256': ph,
                 'observations': [{'offset': offset, 'provenance': {'evaluated_grid_sha256': digest}}
                                  for offset, digest in zip(protocol['offsets'], row['grid_sha256'])]}]
    args = dict(manifest_sha256=mh, protocol_sha256=ph, checkpoint_sha256='test', k_points_sha256=k_hash)
    assert len(validate_paired_proxy(result, physical, **args)) == 1
    physical[0]['observations'][0]['provenance']['evaluated_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='grid identity'):
        validate_paired_proxy(result, physical, **args)
