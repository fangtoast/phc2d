"""Check locked structures become nominal physical requests without reselection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import hashlib
import json
import numpy as np
import pytest
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_handoff import build_physical_request
from phc2d.inverse.audit_perturb_refine import run_case


def test_handoff_retains_roles_and_deduplicates_only_master_geometry():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    grid = np.ascontiguousarray(source.rasterize((64, 64)), dtype=np.uint8)
    candidate = {'structure': source.to_mapping(), 'parameter_hash': source.parameter_hash,
                 'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest(), 'status': 'surrogate_only'}
    points = np.ascontiguousarray([[0., 0.], [0.5, 0.]], dtype=np.float32)
    lock = {'config_sha256': 'cfg', 'k_points': points.tolist(),
            'k_points_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
            'records': [{'label': str(i), 'candidate': candidate} for i in range(12)]}
    config = {'source_resolution': 64, 'lower_band': 1, 'target_frequency': 0.5,
              'solver_config': {}, 'physical_protocol': {'offsets': [0.0], 'raster_resolution': 64}}
    manifest, protocol = build_physical_request(lock, config, lock_sha256='lock', config_sha256='cfg')
    assert len(manifest['rows']) == 1 and len(manifest['role_coverage']) == 12
    assert all(row['case_index'] == 0 for row in manifest['role_coverage'])
    assert protocol['offsets'] == [0.0]
    with pytest.raises(ValueError, match='configuration|registered'):
        build_physical_request(lock, config, lock_sha256='lock', config_sha256='other')


def test_handoff_manifest_runs_existing_pixel_solver_path(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 16})
    grid = np.ascontiguousarray(source.rasterize((16, 16)), dtype=np.uint8)
    candidate = {'structure': source.to_mapping(), 'parameter_hash': source.parameter_hash,
                 'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest(), 'status': 'surrogate_only'}
    points = np.ascontiguousarray([[0., 0.], [0.25, 0.]], dtype=np.float32)
    lock = {'config_sha256': 'cfg', 'k_points': points.tolist(),
            'k_points_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
            'records': [{'label': str(i), 'candidate': candidate} for i in range(12)]}
    config = {'source_resolution': 16, 'lower_band': 1, 'target_frequency': 0.5,
              'solver_config': {'num_bands': 2, 'resolution': 16, 'polarization': 'TE',
                  'eigensolver_tolerance': 1e-7, 'epsilon_air': source.epsilon_air,
                  'epsilon_material': source.epsilon_material},
              'physical_protocol': {'offsets': [0.0], 'raster_resolution': 16,
                  'refinement': {'verification_resolution': 24, 'initial_spacing': 0.25,
                     'min_levels': 2, 'max_levels': 2, 'required_stable_levels': 2,
                     'edge_tolerance': 1, 'resolution_tolerance': 1e-6}}}
    manifest, protocol = build_physical_request(lock, config, lock_sha256='lock', config_sha256='cfg')
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp, mh = save('manifest.json', manifest)
    pp, ph = save('protocol.json', protocol)
    assert manifest['protocol_sha256'] == ph
    def solver(image, *, k_points, config):
        digest = hashlib.sha256(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest()
        return np.tile([0.2, 0.7], (len(k_points), 1)), {'grid_sha256': digest}
    output = run_case(mp, expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                      case_index=0, output_root=tmp_path/'results', solver=solver)
    result = json.loads((output/'results.json').read_text())
    assert result['aggregate']['status'] == 'numerically_stable'
    assert result['observations'][0]['provenance']['initial_solver']['grid_sha256'] == candidate['master_grid_sha256']
