"""Exercise cross-environment comparison, identity and missing cases."""
import copy
import hashlib
import numpy as np
import pytest
from phc2d.inverse.differentiable_crosscheck import compare_crosscheck
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids


def test_crosscheck_rejects_mismatch_and_reports_missing():
    manifest = {'protocol_sha256': 'protocol', 'k_points': [[0, 0], [0.5, 0]],
                'rows': [{'parameter_hash': 'params', 'master_grid_sha256': 'grid'}],
                'role_coverage': [{'label': 'center', 'case_index': 0}]}
    result = {'case_index': 0, 'parameter_hash': 'params', 'manifest_sha256': 'manifest',
              'protocol_sha256': 'protocol', 'elapsed_s': 1,
              'aggregate': {'status': 'numerically_stable'},
              'observations': [{'provenance': {'evaluated_grid_sha256': 'grid'},
                                'arrays': {'k_points': [[0, 0], [0.5, 0]],
                                           'bands': [[0.1, 0.2], [0.3, 0.4]]}}]}
    args = dict(manifest=manifest, roles=['center'], reference={0: result},
                tolerance=1e-6, manifest_sha256='manifest')
    assert compare_crosscheck(**args, replay={0: result})['all_passed']
    assert compare_crosscheck(**args, replay={})['roles'][0]['status'] == 'missing'
    other = copy.deepcopy(result)
    other['observations'][0]['arrays']['bands'][0][0] += 2e-6
    assert not compare_crosscheck(**args, replay={0: other})['all_passed']
    other['manifest_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        compare_crosscheck(**args, replay={0: other})


def test_crosscheck_compares_each_registered_manufacturing_scenario():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32})
    offsets = [-1/64, 0., 1/64]
    grids = periodic_perturbation_grids(source, resolution=64, offsets=tuple(offsets))
    hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
    manifest = {'protocol_sha256': 'protocol', 'k_points': [[0., 0.]], 'raster_resolution': 64,
        'rows': [{'parameter_hash': source.parameter_hash, 'master_grid_sha256': hashes[1],
                  'structure': source.to_mapping()}], 'role_coverage': [{'label': 'top1', 'case_index': 0}]}
    result = {'case_index': 0, 'parameter_hash': source.parameter_hash, 'manifest_sha256': 'manifest',
        'protocol_sha256': 'protocol', 'elapsed_s': 1., 'aggregate': {'status': 'numerically_stable'},
        'observations': [{'offset': offset, 'provenance': {'evaluated_grid_sha256': hashes[i]},
            'arrays': {'k_points': [[0., 0.]], 'bands': [[.4, .6]]},
            'row': {'last_refinement_stable': True, 'resolution_stable': True, 'adaptive_refinement_converged': True}}
            for i, offset in enumerate(offsets)]}
    args = dict(manifest=manifest, roles=['top1'], reference={0: result}, replay={0: result},
                tolerance=1e-6, manifest_sha256='manifest')
    output = compare_crosscheck(**args)
    assert output['all_passed'] and output['roles'][0]['scenario_offsets'] == offsets
    altered = copy.deepcopy(result); altered['observations'][0]['arrays']['bands'][0][0] += 2e-6
    assert not compare_crosscheck(**{**args, 'replay': {0: altered}})['all_passed']
