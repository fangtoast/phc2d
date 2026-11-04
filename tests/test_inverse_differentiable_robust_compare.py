"""Check nominal, scenario and common bands stay tied to physical geometry."""
import copy
import hashlib

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_robust_compare import compare_robust_results
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids


def test_robust_comparison_rebuilds_scenarios_and_common_target_margin():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32})
    offsets = [-1/64, 0., 1/64]
    grids = periodic_perturbation_grids(source, resolution=64, offsets=tuple(offsets))
    hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
    spectra = [[[.40, .60]], [[.41, .59]], [[.42, .58]]]
    candidate = {'structure': source.to_mapping(), 'parameter_hash': source.parameter_hash,
        'master_grid_sha256': hashes[1], 'robust_prediction': {'status': 'surrogate_only',
        'grid_sha256': hashes, 'bands': spectra}}
    lock = {'purpose': 'robust_design', 'records': [{'label': 'trial/top1', 'candidate': candidate}]}
    manifest = {'selection_lock_sha256': 'lock', 'protocol_sha256': 'protocol', 'k_points': [[0., 0.]],
        'lower_band': 1, 'target_frequency': .5, 'role_coverage': [{'label': 'trial/top1', 'case_index': 0}],
        'rows': [{'parameter_hash': source.parameter_hash, 'master_grid_sha256': hashes[1]}]}
    protocol = {'offsets': offsets, 'raster_resolution': 64}
    traces = []
    for index, (low, high) in enumerate(((.405, .595), (.415, .585), (.425, .575))):
        traces.append({'offset': offsets[index], 'provenance': {'evaluated_grid_sha256': hashes[index]},
            'arrays': {'k_points': [[0., 0.]], 'bands': [[low, high]]},
            'row': {'refined_lower_edge_frequency': low, 'refined_upper_edge_frequency': high,
                    'last_refinement_stable': True, 'resolution_stable': True, 'adaptive_refinement_converged': True}})
    result = {'case_index': 0, 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol',
        'parameter_hash': source.parameter_hash, 'observations': traces, 'elapsed_s': 1.,
        'aggregate': {'status': 'numerically_stable', 'row': {'refined_lower_edge_frequency': .425,
            'refined_upper_edge_frequency': .575, 'last_refinement_stable': True,
            'resolution_stable': True, 'adaptive_refinement_converged': True}}}
    args = dict(lock=lock, manifest=manifest, protocol=protocol, results={0: result}, lock_sha256='lock',
                manifest_sha256='manifest', protocol_sha256='protocol')
    report = compare_robust_results(**args)
    row = report['roles'][0]
    assert row['physical_common_edges'] == pytest.approx([.425, .575])
    assert row['physical_common_target_margin'] == pytest.approx(.3)
    assert row['physical_nominal_target_margin'] == pytest.approx(.34)
    assert row['surrogate_common_target_margin'] == pytest.approx(.32)
    assert all(item['same_k_spectrum_mae'] == pytest.approx(.005) for item in row['scenarios'])
    bad = copy.deepcopy(result); bad['observations'][0]['provenance']['evaluated_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='grid mismatch'):
        compare_robust_results(**{**args, 'results': {0: bad}})
