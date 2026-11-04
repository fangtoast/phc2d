"""Exercise exact grid reuse without changing physical solver contracts."""
import copy
import hashlib
import numpy as np
import pytest
from phc2d.inverse.differentiable_feedback_reuse import plan_seed_reuse, rebind_seed_results
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids


def test_seed_reuse_deduplicates_and_rejects_different_contract():
    common = {'protocol_sha256': 'protocol', 'k_points': [[0, 0]], 'lower_band': 1,
              'target_frequency': .5, 'solver_config': {'resolution': 192}, 'raster_resolution': 64}
    reference = {**common, 'rows': [{'label': 'r0', 'master_grid_sha256': 'grid-a'}]}
    target = {**common, 'selection_lock_sha256': 'lock',
              'rows': [{'label': 't0', 'master_grid_sha256': 'grid-a'},
                       {'label': 't1', 'master_grid_sha256': 'grid-b'}]}
    plan, fresh = plan_seed_reuse(target, reference, target_sha256='target', reference_sha256='reference')
    assert plan['reused'][0]['reference_case_index'] == 0
    assert plan['fresh'][0]['target_case_index'] == 1
    assert len(fresh['rows']) == 1 and fresh['rows'][0]['target_case_index'] == 1
    other = copy.deepcopy(reference); other['solver_config']['resolution'] = 128
    with pytest.raises(ValueError, match='contracts'):
        plan_seed_reuse(target, other, target_sha256='target', reference_sha256='reference')
    other_target = {**target, 'target_frequency': .57}
    with pytest.raises(ValueError, match='contracts'):
        plan_seed_reuse(other_target, reference, target_sha256='target', reference_sha256='reference')
    explicit, _ = plan_seed_reuse(other_target, reference, target_sha256='target', reference_sha256='reference',
                                  allow_target_change=True)
    assert explicit['allow_target_change'] is True and explicit['target_frequencies']['reference'] == .5


def test_rebound_evidence_retains_original_source_and_rejects_tampering():
    common = {'protocol_sha256': 'protocol', 'k_points': [[0, 0]], 'lower_band': 1,
              'target_frequency': .5, 'solver_config': {'resolution': 192}, 'raster_resolution': 64}
    reference = {**common, 'rows': [{'label': 'ref', 'source_index': 8,
                                   'parameter_hash': 'old', 'master_grid_sha256': 'grid'}]}
    target = {**common, 'selection_lock_sha256': 'lock', 'rows': [{'label': 'seed',
              'source_index': 8, 'parameter_hash': 'new', 'master_grid_sha256': 'grid'}]}
    plan, fresh = plan_seed_reuse(target, reference, target_sha256='target', reference_sha256='reference')
    plan['fresh_manifest_sha256'] = 'fresh'
    result = {'case_index': 0, 'manifest_sha256': 'reference', 'protocol_sha256': 'protocol',
              'parameter_hash': 'old', 'aggregate': {'provenance': {'parameter_hash': 'old'}},
              'observations': [{'offset': 0.0, 'provenance': {'evaluated_grid_sha256': 'grid'}}]}
    kwargs = dict(target=target, reference=reference, fresh=fresh, plan=plan,
                  target_sha256='target', reference_sha256='reference', fresh_sha256='fresh',
                  reference_results={0: (result, 'source-result-sha')}, fresh_results={})
    derived = rebind_seed_results(**kwargs)[0]
    assert derived['parameter_hash'] == 'new'
    assert derived['source_evidence']['parameter_hash'] == 'old'
    assert derived['source_evidence']['result_sha256'] == 'source-result-sha'
    assert result['parameter_hash'] == 'old'
    bad = copy.deepcopy(result); bad['observations'][0]['provenance']['evaluated_grid_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        rebind_seed_results(**{**kwargs, 'reference_results': {0: (bad, 'sha')}})


def test_reuse_checks_all_manufacturing_scenario_grids():
    structure = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32})
    offsets = [-1/64, 0., 1/64]
    grids = periodic_perturbation_grids(structure, resolution=64, offsets=tuple(offsets))
    hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
    common = {'protocol_sha256': 'protocol', 'k_points': [[0, 0]], 'lower_band': 1,
              'target_frequency': .5, 'solver_config': {'resolution': 192}, 'raster_resolution': 64}
    row = {'label': 'case', 'source_index': 8, 'parameter_hash': structure.parameter_hash,
           'master_grid_sha256': hashes[1], 'structure': structure.to_mapping()}
    reference = {**common, 'rows': [row]}
    target = {**common, 'selection_lock_sha256': 'lock', 'rows': [row]}
    plan, fresh = plan_seed_reuse(target, reference, target_sha256='target', reference_sha256='reference')
    plan['fresh_manifest_sha256'] = 'fresh'
    result = {'case_index': 0, 'manifest_sha256': 'reference', 'protocol_sha256': 'protocol',
        'parameter_hash': structure.parameter_hash, 'aggregate': {'provenance': {'parameter_hash': structure.parameter_hash}},
        'observations': [{'offset': offset, 'provenance': {'parameter_hash': structure.parameter_hash,
                          'evaluated_grid_sha256': hashes[i]}} for i, offset in enumerate(offsets)]}
    kwargs = dict(target=target, reference=reference, fresh=fresh, plan=plan,
        target_sha256='target', reference_sha256='reference', fresh_sha256='fresh',
        reference_results={0: (result, 'source-result-sha')}, fresh_results={})
    assert len(rebind_seed_results(**kwargs)[0]['observations']) == 3
    bad = copy.deepcopy(result); bad['observations'][2]['provenance']['evaluated_grid_sha256'] = 'bad'
    with pytest.raises(ValueError, match='scenario grids'):
        rebind_seed_results(**{**kwargs, 'reference_results': {0: (bad, 'sha')}})
