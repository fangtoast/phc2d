"""Contract tests for independently locked inverse method comparisons.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import copy

import pytest

from phc2d.inverse.locked_variant_evaluation import (adjudicate_projected_anchor_development,
                                                     evaluate_locked_variants)


def fixture():
    manifest, physical, sources = [], [], []
    for source in range(4):
        start = len(manifest)
        for offset, label in enumerate(('anchor', 'z', 'a')):
            row = {'source_index': source, 'label': label, 'parameter_hash': f'{source}-{label}'}
            manifest.append(row)
            physical.append({**row, 'case_index': start+offset, 'target_margin': float(offset),
                             'numerical_gate_passed': True})
        hashes = [f'{source}-z', f'{source}-a']
        methods = {}
        for name in ('raw', 'projected', 'projected_anchor'):
            scores = [{'label': 'z', 'parameter_hash': hashes[0], 'margin': 1.},
                      {'label': 'a', 'parameter_hash': hashes[1], 'margin': 1.}]
            methods[name] = {'status': 'selected', 'label': 'z', 'parameter_hash': hashes[0], 'scores': scores}
        sources.append({'source_index': source, 'anchor_case_index': start,
            'anchor_numerical_gate_passed': True, 'raw_shared_count': 2, 'projected_shared_count': 2,
            'raw_shared_candidate_hashes': list(hashes), 'projected_shared_candidate_hashes': list(hashes),
            'models': [{'seed': seed, 'methods': copy.deepcopy(methods)} for seed in range(3)]})
    args = {'source_strata': {0: 'x', 1: 'x', 2: 'y', 3: 'y'}, 'seeds': [0, 1, 2],
            'physical_group_sha256': {i: f'{i+1:064x}' for i in range(4)},
            'expected_counts': {'x': 2, 'y': 2}, 'stratum_weights': {'x': .5, 'y': .5},
            'bootstrap_seed': 8, 'bootstrap_samples': 50}
    return {'sources': sources}, {'rows': manifest}, {'rows': physical}, args


def test_variants_use_source_units_and_manifest_order_for_ties():
    choices, manifest, physical, args = fixture()
    result = evaluate_locked_variants(choices, manifest, physical, **args)
    assert result['physical_acceptance'] == 'not_evaluated'
    for item in result['methods'].values():
        assert item['status'] == 'ranking_summary_ready'
        assert item['summary']['selected_minus_random_mean'] == -.5
        assert item['summary']['resampling_unit'] == 'source_within_stratum'
        assert item['summary']['source_count'] == 4


@pytest.mark.parametrize('failure', ['missing', 'gate'])
def test_missing_or_failed_physical_case_prevents_all_summaries(failure):
    choices, manifest, physical, args = fixture()
    if failure == 'missing':
        physical['rows'].pop()
    else:
        physical['rows'][2]['numerical_gate_passed'] = False
    result = evaluate_locked_variants(choices, manifest, physical, **args)
    assert all(item['status'] == 'ranking_summary_unavailable' for item in result['methods'].values())


def test_raw_unscoreable_does_not_suppress_projected_comparison():
    choices, manifest, physical, args = fixture()
    for model in choices['sources'][0]['models']:
        model['methods']['raw'] = {'status': 'unscoreable_shared_eligibility'}
    result = evaluate_locked_variants(choices, manifest, physical, **args)
    assert result['methods']['raw']['status'] == 'ranking_summary_unavailable'
    assert result['methods']['projected_anchor']['status'] == 'ranking_summary_ready'


def test_each_variant_uses_its_own_frozen_uniform_candidate_set():
    choices, manifest, physical, args = fixture()
    row = {'source_index': 0, 'label': 'b', 'parameter_hash': '0-b'}
    index = len(manifest['rows'])
    manifest['rows'].append(row)
    physical['rows'].append({**row, 'case_index': index, 'target_margin': 3., 'numerical_gate_passed': True})
    source = choices['sources'][0]
    source['projected_shared_candidate_hashes'].append('0-b')
    source['projected_shared_count'] = 3
    for model in source['models']:
        for name in ('projected', 'projected_anchor'):
            method = model['methods'][name]
            method['scores'].append({'label': 'b', 'parameter_hash': '0-b', 'margin': 2.})
            method.update(label='b', parameter_hash='0-b')
    result = evaluate_locked_variants(choices, manifest, physical, **args)
    assert result['methods']['raw']['summary']['selected_minus_random_mean'] == -.5
    assert result['methods']['projected']['summary']['selected_minus_random_mean'] == -.125


@pytest.mark.parametrize('failure', ['winner', 'eligibility', 'physical_identity'])
def test_inconsistent_locked_evidence_is_rejected(failure):
    choices, manifest, physical, args = fixture()
    if failure == 'winner':
        choices['sources'][0]['models'][0]['methods']['raw']['label'] = 'a'
    elif failure == 'eligibility':
        choices['sources'][0]['projected_shared_candidate_hashes'].reverse()
    else:
        physical['rows'][0]['parameter_hash'] = 'changed'
    with pytest.raises(ValueError):
        evaluate_locked_variants(choices, manifest, physical, **args)


def test_projected_development_decision_requires_crosscheck_and_checks_grid():
    choices, manifest, physical, args = fixture()
    evaluation = evaluate_locked_variants(choices, manifest, physical, **args)
    summary = evaluation['methods']['projected_anchor']['summary']
    summary.update(selected_minus_random_mean=.2, bootstrap_95_low=.1, bootstrap_95_high=.3)
    evidence = {'evaluation': evaluation, 'manifest_sha256': 'a'*64,
                'anchor_archive_sha256': 'b'*64, 'nonanchor_archive_sha256': 'c'*64}
    offset = {'offset': 0., 'grid_match': True, 'k_points_match': True, 'gates_match': True,
              'solver_identity_match': True, 'max_frequency_difference': 1e-7, 'passed': True}
    cross = {'manifest_sha256': 'a'*64, 'anchor_archive_sha256': 'b'*64,
             'nonanchor_archive_sha256': 'c'*64, 'frequency_absolute_tolerance': 1e-6,
             'cases': [{'case_index': i, 'offsets': [copy.deepcopy(offset)], 'passed': True} for i in (0, 1)],
             'all_passed': True}
    params = {'bootstrap_seed': 8, 'bootstrap_samples': 50, 'model_seeds': [0,1,2],
              'expected_source_count': 4, 'expected_case_indices': [0,1],
              'expected_offsets': [0.], 'frequency_tolerance': 1e-6}
    assert adjudicate_projected_anchor_development(evidence, None, **params)['status'] == 'unavailable'
    verdict = adjudicate_projected_anchor_development(evidence, cross, **params)
    assert verdict['status'] == 'pass' and verdict['p2_gate'] == 'not_evaluated'
    cross['cases'][0]['offsets'][0].update(grid_match=False, passed=False)
    cross['cases'][0]['passed'] = cross['all_passed'] = False
    assert adjudicate_projected_anchor_development(evidence, cross, **params)['status'] == 'fail'
    cross['anchor_archive_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='archive identity'):
        adjudicate_projected_anchor_development(evidence, cross, **params)
