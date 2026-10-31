"""Locked-choice evaluation contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import copy

import pytest

from phc2d.inverse.audit_refined_reconcile import (adjudicate_locked_choice_development,
                                                  evaluate_locked_shared_choices)


def fixture():
    manifest, physical, groups = [], [], []
    for source in range(4):
        eligible = []
        for offset, label in enumerate(('anchor', 'a', 'b', 'c')):
            index = len(manifest)
            row = {'source_index': source, 'label': label, 'parameter_hash': f'{source}-{label}'}
            manifest.append(row)
            physical.append({**row, 'case_index': index, 'target_margin': float(offset),
                             'numerical_gate_passed': True})
            if label in ('a', 'b'):
                eligible.append({'case_index': index, 'label': label, 'parameter_hash': row['parameter_hash'],
                                 'raw_score': float(offset), 'corrected_score': -float(offset)})
        for seed in range(3):
            groups.append({'source_index': source, 'seed': seed, 'anchor_case_index': source * 4,
                           'status': 'choices_locked', 'eligible': copy.deepcopy(eligible),
                           'raw_choice': source * 4 + 2, 'corrected_choice': source * 4 + 1})
    args = dict(source_strata={0: 'x', 1: 'x', 2: 'y', 3: 'y'}, seeds=[0, 1, 2],
                physical_group_sha256={i: f'{i + 1:064x}' for i in range(4)},
                expected_counts={'x': 2, 'y': 2}, stratum_weights={'x': .5, 'y': .5},
                bootstrap_seed=8, bootstrap_samples=50)
    return {'sources': groups}, {'rows': manifest}, {'rows': physical}, args


def test_locked_choices_use_same_uniform_baseline_and_source_bootstrap():
    choices, manifest, physical, args = fixture()
    result = evaluate_locked_shared_choices(choices, manifest, physical, **args)
    assert result['status'] == 'ready'
    assert result['physical_acceptance'] == 'not_evaluated'
    assert result['raw']['summary']['selected_minus_random_mean'] == .5
    assert result['corrected']['summary']['selected_minus_random_mean'] == -.5
    assert result['raw']['summary']['resampling_unit'] == 'source_within_stratum'


@pytest.mark.parametrize('problem', ['missing', 'gate', 'unscoreable'])
def test_incomplete_physical_group_prevents_summary(problem):
    choices, manifest, physical, args = fixture()
    if problem == 'missing':
        physical['rows'].pop()
    elif problem == 'gate':
        physical['rows'][3]['numerical_gate_passed'] = False
    else:
        for group in choices['sources'][:3]:
            group['status'] = 'anchor_unscoreable'
            group.pop('eligible')
    result = evaluate_locked_shared_choices(choices, manifest, physical, **args)
    assert result['status'] == 'unavailable'
    assert result['raw']['summary'] is None


@pytest.mark.parametrize('problem', ['winner', 'eligibility', 'identity'])
def test_tampered_choice_is_rejected(problem):
    choices, manifest, physical, args = fixture()
    if problem == 'winner':
        choices['sources'][0]['raw_choice'] = 1
    elif problem == 'eligibility':
        choices['sources'][0]['eligible'].pop()
        choices['sources'][0]['status'] = 'insufficient_shared_candidates'
    else:
        physical['rows'][0]['parameter_hash'] = 'changed'
    with pytest.raises(ValueError):
        evaluate_locked_shared_choices(choices, manifest, physical, **args)


def test_statistical_and_cross_environment_criteria_are_separate():
    choices, manifest, physical, args = fixture()
    result = evaluate_locked_shared_choices(choices, manifest, physical, **args)
    evaluation = {'manifest_sha256': 'a'*64, 'archive_sha256': 'b'*64,
                  'locked_choice_evaluation': result}
    offset = {'offset': 0., 'k_points_match': True, 'gates_match': True,
              'solver_identity_match': True, 'max_frequency_difference': 1e-7, 'passed': True}
    check = {'manifest_sha256': 'a'*64, 'archive_sha256': 'b'*64, 'frequency_absolute_tolerance': 1e-6,
             'cases': [{'case_index': i, 'offsets': [copy.deepcopy(offset)], 'passed': True} for i in (0, 4)],
             'all_passed': True}
    contract = {'bootstrap_seed': 8, 'bootstrap_samples': 50, 'model_seeds': [0, 1, 2],
                'expected_source_count': 4, 'expected_case_indices': [0, 4],
                'expected_offsets': [0.], 'frequency_tolerance': 1e-6}
    # Corrected winner is worse than the matched random baseline.
    verdict = adjudicate_locked_choice_development(evaluation, check, **contract)
    assert verdict['status'] == 'fail' and verdict['statistical_criterion'] == 'fail'
    assert verdict['cross_environment_criterion'] == 'pass' and verdict['p2_gate'] == 'not_evaluated'
    evaluation['locked_choice_evaluation']['corrected']['summary']['selected_minus_random_mean'] = .5
    evaluation['locked_choice_evaluation']['corrected']['summary']['bootstrap_95_low'] = .1
    evaluation['locked_choice_evaluation']['corrected']['summary']['bootstrap_95_high'] = .9
    assert adjudicate_locked_choice_development(evaluation, check, **contract)['status'] == 'pass'
    assert adjudicate_locked_choice_development(evaluation, None, **contract)['status'] == 'unavailable'
    check['cases'][0]['offsets'][0]['max_frequency_difference'] = 2e-6
    with pytest.raises(ValueError, match='offset decision'):
        adjudicate_locked_choice_development(evaluation, check, **contract)
    check['cases'][0]['offsets'][0]['passed'] = False
    check['cases'][0]['passed'] = False
    check['all_passed'] = False
    assert adjudicate_locked_choice_development(evaluation, check, **contract)['status'] == 'fail'


def test_crosscheck_missing_offset_or_hash_cannot_pass():
    choices, manifest, physical, args = fixture()
    evaluation = {'manifest_sha256': 'a'*64, 'archive_sha256': 'b'*64,
                  'locked_choice_evaluation': evaluate_locked_shared_choices(choices, manifest, physical, **args)}
    check = {'manifest_sha256': 'a'*64, 'archive_sha256': 'b'*64, 'frequency_absolute_tolerance': 1e-6,
             'cases': [{'case_index': 0, 'offsets': [], 'passed': True}], 'all_passed': True}
    contract = {'bootstrap_seed': 8, 'bootstrap_samples': 50, 'model_seeds': [0, 1, 2],
                'expected_source_count': 4, 'expected_case_indices': [0],
                'expected_offsets': [0.], 'frequency_tolerance': 1e-6}
    with pytest.raises(ValueError, match='offset roster'):
        adjudicate_locked_choice_development(evaluation, check, **contract)
    evaluation.pop('archive_sha256')
    with pytest.raises(ValueError, match='digest'):
        adjudicate_locked_choice_development(evaluation, None, **contract)
