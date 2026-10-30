"""Test inverse-design denominators and physical evidence boundaries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.design_metrics import (compare_paired_success, compare_paired_success_audited,
                                          summarize_design_outcomes)
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger, reserve_resources, settle_slurm_resources


def verified(lower, upper):
    return {'status': 'verified', 'numerical_gate_passed': True, 'artifact_sha256': 'a' * 64,
            'lower_edge': lower, 'upper_edge': upper}


def planned(*targets):
    return [{'target_id': target, 'seed': 0, 'target_frequency': 1.0} for target in targets]


def test_missing_and_unresolved_do_not_inflate_success_or_false_positive():
    positive = {'checkpoint_sha256': 'b' * 64, 'target_margin': .1}
    records = [
        {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2), 'unseen': verified(.9, 1.1),
         'prediction': positive},
        {'target_id': 'b', 'seed': 0, 'target_frequency': 1., 'nominal': verified(1.1, .9),
         'unseen': {'status': 'failed', 'reason': 'solver failed'}, 'prediction': positive},
        {'target_id': 'c', 'seed': 0, 'target_frequency': 1., 'nominal': {'status': 'unresolved', 'reason': 'resolution'},
         'unseen': {'status': 'not_run'}, 'prediction': positive},
    ]
    result = summarize_design_outcomes(planned('a', 'b', 'c', 'd'), records)
    assert result['counts']['missing'] == 1
    assert result['counts']['nominal_verified'] == 2
    assert result['counts']['predicted_positive_unverified'] == 1
    assert result['nominal_success_rate_all'] == .25
    assert result['nominal_success_rate_verified'] == .5
    assert result['unseen_success_rate_all'] == .25
    assert result['unseen_success_rate_verified'] == 1.
    assert result['false_positive_rate_verified_predictions'] == .5
    assert result['mean_successful_nominal_common_width'] == pytest.approx(.4)


def test_no_verified_or_predicted_positive_has_undefined_conditional_rates():
    records = [{'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': {'status': 'not_run'},
                'unseen': {'status': 'not_run'}}]
    result = summarize_design_outcomes(planned('a'), records)
    assert result['nominal_success_rate_all'] is None
    assert result['nominal_success_rate_verified'] is None
    assert result['unseen_success_rate_all'] is None
    assert result['both_success_rate_all'] is None
    assert result['false_positive_rate_verified_predictions'] is None
    assert result['mean_successful_nominal_common_width'] is None
    records[0]['nominal'] = {'status': 'failed', 'reason': 'solver failed'}
    attempted = summarize_design_outcomes(planned('a'), records)
    assert attempted['nominal_success_rate_all'] == 0.
    assert attempted['nominal_success_rate_verified'] is None


def test_forged_verified_evidence_duplicate_or_unknown_roster_fails():
    record = {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2), 'unseen': {'status': 'not_run'}}
    with pytest.raises(ValueError, match='duplicate'):
        summarize_design_outcomes(planned('a'), [record, record])
    with pytest.raises(ValueError, match='unexpected'):
        summarize_design_outcomes(planned('b'), [record])
    record['nominal']['numerical_gate_passed'] = False
    with pytest.raises(ValueError, match='numerical gate'):
        summarize_design_outcomes(planned('a'), [record])
    record['nominal'] = {'status': 'unresolved', 'reason': 'resolution', 'lower_edge': .8}
    with pytest.raises(ValueError, match='unverified'):
        summarize_design_outcomes(planned('a'), [record])


def test_same_target_cannot_silently_change_frequency_across_seeds():
    roster = [{'target_id': 'a', 'seed': 0, 'target_frequency': 1.},
              {'target_id': 'a', 'seed': 1, 'target_frequency': 1.1}]
    with pytest.raises(ValueError, match='one frequency'):
        summarize_design_outcomes(roster, [])


def test_observation_frequency_and_checkpoint_are_stable_within_seed():
    first = {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2),
             'unseen': {'status': 'not_run'}, 'prediction': {'checkpoint_sha256': 'a' * 64, 'target_margin': .1}}
    second = {'target_id': 'b', 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2),
              'unseen': {'status': 'not_run'}, 'prediction': {'checkpoint_sha256': 'b' * 64, 'target_margin': .1}}
    with pytest.raises(ValueError, match='checkpoint identity differs'):
        summarize_design_outcomes(planned('a', 'b'), [first, second])
    second['prediction']['checkpoint_sha256'] = 'a' * 64
    second['target_frequency'] = 1.1
    with pytest.raises(ValueError, match='frequency differs'):
        summarize_design_outcomes(planned('a', 'b'), [first, second])


def test_paired_comparison_resamples_targets_not_correlated_seeds():
    roster = [{'target_id': target, 'seed': seed, 'target_frequency': 1.}
              for target in ('a', 'b') for seed in (0, 1)]
    def result(target, seed, success):
        return {'target_id': target, 'seed': seed, 'target_frequency': 1.,
                'nominal': verified(.8, 1.2) if success else verified(1.1, .9),
                'unseen': {'status': 'not_run'}}
    left = [result('a', 0, True), result('a', 1, True), result('b', 0, True), result('b', 1, False)]
    right = [result('a', 0, True), result('a', 1, False), result('b', 0, True), result('b', 1, False)]
    caps = {'cpu_core_seconds': 100, 'gpu_seconds': 10}
    comparison = compare_paired_success(roster, left=left, right=right, left_budget=caps, right_budget=caps,
                                        metric='nominal', bootstrap_seed=7, bootstrap_samples=100)
    assert comparison['resampling_unit'] == 'target_id'
    assert comparison['all_roster']['target_count'] == 2
    assert comparison['all_roster']['pair_count'] == 4
    assert comparison['all_roster']['mean_difference'] == .25
    assert comparison['verified_pairs_conditional']['pair_count'] == 4
    with pytest.raises(ValueError, match='identical resource caps'):
        compare_paired_success(roster, left=left, right=right, left_budget=caps,
                               right_budget={'cpu_core_seconds': 101, 'gpu_seconds': 10},
                               metric='nominal', bootstrap_seed=7)
    with pytest.raises(ValueError, match='same frozen seed roster'):
        compare_paired_success(roster[:-1], left=left[:-1], right=right[:-1], left_budget=caps,
                               right_budget=caps, metric='nominal', bootstrap_seed=7)


def test_missing_pairs_and_unattempted_unseen_are_explicit():
    roster = planned('a', 'b')
    one = {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2), 'unseen': {'status': 'not_run'}}
    caps = {'cpu_core_seconds': 10, 'gpu_seconds': 0}
    result = compare_paired_success(roster, left=[one], right=[one], left_budget=caps, right_budget=caps,
                                    metric='nominal', bootstrap_seed=1, bootstrap_samples=10)
    assert result['left']['counts']['missing'] == 1
    assert result['all_roster']['pair_count'] == 2
    assert result['verified_pairs_conditional']['pair_count'] == 1
    assert result['verified_pairs_conditional']['bootstrap_95_low'] is None
    with pytest.raises(ValueError, match='attempted'):
        compare_paired_success(roster, left=[one], right=[one], left_budget=caps, right_budget=caps,
                               metric='unseen', bootstrap_seed=1)


def test_audited_comparison_checks_every_trial_allocation_and_budget():
    roster = planned('a', 'b')
    records = [{'target_id': target, 'seed': 0, 'target_frequency': 1., 'nominal': verified(.8, 1.2),
                'unseen': {'status': 'not_run'}} for target in ('a', 'b')]
    cap = ResourceAmount(100, 0)
    ledgers, accounting = {}, {}
    for index, identity in enumerate([('left', 'a'), ('left', 'b'), ('right', 'a'), ('right', 'b')]):
        job = str(index + 10)
        raw = f'{job}|COMPLETED|5|3|cpu=3,node=1\n'.encode()
        ledger = reserve_resources(ResourceLedger(cap), task_id='solve', amount=ResourceAmount(30, 0))
        ledger = settle_slurm_resources(ledger, task_id='solve', cluster_id='m9', accounting_bytes=raw,
                                        expected_gpus={job: 0})
        ledgers[identity] = ledger
        accounting[identity] = {'solve': {'cluster_id': 'm9', 'raw': raw, 'expected_gpus': {job: 0}}}
    def select(items, arm):
        return {(target, 0): value for (side, target), value in items.items() if side == arm}
    args = dict(expected=roster, left=records, right=records,
                budget={'cpu_core_seconds': 100, 'gpu_seconds': 0},
                left_ledgers=select(ledgers, 'left'), right_ledgers=select(ledgers, 'right'),
                left_accounting=select(accounting, 'left'), right_accounting=select(accounting, 'right'),
                metric='nominal', bootstrap_seed=7, bootstrap_samples=10)
    result = compare_paired_success_audited(**args)
    assert result['comparison']['all_roster']['mean_difference'] == 0
    assert result['resources']['left']['cpu_core_seconds'] == 30
    assert result['resources']['right']['cpu_core_seconds'] == 30
    duplicate = dict(args['right_ledgers'])
    duplicate[('a', 0)] = args['left_ledgers'][('a', 0)]
    duplicate_evidence = dict(args['right_accounting'])
    duplicate_evidence[('a', 0)] = args['left_accounting'][('a', 0)]
    with pytest.raises(ValueError, match='shared or counted'):
        compare_paired_success_audited(**dict(args, right_ledgers=duplicate, right_accounting=duplicate_evidence))
    with pytest.raises(ValueError, match='frozen roster'):
        compare_paired_success_audited(**dict(args, left_ledgers={}))
    wrong_caps = dict(args['left_ledgers'])
    wrong_caps[('a', 0)] = ResourceLedger(ResourceAmount(101, 0))
    with pytest.raises(ValueError, match='limits'):
        compare_paired_success_audited(**dict(args, left_ledgers=wrong_caps))
    unpaid = dict(args['left_ledgers'])
    unpaid[('a', 0)] = ResourceLedger(cap)
    no_accounting = dict(args['left_accounting'])
    no_accounting[('a', 0)] = {}
    with pytest.raises(ValueError, match='no settled accounting'):
        compare_paired_success_audited(**dict(args, left_ledgers=unpaid, left_accounting=no_accounting))
