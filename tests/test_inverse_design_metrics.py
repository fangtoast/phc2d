"""Test inverse-design denominators and physical evidence boundaries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.design_metrics import (compare_paired_budget_curves, compare_paired_success, compare_paired_success_audited,
                                          summarize_design_outcomes, summarize_verified_budget_curve)
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger, reserve_resources, settle_slurm_resources


def verified(lower, upper):
    return {'status': 'verified', 'numerical_gate_passed': True, 'artifact_sha256': 'a' * 64,
            'lower_edge': lower, 'upper_edge': upper}


def planned(*targets):
    return [{'target_id': target, 'seed': 0, 'target_frequency': 1.0} for target in targets]


def test_budget_curve_counts_failed_attempts_and_honors_both_resource_caps():
    design_a, design_b, design_c = 'a' * 64, 'b' * 64, 'c' * 64
    def entry(name, design, cpu, gpu, evidence):
        return {'attempt_id': name, 'design_sha256': design, 'cumulative_cpu_core_seconds': cpu,
                'cumulative_gpu_seconds': gpu, 'physical': evidence}
    attempts = [
        entry('first', design_a, 3, 2, {**verified(1.1, 1.3), 'design_sha256': design_a}),
        entry('failed', design_b, 5, 2, {'status': 'failed', 'reason': 'timeout'}),
        entry('improved', design_b, 8, 5, {**verified(.9, 1.2), 'design_sha256': design_b}),
        entry('over_gpu', design_c, 9, 9, {**verified(.8, 1.2), 'design_sha256': design_c}),
    ]
    curve = summarize_verified_budget_curve(attempts, target_frequency=1.,
        cpu_milestones=[0, 3, 5, 8, 12], gpu_cap_seconds=6)
    assert [p['attempts_completed'] for p in curve['points']] == [0, 1, 2, 3, 3]
    assert curve['points'][0]['best_verified_margin'] is None
    assert curve['points'][1]['best_verified_margin'] == pytest.approx(-.2)
    assert curve['points'][2]['best_verified_margin'] == pytest.approx(-.2)
    assert curve['points'][-1]['best_verified_margin'] == pytest.approx(.2)
    assert curve['points'][-1]['best_design_sha256'] == design_b
    assert curve['points'][-1]['best_attempt_id'] == 'improved'
    assert curve['points'][-1]['best_artifact_sha256'] == 'a' * 64
    assert curve['first_reached']['attempt_id'] == 'improved'
    assert curve['first_reached']['cpu_core_seconds'] == 8


def test_budget_curve_rejects_wrong_identity_and_nonmonotone_or_duplicate_attempts():
    design = 'a' * 64
    row = {'attempt_id': 'one', 'design_sha256': design, 'cumulative_cpu_core_seconds': 2,
           'cumulative_gpu_seconds': 1, 'physical': {**verified(.9, 1.1), 'design_sha256': design}}
    kwargs = {'target_frequency': 1., 'cpu_milestones': [0, 3], 'gpu_cap_seconds': 2}
    with pytest.raises(ValueError, match='different design'):
        summarize_verified_budget_curve([{**row, 'physical': {**row['physical'], 'design_sha256': 'b' * 64}}], **kwargs)
    with pytest.raises(ValueError, match='identify the attempted design'):
        summarize_verified_budget_curve([{**row, 'physical': verified(.9, 1.1)}], **kwargs)
    with pytest.raises(ValueError, match='unique'):
        summarize_verified_budget_curve([row, row], **kwargs)
    with pytest.raises(ValueError, match='cannot decrease'):
        summarize_verified_budget_curve([row, {**row, 'attempt_id': 'two', 'cumulative_gpu_seconds': 0}], **kwargs)
    with pytest.raises(ValueError, match='strictly increasing'):
        summarize_verified_budget_curve([row], **{**kwargs, 'cpu_milestones': [3, 3]})
    with pytest.raises(ValueError, match='unrun candidate'):
        summarize_verified_budget_curve([{**row, 'physical': {'status': 'not_run'}}], **kwargs)


def test_budget_curve_strict_target_and_unverified_initial_state():
    design = 'd' * 64
    attempts = [{'attempt_id': 'boundary', 'design_sha256': design, 'cumulative_cpu_core_seconds': 1,
                 'cumulative_gpu_seconds': 0, 'physical': {**verified(1., 1.2), 'design_sha256': design}}]
    curve = summarize_verified_budget_curve(attempts, target_frequency=1., cpu_milestones=[0, 1], gpu_cap_seconds=0)
    assert curve['points'][0]['best_verified_margin'] is None
    assert curve['points'][1]['best_verified_margin'] == 0
    assert curve['first_reached'] is None


def test_paired_budget_curve_resamples_targets_and_keeps_missing_failures():
    roster = [{'target_id': target, 'seed': seed, 'target_frequency': 1.}
              for target in ('a', 'b') for seed in (0, 1)]
    design = 'd' * 64
    def attempt(name, cpu, successful):
        physical = ({**verified(.9, 1.1), 'design_sha256': design} if successful
                    else {'status': 'failed', 'reason': 'timeout'})
        return {'attempt_id': name, 'design_sha256': design, 'cumulative_cpu_core_seconds': cpu,
                'cumulative_gpu_seconds': 1, 'physical': physical}
    left = {('a', 0): [attempt('a0', 2, True)], ('a', 1): [attempt('a1', 4, True)],
            ('b', 0): [attempt('b0', 2, True)], ('b', 1): [attempt('b1', 2, False)]}
    right = {('a', 0): [attempt('a0', 2, True)], ('a', 1): [attempt('a1', 2, False)],
             ('b', 0): [attempt('b0', 2, True)]}
    kwargs = {'cpu_milestones': [0, 2, 4], 'gpu_cap_seconds': 1,
              'bootstrap_seed': 7, 'bootstrap_samples': 100}
    result = compare_paired_budget_curves(roster, left=left, right=right, **kwargs)
    assert result['resampling_unit'] == 'target_id' and result['target_seed_count'] == 4
    assert result['interval_type'] == 'pointwise_not_simultaneous'
    assert result['points'][0]['paired_attainment_difference'] is None
    assert result['points'][1]['paired_attainment_difference'] == 0
    assert result['points'][2]['paired_attainment_difference'] == .25
    assert result['points'][2]['left']['attainment_rate_all'] == .75
    assert result['points'][2]['right']['attainment_rate_all'] == .5
    assert result['points'][2]['right']['missing_trajectories'] == 1
    assert result['points'][2]['right']['verified'] == 2
    assert result == compare_paired_budget_curves(roster, left=left, right=right, **kwargs)
    with pytest.raises(ValueError, match='outside the frozen roster'):
        compare_paired_budget_curves(roster, left={('other', 0): []}, right=right, **kwargs)
    with pytest.raises(ValueError, match='nonnegative integer seed'):
        compare_paired_budget_curves(roster, left={('a', True): []}, right=right, **kwargs)
    with pytest.raises(ValueError, match='same frozen seed roster'):
        compare_paired_budget_curves(roster[:-1], left={}, right={}, **kwargs)


def test_paired_budget_curve_does_not_report_rate_for_unattempted_method():
    roster = planned('a')
    design = 'e' * 64
    failed = {'attempt_id': 'f', 'design_sha256': design, 'cumulative_cpu_core_seconds': 1,
              'cumulative_gpu_seconds': 0, 'physical': {'status': 'failed', 'reason': 'timeout'}}
    result = compare_paired_budget_curves(roster, left={('a', 0): [failed]}, right={},
        cpu_milestones=[1], gpu_cap_seconds=0, bootstrap_seed=1, bootstrap_samples=10)
    point = result['points'][0]
    assert point['left']['attainment_rate_all'] == 0
    assert point['right']['attainment_rate_all'] is None
    assert point['paired_attainment_difference'] is None and point['bootstrap_95_low'] is None


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
    assert result['mean_successful_joint_common_width'] is None
    records[0]['nominal'] = {'status': 'failed', 'reason': 'solver failed'}
    attempted = summarize_design_outcomes(planned('a'), records)
    assert attempted['nominal_success_rate_all'] == 0.
    assert attempted['nominal_success_rate_verified'] is None


def test_joint_bandwidth_intersects_only_the_same_verified_base_design():
    nominal = {**verified(.7, 1.1), 'design_sha256': 'c' * 64}
    unseen = {**verified(.95, 1.3), 'design_sha256': 'c' * 64}
    row = {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': nominal, 'unseen': unseen}
    result = summarize_design_outcomes(planned('a'), [row])
    assert result['counts']['both_verified'] == 1
    assert result['counts']['both_success'] == 1
    assert result['mean_successful_nominal_common_width'] == pytest.approx(.4)
    assert result['mean_successful_joint_common_width'] == pytest.approx(.15)
    missing = {**row, 'unseen': verified(.95, 1.3)}
    result = summarize_design_outcomes(planned('a'), [missing])
    assert result['counts']['both_identity_unavailable'] == 1
    assert result['counts']['both_verified'] == 0 and result['counts']['both_success'] == 0
    assert result['mean_successful_joint_common_width'] is None
    different = {**row, 'unseen': {**unseen, 'design_sha256': 'd' * 64}}
    with pytest.raises(ValueError, match='different base designs'):
        summarize_design_outcomes(planned('a'), [different])
    malformed = {**row, 'unseen': {**unseen, 'design_sha256': 'bad'}}
    with pytest.raises(ValueError, match='invalid base design identity'):
        summarize_design_outcomes(planned('a'), [malformed])


def test_paired_joint_metric_excludes_unidentified_or_unverified_intersections():
    nominal = {**verified(.7, 1.1), 'design_sha256': 'c' * 64}
    unseen = {**verified(.95, 1.3), 'design_sha256': 'c' * 64}
    known = {'target_id': 'a', 'seed': 0, 'target_frequency': 1., 'nominal': nominal, 'unseen': unseen}
    unknown = {**known, 'unseen': verified(.95, 1.3)}
    caps = {'cpu_core_seconds': 10, 'gpu_seconds': 0}
    result = compare_paired_success(planned('a'), left=[known], right=[unknown],
                                    left_budget=caps, right_budget=caps, metric='both',
                                    bootstrap_seed=2, bootstrap_samples=10)
    assert result['all_roster']['mean_difference'] == 1.
    assert result['verified_pairs_conditional']['pair_count'] == 0
    assert result['right']['counts']['both_identity_unavailable'] == 1


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
