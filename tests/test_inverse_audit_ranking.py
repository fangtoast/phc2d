"""Tests for paired inverse candidate ranking and baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.audit_ranking import (paired_candidate_ranking, summarize_paired_gains,
                                        summarize_stratified_paired_gains,
                                        summarize_seed_averaged_stratified_gains)


def rows(scores, actual):
    labels = ['anchor', 'a', 'b']
    proxy = [dict(source_index=1, label=l, parameter_hash=l, status='surrogate_only', predicted_margin=x)
             for l, x in zip(labels, scores)]
    solver = [dict(source_index=1, label=l, parameter_hash=l, status='sampled_solver_only', gap={'target_margin': x})
              for l, x in zip(labels, actual)]
    return proxy, solver


def test_paired_gain_and_exact_random_baseline():
    p, s = rows([0, 2, 1], [0, .4, .2])
    out = paired_candidate_ranking(p, s)[0]
    assert out['selected_label'] == 'a'
    assert out['selected_gain'] == pytest.approx(.4)
    assert out['random_expected_gain'] == pytest.approx(.3)
    assert out['kendall_tau'] == pytest.approx(1)


def test_missing_duplicate_and_undefined_rank_are_not_silenced():
    p, s = rows([0, 1, 1], [0, .2, .2])
    out = paired_candidate_ranking(p, s)[0]
    assert out['kendall_tau'] is None
    with pytest.raises(ValueError, match='identities differ'):
        paired_candidate_ranking(p, s[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        paired_candidate_ranking(p+[p[0]], s)


def test_bootstrap_uses_sources_and_is_reproducible():
    groups = [{'source_index': 1, 'selected_gain': .4, 'random_expected_gain': .3},
              {'source_index': 2, 'selected_gain': .2, 'random_expected_gain': .3}]
    result = summarize_paired_gains(groups, seed=123, bootstrap_samples=1000)
    assert result == summarize_paired_gains(groups, seed=123, bootstrap_samples=1000)
    assert result['selected_minus_random_mean'] == pytest.approx(0)
    assert result['resampling_unit'] == 'source' and result['source_count'] == 2
    assert result['bootstrap_95_low'] <= 0 <= result['bootstrap_95_high']
    with pytest.raises(ValueError, match='unique'):
        summarize_paired_gains(groups + [groups[0]], seed=1)


def test_stratified_bootstrap_respects_frozen_weights_not_pooled_counts():
    groups = [{'source_index': i, 'selected_gain': 1.0 if i < 2 else -.25,
               'random_expected_gain': 0.0} for i in range(6)]
    strata = {i: 'positive' if i < 2 else 'negative' for i in range(6)}
    args = dict(source_strata=strata, expected_counts={'positive': 2, 'negative': 4},
                stratum_weights={'positive': .5, 'negative': .5}, seed=21, bootstrap_samples=1000)
    result = summarize_stratified_paired_gains(groups, **args)
    assert result == summarize_stratified_paired_gains(groups, **args)
    assert result['selected_minus_random_mean'] == pytest.approx(.375)
    assert result['bootstrap_95_low'] == pytest.approx(.375)
    assert result['bootstrap_95_high'] == pytest.approx(.375)
    assert result['resampling_unit'] == 'source_within_stratum'
    assert result['status'] == 'descriptive_only'


def test_stratified_bootstrap_rejects_partial_duplicate_and_unfrozen_inputs():
    groups = [{'source_index': i, 'selected_gain': float(i), 'random_expected_gain': 0.0}
              for i in range(4)]
    args = dict(source_strata={0: 'a', 1: 'a', 2: 'b', 3: 'b'},
                expected_counts={'a': 2, 'b': 2}, stratum_weights={'a': .5, 'b': .5}, seed=1)
    with pytest.raises(ValueError, match='frozen stratum counts'):
        summarize_stratified_paired_gains(groups[:-1], **args)
    with pytest.raises(ValueError, match='unique and frozen'):
        summarize_stratified_paired_gains(groups + [groups[0]], **args)
    with pytest.raises(ValueError, match='weights'):
        summarize_stratified_paired_gains(groups, **dict(args, stratum_weights={'a': .5, 'b': .6}))
    with pytest.raises(ValueError, match='source strata'):
        summarize_stratified_paired_gains(groups, **dict(args, source_strata={0: 'a', 1: 'a', 2: 'c', 3: 'c'}))


def test_model_seeds_are_averaged_before_resampling_sources():
    strata = {0: 'a', 1: 'a', 2: 'b', 3: 'b'}
    groups = [{'source_index': source, 'seed': seed, 'selected_gain': float(source + seed),
               'random_expected_gain': float(seed), 'physical_group_sha256': str(source) * 64}
              for source in strata for seed in (0, 1, 2)]
    args = dict(source_strata=strata, seeds=[0, 1, 2], expected_counts={'a': 2, 'b': 2},
                stratum_weights={'a': .5, 'b': .5}, bootstrap_seed=9, bootstrap_samples=1000)
    result = summarize_seed_averaged_stratified_gains(groups, **args)
    manual = [{'source_index': source, 'selected_gain': source, 'random_expected_gain': 0.}
              for source in strata]
    reference = summarize_stratified_paired_gains(manual, source_strata=strata,
        expected_counts={'a': 2, 'b': 2}, stratum_weights={'a': .5, 'b': .5}, seed=9, bootstrap_samples=1000)
    assert result['source_count'] == 4 and result['source_seed_count'] == 12
    assert result['selected_minus_random_mean'] == reference['selected_minus_random_mean']
    assert result['bootstrap_95_low'] == reference['bootstrap_95_low']
    assert result['bootstrap_95_high'] == reference['bootstrap_95_high']
    assert result['resampling_unit'] == 'source_within_stratum'
    assert result['status'] == 'descriptive_only'


def test_model_seed_summary_rejects_missing_duplicate_and_mismatched_physics():
    strata = {0: 'a', 1: 'a'}
    groups = [{'source_index': source, 'seed': seed, 'selected_gain': .2, 'random_expected_gain': .1,
               'physical_group_sha256': str(source) * 64} for source in strata for seed in (0, 1)]
    args = dict(source_strata=strata, seeds=[0, 1], expected_counts={'a': 2},
                stratum_weights={'a': 1.}, bootstrap_seed=2)
    with pytest.raises(ValueError, match='incomplete'):
        summarize_seed_averaged_stratified_gains(groups[:-1], **args)
    with pytest.raises(ValueError, match='exactly once'):
        summarize_seed_averaged_stratified_gains(groups+[groups[0]], **args)
    changed = [dict(row) for row in groups]
    changed[1]['physical_group_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='share one physical group'):
        summarize_seed_averaged_stratified_gains(changed, **args)
    changed = [dict(row) for row in groups]
    changed[1]['selected_gain'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        summarize_seed_averaged_stratified_gains(changed, **args)
