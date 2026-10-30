"""Tests for paired inverse candidate ranking and baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.audit_ranking import paired_candidate_ranking, summarize_paired_gains


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
