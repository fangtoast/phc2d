"""Tests for numerically gated inverse optimization feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.refinement_feedback import compare_refined_candidates


def observation(lo, hi, *, stable=True, signature='same'):
    return {'row': {'lower_band_one_based': 2, 'refined_lower_edge_frequency': lo,
                    'refined_upper_edge_frequency': hi, 'last_refinement_stable': stable,
                    'resolution_stable': stable, 'adaptive_refinement_converged': stable},
            'provenance': {'comparison_contract_sha256': signature}}


def compare(a, b):
    return compare_refined_candidates(a, b, target_frequency=0.5, predicted_center=0.1,
                                      predicted_candidate=0.2, radius=0.3,
                                      min_radius=0.05, max_radius=1)


def test_acceptance_uses_real_edge_improvement():
    r = compare(observation(.45, .55), observation(.4, .6))
    assert r['status'] == 'compared' and r['decision']['accepted']
    assert r['candidate_margin'] > r['center_margin']


def test_unconverged_candidate_is_inconclusive_not_rejected():
    r = compare(observation(.45, .55), observation(.4, .6, stable=False))
    assert r == {'status': 'numerical_gate_failed', 'decision': None, 'radius': 0.3}


def test_mismatched_contract_is_never_compared():
    with pytest.raises(ValueError, match='contracts'):
        compare(observation(.45, .55), observation(.4, .6, signature='other'))
