"""Tests for physically checked trust-region step decisions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.trust_region import update_trust_region


def decide(predicted, actual, **kwargs):
    return update_trust_region(predicted_center=0, predicted_candidate=predicted,
                               actual_center=1, actual_candidate=1 + actual,
                               radius=0.4, min_radius=0.1, max_radius=0.5, **kwargs)


def test_accepts_physical_improvement_and_caps_expansion():
    d = decide(1, 0.8)
    assert d.accepted and d.radius == 0.5
    assert d.ratio == pytest.approx(0.8)


def test_rejects_surrogate_exploitation_and_shrinks():
    d = decide(1, -0.1)
    assert not d.accepted and d.radius == 0.2 and d.ratio < 0


def test_small_gain_has_no_ratio_and_no_acceptance():
    d = decide(0, 1)
    assert not d.accepted and d.ratio is None


def test_positive_but_insufficient_agreement_is_rejected():
    d = decide(1, 0.05)
    assert not d.accepted and d.reason == 'physical_improvement_not_confirmed'


def test_invalid_numerical_and_policy_inputs_fail_closed():
    with pytest.raises(ValueError):
        decide(float('nan'), 1)
    with pytest.raises(ValueError):
        decide(1, 1, shrink_factor=1)
