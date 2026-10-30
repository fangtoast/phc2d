"""Test inverse-design denominators and physical evidence boundaries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.design_metrics import summarize_design_outcomes


def verified(lower, upper):
    return {'status': 'verified', 'numerical_gate_passed': True, 'artifact_sha256': 'a' * 64,
            'lower_edge': lower, 'upper_edge': upper}


def planned(*targets):
    return [{'target_id': target, 'seed': 0, 'target_frequency': 1.0} for target in targets]


def test_missing_and_unresolved_do_not_inflate_success_or_false_positive():
    positive = {'checkpoint_sha256': 'b' * 64, 'target_margin': .1}
    records = [
        {'target_id': 'a', 'seed': 0, 'nominal': verified(.8, 1.2), 'unseen': verified(.9, 1.1),
         'prediction': positive},
        {'target_id': 'b', 'seed': 0, 'nominal': verified(1.1, .9),
         'unseen': {'status': 'failed', 'reason': 'solver failed'}, 'prediction': positive},
        {'target_id': 'c', 'seed': 0, 'nominal': {'status': 'unresolved', 'reason': 'resolution'},
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
    records = [{'target_id': 'a', 'seed': 0, 'nominal': {'status': 'not_run'},
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
    record = {'target_id': 'a', 'seed': 0, 'nominal': verified(.8, 1.2), 'unseen': {'status': 'not_run'}}
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
