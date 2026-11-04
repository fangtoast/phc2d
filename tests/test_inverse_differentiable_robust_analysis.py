"""Verify matched robust outcomes preserve missing physical results."""
import copy

import pytest

from phc2d.inverse.differentiable_robust_analysis import analyze_robust_comparison


def test_paired_robust_summary_excludes_missing_from_continuous_metrics():
    def row(label, value):
        return {'label': label, 'status': 'numerically_stable', 'physical_common_target_margin': value,
                'physical_nominal_target_margin': value + .1, 'physical_contains_target': value > 0,
                'physical_common_edges': [.4, .6]}
    pure = {'scope': 'locked_robust_surrogate_vs_physical', 'roles': [
        row('source_0008_initial_0_'+mode+'/top1', value)
        for mode, value in [('nominal', -.2), ('mean', -.1), ('worst', -.05)]]}
    corrected = copy.deepcopy(pure)
    for item in corrected['roles']:
        item['label'] = item['label'].replace('/top1', '_corrected/top1')
        item['physical_common_target_margin'] += .02
    corrected['roles'][1] = {'label': corrected['roles'][1]['label'], 'status': 'physical_result_missing'}
    result = analyze_robust_comparison(pure, corrected)
    assert result['groups']['pure']['worst']['mean_common_margin'] == pytest.approx(-.05)
    assert result['corrected_minus_pure']['worst']['mean_common_change'] == pytest.approx(.02)
    assert result['corrected_minus_pure']['mean']['count'] == 0
    assert result['groups']['corrected']['mean']['registered'] == 1
    assert result['groups']['corrected']['mean']['mean_common_margin'] is None
