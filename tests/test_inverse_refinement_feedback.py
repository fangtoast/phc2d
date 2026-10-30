"""Tests for numerically gated inverse optimization feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.refinement_feedback import (aggregate_refined_perturbations, assess_refinement_group,
                                               compare_refined_candidates)


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


def perturbation(offset, lo, hi, *, stable=True, parent='p'):
    value = observation(lo, hi, stable=stable)
    value['offset'] = offset
    value['provenance'].update(parameter_hash=parent, evaluated_geometry='periodic_binary_pixel_cells_v1',
                               evaluated_grid_sha256=str(offset))
    return value


def test_common_interval_uses_worst_edges_and_frozen_offsets():
    a = aggregate_refined_perturbations([perturbation(-.1, .3, .7), perturbation(0, .2, .8),
                                        perturbation(.1, .4, .6)], expected_offsets=(-.1, 0, .1))
    assert a['status'] == 'numerically_stable'
    assert a['row']['refined_lower_edge_frequency'] == .4
    assert a['row']['refined_upper_edge_frequency'] == .6
    assert compare(a, a)['status'] == 'compared'


def test_unstable_or_missing_perturbations_cannot_be_promoted():
    records = [perturbation(0, .2, .8), perturbation(.1, .4, .6, stable=False)]
    a = aggregate_refined_perturbations(records, expected_offsets=(0, .1))
    assert a['status'] == 'numerical_gate_failed'
    assert compare(a, a)['status'] == 'numerical_gate_failed'
    with pytest.raises(ValueError, match='expected offsets'):
        aggregate_refined_perturbations(records, expected_offsets=(0, .1, .2))


def test_group_escalation_requires_all_members_at_one_contract():
    roster = {'anchor': 'a', 'candidate': 'b'}
    anchor = observation(.4, .6)
    anchor['provenance']['parameter_hash'] = 'a'
    candidate = observation(.35, .65, stable=False)
    candidate['provenance']['parameter_hash'] = 'b'
    args = dict(expected_members=roster, comparison_contract_sha256='same', stage_index=0, stage_count=2)
    waiting = assess_refinement_group({'anchor': anchor}, **args)
    assert waiting['status'] == 'pending' and waiting['pending_members'] == ['candidate']
    result = assess_refinement_group({'anchor': anchor, 'candidate': candidate}, **args)
    assert result['status'] == 'group_escalation_required' and result['next_stage'] == 1
    assert result['reevaluate_members'] == ['anchor', 'candidate']
    assert result['unstable_members'] == ['candidate']
    exhausted = assess_refinement_group({'anchor': anchor, 'candidate': candidate},
                                        **dict(args, stage_index=1))
    assert exhausted['status'] == 'numerically_unresolved' and exhausted['next_stage'] is None
    candidate['row'].update(last_refinement_stable=True, resolution_stable=True,
                            adaptive_refinement_converged=True)
    assert assess_refinement_group({'anchor': anchor, 'candidate': candidate}, **args)['status'] == 'admissible'


def test_group_rejects_mixed_identity_contract_and_missing_gates():
    roster = {'a': 'a', 'b': 'b'}
    a = observation(.4, .6)
    a['provenance']['parameter_hash'] = 'a'
    b = observation(.4, .6, signature='other')
    b['provenance']['parameter_hash'] = 'b'
    args = dict(expected_members=roster, comparison_contract_sha256='same', stage_index=0, stage_count=2)
    with pytest.raises(ValueError, match='identity or numerical contract'):
        assess_refinement_group({'a': a, 'b': b}, **args)
    b['provenance']['comparison_contract_sha256'] = 'same'
    b['provenance']['parameter_hash'] = 'wrong'
    with pytest.raises(ValueError, match='identity or numerical contract'):
        assess_refinement_group({'a': a, 'b': b}, **args)
    b['provenance']['parameter_hash'] = 'b'
    del b['row']['resolution_stable']
    with pytest.raises(ValueError, match='explicit booleans'):
        assess_refinement_group({'a': a, 'b': b}, **args)
    with pytest.raises(ValueError, match='roster'):
        assess_refinement_group({'unexpected': a}, **args)


def test_group_solver_failure_is_not_automatically_escalated():
    roster = {'a': 'a', 'b': 'b'}
    a = observation(.4, .6)
    a['provenance']['parameter_hash'] = 'a'
    result = assess_refinement_group({'a': a, 'b': None}, expected_members=roster,
        comparison_contract_sha256='same', stage_index=0, stage_count=2)
    assert result['status'] == 'solver_failure' and result['failed_members'] == ['b']
    assert result['next_stage'] is None and result['reevaluate_members'] == []
