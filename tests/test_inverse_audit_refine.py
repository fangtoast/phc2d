"""Contracts for isolated inverse refinement audit cases.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_refine import run_case
from phc2d.inverse.audit_refined_reconcile import (rank_eligible_refinements, rank_refinements,
                                                   reconcile_refinements, summarize_eligible_stratified,
                                                   summarize_multiseed_eligible_stratified)
from phc2d.inverse.audit_refined_reconcile import rank_shared_eligible_refinements, shared_eligibility_from_predictions


def test_frozen_refinement_writes_numerical_evidence_and_refuses_overwrite(tmp_path):
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    manifest = {'rows': [{'structure': structure.to_mapping(), 'parameter_hash': structure.parameter_hash,
                          'source_index': 1, 'label': 'nominal'}],
                'k_points': [[0, 0], [.25, 0]], 'lower_band': 1, 'target_frequency': .5,
                'solver_config': {'num_bands': 2, 'resolution': 16, 'polarization': 'TE',
                                  'eigensolver_tolerance': 1e-7,
                                  'epsilon_air': structure.epsilon_air,
                                  'epsilon_material': structure.epsilon_material}}
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp, mh = save('manifest.json', manifest)
    op, oh = save('options.json', options)
    def solver(item, *, k_points, config):
        r = np.sum(np.asarray(k_points) ** 2, axis=1)
        return np.column_stack((.2 + .01*r, .7 - .01*r)), {'geometry_hash': item.geometry_hash}
    args = dict(expected_sha256=mh, refinement_path=op, expected_refinement_sha256=oh,
                case_index=0, output_root=tmp_path / 'out', solver=solver)
    result = json.loads((run_case(mp, **args) / 'results.json').read_text())
    assert result['rows'][0]['adaptive_refinement_converged']
    assert result['rows'][0]['resolution_stable']
    assert result['provenance']['comparison_contract_sha256']
    assert len(result['arrays']['k_points']) > 2
    assert result['solver_work'][0]['primary_k_points'] == len(result['arrays']['k_points'])
    # The reconciliation contract also requires the higher-resolution solver's
    # geometry identity; a mock solver supplies the same one for both passes.
    summary = reconcile_refinements(mp, [result], refinement_sha256=oh)
    assert summary['numerical_gate_passed_count'] == 1
    result['rows'][0]['resolution_stable'] = False
    assert reconcile_refinements(mp, [result], refinement_sha256=oh)['numerical_gate_passed_count'] == 0
    result['provenance']['verification_solver']['geometry_hash'] = 'wrong'
    with pytest.raises(ValueError, match='geometry identity'):
        reconcile_refinements(mp, [result], refinement_sha256=oh)
    with pytest.raises(FileExistsError):
        run_case(mp, **args)
    with pytest.raises(ValueError, match='SHA-256'):
        run_case(mp, **{**args, 'expected_refinement_sha256': 'wrong'})


def test_refined_ranking_keeps_failed_gates_visible():
    labels = ['anchor', 'left', 'right']
    proxy = [{'source_index': 1, 'label': label, 'parameter_hash': label,
              'status': 'surrogate_only', 'predicted_margin': value}
             for label, value in zip(labels, [0, 1, 2])]
    rows = [{'source_index': 1, 'label': label, 'parameter_hash': label,
             'target_margin': value, 'numerical_gate_passed': label != 'left'}
            for label, value in zip(labels, [0, 1, 3])]
    group = rank_refinements(proxy, {'status': 'exploratory_nominal_finite_local_refinement',
                                    'rows': rows})[0]
    assert group['selected_label'] == 'right'
    assert group['selected_gain'] == 3
    assert group['random_expected_gain'] == 2
    assert group['numerical_gate_passed_count'] == 2
    assert not group['all_numerical_gates_passed']


def test_eligible_ranking_keeps_full_physical_gate_and_uses_same_random_set():
    labels = ['anchor', 'a', 'b', 'c', 'd']
    proxy = [{'source_index': 1, 'label': label, 'parameter_hash': label,
              'status': 'surrogate_only' if score is not None else 'band_order_violation',
              'predicted_margin': score}
             for label, score in zip(labels, [0, 2, None, 1, None])]
    physical = [{'source_index': 1, 'label': label, 'parameter_hash': label,
                 'target_margin': value, 'numerical_gate_passed': label != 'b'}
                for label, value in zip(labels, [0, 4, 100, 0, -100])]
    summary = {'status': 'exploratory_finite_offsets_and_local_refinement', 'rows': physical}
    result = rank_eligible_refinements(proxy, summary)[0]
    assert result['status'] == 'ranked_descriptive' and result['eligible_candidate_count'] == 2
    assert result['candidate_count'] == 4 and len(result['unscoreable']) == 2
    assert not result['all_numerical_gates_passed'] and result['numerical_gate_passed_count'] == 4
    assert result['ranking']['selected_label'] == 'a'
    assert result['ranking']['random_expected_gain'] == 2
    assert result['random_baseline'] == 'uniform_over_scoreable_non_anchor_candidates'


def test_eligible_ranking_does_not_hide_unscoreable_anchors_or_missing_candidates():
    labels = ['anchor', 'a', 'b']
    proxy = [{'source_index': 1, 'label': label, 'parameter_hash': label,
              'status': 'surrogate_only', 'predicted_margin': 1.0} for label in labels]
    physical = [{'source_index': 1, 'label': label, 'parameter_hash': label,
                 'target_margin': 0.1, 'numerical_gate_passed': True} for label in labels]
    summary = {'status': 'exploratory_finite_offsets_and_local_refinement', 'rows': physical}
    proxy[0].update(status='band_order_violation', predicted_margin=None)
    result = rank_eligible_refinements(proxy, summary)[0]
    assert result['status'] == 'anchor_unscoreable' and result['ranking'] is None
    proxy[0].update(status='surrogate_only', predicted_margin=1.)
    proxy[1].update(status='band_order_violation', predicted_margin=None)
    assert rank_eligible_refinements(proxy, summary)[0]['status'] == 'insufficient_eligible_candidates'
    with pytest.raises(ValueError, match='identities differ'):
        rank_eligible_refinements(proxy[:-1], summary)
    proxy[1]['predicted_margin'] = 0.0
    with pytest.raises(ValueError, match='explicit failure'):
        rank_eligible_refinements(proxy, summary)
    proxy[1]['predicted_margin'] = None
    proxy[1]['predicted_gap'] = {'lower_edge': .4, 'upper_edge': .6}
    with pytest.raises(ValueError, match='explicit failure'):
        rank_eligible_refinements(proxy, summary)


def test_shared_eligibility_uses_identical_baselines_and_preserves_full_gate():
    labels = ['anchor', 'a', 'b', 'c', 'd']
    physical = [{'source_index': 1, 'label': label, 'parameter_hash': label,
                 'target_margin': value, 'numerical_gate_passed': label != 'a'}
                for label, value in zip(labels, [0., 100., -100., 1., 3.])]
    summary = {'status': 'exploratory_finite_offsets_and_local_refinement', 'rows': physical}
    def rows(invalid):
        return [{'source_index': 1, 'label': label, 'parameter_hash': label,
                 'status': 'band_order_violation' if label == invalid else 'surrogate_only',
                 'predicted_margin': None if label == invalid else value}
                for label, value in zip(labels, [0., 4., 3., 2., 1.])]
    proxies = {0: rows('a'), 1: rows('b')}
    prephysical = shared_eligibility_from_predictions(proxies, seeds=[0, 1])
    assert prephysical == {'common_nonanchor_identities': [(1, 'c', 'c'), (1, 'd', 'd')],
                           'unscoreable_anchor_sources': [], 'unscoreable_anchor_details': []}
    with pytest.raises(ValueError, match='rosters differ'):
        shared_eligibility_from_predictions({0: proxies[0], 1: proxies[1][:-1]}, seeds=[0, 1])
    result = rank_shared_eligible_refinements(proxies, summary, seeds=[0, 1])
    for seed in (0, 1):
        group = result[seed][0]
        assert group['eligible_candidate_identities'] == [(1, 'c', 'c'), (1, 'd', 'd')]
        assert group['ranking']['random_expected_gain'] == 2.
        assert not group['all_numerical_gates_passed'] and group['numerical_gate_passed_count'] == 4
        assert {item['status'] for item in group['unscoreable']} == {'band_order_violation', 'excluded_by_other_seed'}
    proxies[1][0].update(status='band_order_violation', predicted_margin=None)
    result = rank_shared_eligible_refinements(proxies, summary, seeds=[0, 1])
    assert all(result[seed][0]['status'] == 'anchor_unscoreable' and result[seed][0]['ranking'] is None
               for seed in (0, 1))
    assert result[0][0]['shared_anchor_failures'] == [
        {'source_index': 1, 'seed': 1, 'status': 'band_order_violation'}]
    with pytest.raises(ValueError, match='seed roster'):
        rank_shared_eligible_refinements(proxies, summary, seeds=[0, 2])


def test_stratified_summary_requires_frozen_roster_and_full_numerical_gates():
    strata = {1: 'positive', 2: 'positive', 3: 'negative', 4: 'negative'}
    kwargs = dict(source_strata=strata, expected_counts={'positive': 2, 'negative': 2},
                  stratum_weights={'positive': .5, 'negative': .5}, seed=7, bootstrap_samples=50)
    groups = [{'source_index': i, 'status': 'ranked_descriptive', 'all_numerical_gates_passed': True,
               'unscoreable': [],
               'ranking': {'source_index': i, 'selected_gain': float(i), 'random_expected_gain': 0.}}
              for i in strata]
    ready = summarize_eligible_stratified(groups, **kwargs)
    assert ready['status'] == 'ranking_summary_ready'
    assert ready['selection_scope'] == 'all_frozen_candidates_scoreable'
    assert ready['summary']['selected_minus_random_mean'] == 2.5
    groups[0]['unscoreable'] = [{'label': 'a', 'status': 'band_order_violation'}]
    conditional = summarize_eligible_stratified(groups, **kwargs)
    assert conditional['selection_scope'] == 'conditional_on_scoreable_candidates'
    assert conditional['unscoreable_by_source'][0]['source_index'] == 1
    groups[0]['all_numerical_gates_passed'] = False
    groups[1].update(status='anchor_unscoreable', ranking=None)
    unavailable = summarize_eligible_stratified(groups[:3], **kwargs)
    assert unavailable['summary'] is None
    assert [item['reason'] for item in unavailable['incomplete_sources']] == [
        'full_group_numerical_gate_failed', 'anchor_unscoreable', 'missing_group']
    with pytest.raises(ValueError, match='unique and frozen'):
        summarize_eligible_stratified(groups + [groups[0]], **kwargs)
    with pytest.raises(ValueError, match='stratum weights'):
        summarize_eligible_stratified(groups[:3], **{**kwargs, 'stratum_weights': {'positive': .5, 'negative': .4}})
    del groups[2]['unscoreable']
    with pytest.raises(ValueError, match='unscoreable'):
        summarize_eligible_stratified(groups, **kwargs)


def test_multiseed_summary_fails_closed_until_all_source_groups_are_gated():
    strata = {1: 'positive', 2: 'positive', 3: 'negative', 4: 'negative'}
    physical = {source: f'{source:064x}' for source in strata}
    groups = {seed: [{'source_index': source, 'status': 'ranked_descriptive',
                      'all_numerical_gates_passed': True, 'unscoreable': [],
                      'physical_group_sha256': physical[source],
                      'ranking': {'source_index': source, 'selected_gain': float(source+seed),
                                  'random_expected_gain': float(seed)}} for source in strata]
              for seed in (0, 1, 2)}
    kwargs = dict(source_strata=strata, seeds=[0, 1, 2], physical_group_sha256=physical,
                  expected_counts={'positive': 2, 'negative': 2},
                  stratum_weights={'positive': .5, 'negative': .5}, bootstrap_seed=3, bootstrap_samples=50)
    result = summarize_multiseed_eligible_stratified(groups, **kwargs)
    assert result['status'] == 'ranking_summary_ready'
    assert result['summary']['source_count'] == 4 and result['summary']['source_seed_count'] == 12
    assert result['summary']['selected_minus_random_mean'] == 2.5
    for seed in groups:
        for group in groups[seed]:
            source = group['source_index']
            group['eligible_candidate_identities'] = [[source, 'a', 'hash-a'], [source, 'b', 'hash-b']]
            group['eligible_candidate_count'] = 2
            group['ranking'].update(selected_label='a', selected_parameter_hash='hash-a')
    assert summarize_multiseed_eligible_stratified(groups, **kwargs, require_shared_eligible=True)['status'] == 'ranking_summary_ready'
    groups[1][0]['eligible_candidate_identities'][0][2] = 'different'
    with pytest.raises(ValueError, match='shared eligibility differs'):
        summarize_multiseed_eligible_stratified(groups, **kwargs, require_shared_eligible=True)
    groups[1][0]['eligible_candidate_identities'][0][2] = 'hash-a'
    groups[1][0]['ranking']['selected_parameter_hash'] = 'not-eligible'
    with pytest.raises(ValueError, match='outside shared eligibility'):
        summarize_multiseed_eligible_stratified(groups, **kwargs, require_shared_eligible=True)
    groups[1][0]['ranking']['selected_parameter_hash'] = 'hash-a'
    groups[0][1]['unscoreable'] = [{'label': 'a', 'status': 'band_order_violation'}]
    conditional = summarize_multiseed_eligible_stratified(groups, **kwargs)
    assert conditional['selection_scope'] == 'conditional_on_scoreable_candidates'
    assert conditional['per_seed'][0]['unscoreable_by_source'][0]['source_index'] == 2
    groups[0][1]['unscoreable'] = []
    groups[1][0]['all_numerical_gates_passed'] = False
    unavailable = summarize_multiseed_eligible_stratified(groups, **kwargs)
    assert unavailable['summary'] is None
    assert unavailable['incomplete_source_seeds'] == [
        {'seed': 1, 'source_index': 1, 'reason': 'full_group_numerical_gate_failed'}]
    with pytest.raises(ValueError, match='rosters must be complete'):
        summarize_multiseed_eligible_stratified(groups, **{**kwargs, 'bootstrap_samples': 0})
    groups[1][0]['all_numerical_gates_passed'] = True
    shortened = {**groups, 0: groups[0][:-1]}
    assert summarize_multiseed_eligible_stratified(shortened, **kwargs)['incomplete_source_seeds'] == [
        {'seed': 0, 'source_index': 4, 'reason': 'missing_group'}]
    groups[2][0]['physical_group_sha256'] = physical[2]
    with pytest.raises(ValueError, match='physical identity differs'):
        summarize_multiseed_eligible_stratified(groups, **kwargs)
