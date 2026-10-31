"""Validate and summarize finite-grid inverse-audit refinement records.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_ranking import (paired_candidate_ranking, summarize_stratified_paired_gains,
                                        summarize_seed_averaged_stratified_gains)


def reconcile_refinements(manifest_path: str | Path, records: Sequence[Mapping[str, Any]], *,
                          refinement_sha256: str) -> dict[str, Any]:
    """Validate all frozen candidates and retain both converged and failed gates.

    These are nominal, locally refined finite-grid results, not manufacturing
    robustness or continuous-Brillouin-zone certification.
    """
    raw = Path(manifest_path).read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    expected = manifest['rows']
    by_index = {r['case_index']: r for r in records}
    if len(records) != len(expected) or len(by_index) != len(expected) or set(by_index) != set(range(len(expected))):
        raise ValueError('incomplete or duplicate refinement matrix')
    target = float(manifest['target_frequency'])
    if not math.isfinite(target) or target <= 0:
        raise ValueError('invalid target frequency')
    signatures = set()
    results = []
    for index, source in enumerate(expected):
        record = by_index[index]
        structure = FullCellStructureSpec.from_mapping(source['structure'])
        if (record.get('manifest_sha256') != manifest_sha or record.get('refinement_sha256') != refinement_sha256
                or record.get('source_index') != source['source_index'] or record.get('label') != source['label']
                or record.get('parameter_hash') != source['parameter_hash'] or structure.parameter_hash != source['parameter_hash']):
            raise ValueError('candidate or input identity mismatch')
        provenance = record['provenance']
        if (provenance.get('parameter_hash') != structure.parameter_hash
                or provenance.get('initial_solver', {}).get('geometry_hash') != structure.geometry_hash
                or provenance.get('verification_solver', {}).get('geometry_hash') != structure.geometry_hash
                or provenance.get('evaluated_geometry') != 'analytic'):
            raise ValueError('solver geometry identity mismatch')
        signature = provenance.get('comparison_contract_sha256')
        if not signature:
            raise ValueError('missing refinement comparison contract')
        signatures.add(signature)
        if len(record['rows']) != 1:
            raise ValueError('expected one band pair per candidate')
        row = record['rows'][0]
        if row['lower_band_one_based'] != manifest['lower_band']:
            raise ValueError('band-pair identity mismatch')
        lower, upper = float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError('invalid refined edges')
        stable = all(row.get(key) is True for key in
                     ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged'))
        results.append({'case_index': index, 'source_index': source['source_index'], 'label': source['label'],
                        'parameter_hash': structure.parameter_hash, 'lower_edge': lower, 'upper_edge': upper,
                        'target_margin': 2 * min(target - lower, upper - target) / target,
                        'numerical_gate_passed': stable})
    if len(signatures) != 1:
        raise ValueError('refinement comparison contracts differ')
    return {'status': 'exploratory_nominal_finite_local_refinement', 'manifest_sha256': manifest_sha,
            'refinement_sha256': refinement_sha256, 'comparison_contract_sha256': next(iter(signatures)),
            'candidate_count': len(results), 'numerical_gate_passed_count': sum(r['numerical_gate_passed'] for r in results),
            'rows': results}


def rank_refinements(proxy_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank paired candidates while retaining whether every group passed its gates.

    Caller must independently verify proxy checkpoint and dataset provenance.
    Numerical failures remain in the descriptive ranking and are marked;
    selecting only passing groups can introduce selection bias.
    """
    scope = summary.get('status')
    if scope not in {'exploratory_nominal_finite_local_refinement',
                     'exploratory_finite_offsets_and_local_refinement'}:
        raise ValueError('unexpected refinement summary status')
    rows = summary['rows']
    # The shared paired ranking consumes the same signed-margin contract.
    inputs = [{'source_index': r['source_index'], 'label': r['label'],
               'parameter_hash': r['parameter_hash'], 'status': 'sampled_solver_only',
               'gap': {'target_margin': r['target_margin']}} for r in rows]
    ranked = paired_candidate_ranking(proxy_rows, inputs)
    for result in ranked:
        group = [r for r in rows if r['source_index'] == result['source_index']]
        result['numerical_gate_passed_count'] = sum(r['numerical_gate_passed'] is True for r in group)
        result['all_numerical_gates_passed'] = result['numerical_gate_passed_count'] == len(group)
        result['evidence_scope'] = scope
    return ranked


def rank_eligible_refinements(proxy_rows: Sequence[Mapping[str, Any]],
                              summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank scoreable candidates without silently dropping incomplete groups.

    The physical roster must contain every candidate, including unscoreable
    ones. A group needs one scoreable anchor and at least two scoreable
    non-anchor candidates; its random baseline uses that same eligible set.
    Numerical-gate coverage is always reported over the *full* group, not just
    eligible candidates. This is descriptive: callers validate checkpoint,
    physical artifacts and the frozen roster independently.
    """
    scope = summary.get('status')
    if scope not in {'exploratory_nominal_finite_local_refinement',
                     'exploratory_finite_offsets_and_local_refinement'}:
        raise ValueError('unexpected refinement summary status')
    def identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return row['source_index'], row['label'], row['parameter_hash']
    physical_rows = summary['rows']
    predicted = {identity(row): row for row in proxy_rows}
    physical = {identity(row): row for row in physical_rows}
    if (not predicted or len(predicted) != len(proxy_rows) or len(physical) != len(physical_rows)
            or predicted.keys() != physical.keys()):
        raise ValueError('paired candidate identities differ or are duplicated')
    groups: dict[Any, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for key, prediction in predicted.items():
        truth = physical[key]
        score = prediction.get('predicted_margin')
        if prediction.get('status') == 'surrogate_only':
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('scoreable candidate needs a finite prediction')
        elif (prediction.get('status') not in {'band_order_violation', 'nonfinite_prediction', 'negative_frequency'}
              or score is not None or prediction.get('predicted_gap') is not None):
            raise ValueError('unscoreable candidate needs an explicit failure and no score')
        if (not isinstance(truth.get('numerical_gate_passed'), bool)
                or isinstance(truth.get('target_margin'), bool)
                or not isinstance(truth.get('target_margin'), (int, float))
                or not math.isfinite(truth['target_margin'])):
            raise ValueError('physical candidate needs a finite margin and explicit gate')
        groups.setdefault(key[0], []).append((prediction, truth))
    result = []
    for source, entries in sorted(groups.items()):
        anchors = [pair for pair in entries if pair[0]['label'] == 'anchor']
        candidates = [pair for pair in entries if pair[0]['label'] != 'anchor']
        if len(anchors) != 1 or len(candidates) < 2:
            raise ValueError('each source needs one anchor and at least two candidates')
        eligible = [pair for pair in candidates if pair[0]['status'] == 'surrogate_only']
        missing = [{'label': pair[0]['label'], 'status': pair[0]['status']}
                   for pair in entries if pair[0]['status'] != 'surrogate_only']
        row = {'source_index': source, 'candidate_count': len(candidates),
               'eligible_candidate_count': len(eligible), 'unscoreable': missing,
               'numerical_gate_passed_count': sum(pair[1]['numerical_gate_passed'] for pair in entries),
               'all_numerical_gates_passed': all(pair[1]['numerical_gate_passed'] for pair in entries),
               'evidence_scope': scope, 'random_baseline': 'uniform_over_scoreable_non_anchor_candidates'}
        if anchors[0][0]['status'] != 'surrogate_only':
            row.update(status='anchor_unscoreable', ranking=None)
        elif len(eligible) < 2:
            row.update(status='insufficient_eligible_candidates', ranking=None)
        else:
            pairs = anchors + eligible
            solver = [{'source_index': p['source_index'], 'label': p['label'],
                       'parameter_hash': p['parameter_hash'], 'status': 'sampled_solver_only',
                       'gap': {'target_margin': t['target_margin']}} for p, t in pairs]
            ranked = paired_candidate_ranking([p for p, _ in pairs], solver)
            if len(ranked) != 1:
                raise ValueError('eligible comparison must produce one source group')
            row.update(status='ranked_descriptive', ranking=ranked[0])
        result.append(row)
    return result


def summarize_eligible_stratified(
    groups: Sequence[Mapping[str, Any]], *, source_strata: Mapping[int, str],
    expected_counts: Mapping[str, int], stratum_weights: Mapping[str, float],
    seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Summarize only a complete frozen roster with full-group numerical gates.

    The caller validates the frozen inputs, solver evidence, checkpoint and
    perturbation provenance. A ready summary is not physical acceptance;
    exclusions make it conditional on the scoreable candidate subset.
    """
    # Validate the entire frozen statistical contract even when observations
    # are incomplete; fabricated values are never used in the returned result.
    dummy = [{'source_index': source, 'selected_gain': 0., 'random_expected_gain': 0.}
             for source in source_strata]
    summarize_stratified_paired_gains(dummy, source_strata=source_strata,
        expected_counts=expected_counts, stratum_weights=stratum_weights,
        seed=seed, bootstrap_samples=1)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be a positive integer')
    by_source = {}
    for group in groups:
        source = group.get('source_index')
        if isinstance(source, bool) or not isinstance(source, int) or source in by_source or source not in source_strata:
            raise ValueError('group source identities must be unique and frozen')
        if not isinstance(group.get('unscoreable'), list):
            raise ValueError('each group must retain its unscoreable candidates')
        by_source[source] = group
    exclusions = [{'source_index': source, 'candidates': group['unscoreable']}
                  for source, group in sorted(by_source.items()) if group['unscoreable']]
    incomplete = []
    for source in sorted(source_strata):
        group = by_source.get(source)
        if group is None:
            incomplete.append({'source_index': source, 'reason': 'missing_group'})
        elif group.get('status') != 'ranked_descriptive' or group.get('ranking') is None:
            incomplete.append({'source_index': source, 'reason': str(group.get('status', 'missing_status'))})
        elif group.get('all_numerical_gates_passed') is not True:
            incomplete.append({'source_index': source, 'reason': 'full_group_numerical_gate_failed'})
        elif group['ranking'].get('source_index') != source:
            raise ValueError('ranking source identity mismatch')
    if incomplete:
        return {'status': 'ranking_summary_unavailable', 'expected_source_count': len(source_strata),
                'incomplete_sources': incomplete, 'unscoreable_by_source': exclusions, 'summary': None}
    result = summarize_stratified_paired_gains([by_source[s]['ranking'] for s in sorted(source_strata)],
        source_strata=source_strata, expected_counts=expected_counts, stratum_weights=stratum_weights,
        seed=seed, bootstrap_samples=bootstrap_samples)
    return {'status': 'ranking_summary_ready', 'expected_source_count': len(source_strata),
            'incomplete_sources': [], 'unscoreable_by_source': exclusions,
            'selection_scope': ('conditional_on_scoreable_candidates' if exclusions
                                else 'all_frozen_candidates_scoreable'), 'summary': result}


def summarize_multiseed_eligible_stratified(
    groups_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], *, source_strata: Mapping[int, str],
    seeds: Sequence[int], physical_group_sha256: Mapping[int, str], expected_counts: Mapping[str, int],
    stratum_weights: Mapping[str, float], bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Apply full-group gates before averaging seeds and resampling sources.

    Input groups must derive from independently validated solver evidence and
    locked model choices. Digests identify the same physical group across
    seeds; their authenticity and candidate eligibility remain caller duties.
    Missing or unresolved source/seed groups prevent the summary, never cause
    implicit source deletion. A ready summary is descriptive, not acceptance.
    """
    if (not isinstance(groups_by_seed, Mapping) or not isinstance(physical_group_sha256, Mapping)
            or not isinstance(source_strata, Mapping)
            or not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds)
            or any(isinstance(x, bool) or not isinstance(x, int) for x in groups_by_seed)
            or len(set(seeds)) != len(seeds) or set(groups_by_seed) != set(seeds)
            or set(physical_group_sha256) != set(source_strata)
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1):
        raise ValueError('frozen seed and physical group rosters must be complete')
    digests = list(physical_group_sha256.values())
    if (any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x)
            for x in digests) or len(set(digests)) != len(digests)):
        raise ValueError('unique physical group digests are required')
    per_seed = {}
    incomplete = []
    for model_seed in seeds:
        if (not isinstance(groups_by_seed[model_seed], Sequence)
                or isinstance(groups_by_seed[model_seed], (str, bytes))
                or any(not isinstance(group, Mapping) for group in groups_by_seed[model_seed])):
            raise ValueError('each seed needs a sequence of candidate groups')
        for group in groups_by_seed[model_seed]:
            source = group.get('source_index')
            if (isinstance(source, bool) or not isinstance(source, int) or source not in physical_group_sha256
                    or group.get('physical_group_sha256') != physical_group_sha256[source]):
                raise ValueError('seed group physical identity differs from the frozen source')
        result = summarize_eligible_stratified(groups_by_seed[model_seed], source_strata=source_strata,
            expected_counts=expected_counts, stratum_weights=stratum_weights,
            seed=bootstrap_seed, bootstrap_samples=1)
        per_seed[model_seed] = result
        incomplete.extend({'seed': model_seed, **item} for item in result['incomplete_sources'])
    if incomplete:
        return {'status': 'ranking_summary_unavailable', 'incomplete_source_seeds': incomplete,
                'per_seed': per_seed, 'summary': None}
    rows = []
    for model_seed in seeds:
        for group in groups_by_seed[model_seed]:
            source = group['source_index']
            rows.append({**group['ranking'], 'seed': model_seed,
                         'physical_group_sha256': physical_group_sha256[source]})
    summary = summarize_seed_averaged_stratified_gains(rows, source_strata=source_strata, seeds=seeds,
        expected_counts=expected_counts, stratum_weights=stratum_weights,
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    conditional = any(result['unscoreable_by_source'] for result in per_seed.values())
    return {'status': 'ranking_summary_ready', 'incomplete_source_seeds': [], 'per_seed': per_seed,
            'selection_scope': ('conditional_on_scoreable_candidates' if conditional
                                else 'all_frozen_candidates_scoreable'), 'summary': summary}
