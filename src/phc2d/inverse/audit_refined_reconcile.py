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
                              summary: Mapping[str, Any], *,
                              shared_eligible: set[tuple[Any, Any, Any]] | None = None) -> list[dict[str, Any]]:
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
    if shared_eligible is not None:
        if not isinstance(shared_eligible, set) or not shared_eligible.issubset(predicted):
            raise ValueError('shared eligibility must be a subset of paired identities')
        if any(predicted[key]['label'] == 'anchor' or predicted[key].get('status') != 'surrogate_only'
               for key in shared_eligible):
            raise ValueError('shared eligible identities must be scoreable nonanchors')
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
        eligible = [pair for pair in candidates if pair[0]['status'] == 'surrogate_only'
                    and (shared_eligible is None or identity(pair[0]) in shared_eligible)]
        missing = [{'label': pair[0]['label'], 'status': pair[0]['status']}
                   for pair in entries if pair[0]['status'] != 'surrogate_only']
        if shared_eligible is not None:
            missing.extend({'label': pair[0]['label'], 'status': 'excluded_by_other_seed'}
                           for pair in candidates if pair[0]['status'] == 'surrogate_only'
                           and identity(pair[0]) not in shared_eligible)
        row = {'source_index': source, 'candidate_count': len(candidates),
               'eligible_candidate_count': len(eligible), 'unscoreable': missing,
               'eligible_candidate_identities': sorted((identity(pair[0]) for pair in eligible)),
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


def shared_eligibility_from_predictions(
    proxy_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], *, seeds: Sequence[int],
) -> dict[str, Any]:
    """Compute shared identities without reading any physical outcomes.

    Caller independently validates every prediction, frozen roster and hash.
    """
    if (not isinstance(proxy_by_seed, Mapping) or not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes)) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or set(proxy_by_seed) != set(seeds)):
        raise ValueError('shared ranking needs the frozen seed roster')
    def identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return row['source_index'], row['label'], row['parameter_hash']
    by_seed = {}
    for seed in seeds:
        rows = proxy_by_seed[seed]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ValueError('each seed needs a nonempty prediction roster')
        indexed = {identity(row): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate prediction identity')
        for row in rows:
            score = row.get('predicted_margin')
            if row.get('status') == 'surrogate_only':
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError('scoreable candidate needs a finite prediction')
            elif (row.get('status') not in {'band_order_violation', 'nonfinite_prediction', 'negative_frequency'}
                  or score is not None or row.get('predicted_gap') is not None):
                raise ValueError('unscoreable candidate needs an explicit failure and no score')
        by_seed[seed] = indexed
    keys = set(by_seed[seeds[0]])
    if any(set(by_seed[seed]) != keys for seed in seeds):
        raise ValueError('prediction candidate rosters differ')
    sources = {key[0] for key in keys}
    if any(sum(key[0] == source and key[1] == 'anchor' for key in keys) != 1 for source in sources):
        raise ValueError('each source needs exactly one anchor')
    common = {key for key in keys if key[1] != 'anchor'
              and all(by_seed[seed][key]['status'] == 'surrogate_only' for seed in seeds)}
    invalid_anchor_sources = {key[0] for key in keys if key[1] == 'anchor'
                              and any(by_seed[seed][key]['status'] != 'surrogate_only' for seed in seeds)}
    anchor_failures = [{'source_index': key[0], 'seed': seed, 'status': by_seed[seed][key]['status']}
                       for key in sorted(keys) if key[1] == 'anchor' for seed in seeds
                       if by_seed[seed][key]['status'] != 'surrogate_only']
    return {'common_nonanchor_identities': sorted(common), 'unscoreable_anchor_sources': sorted(invalid_anchor_sources),
            'unscoreable_anchor_details': anchor_failures}


def rank_shared_eligible_refinements(
    proxy_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], summary: Mapping[str, Any], *,
    seeds: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    """Rank each seed against a common valid set and gate the full physical roster."""
    shared = shared_eligibility_from_predictions(proxy_by_seed, seeds=seeds)
    common = set(shared['common_nonanchor_identities'])
    invalid_anchor_sources = set(shared['unscoreable_anchor_sources'])
    output = {}
    for seed in seeds:
        groups = rank_eligible_refinements(proxy_by_seed[seed], summary, shared_eligible=common)
        for group in groups:
            group['random_baseline'] = 'uniform_over_shared_scoreable_non_anchor_candidates'
            if group['source_index'] in invalid_anchor_sources:
                group.update(status='anchor_unscoreable', ranking=None,
                             shared_anchor_failures=[item for item in shared['unscoreable_anchor_details']
                                                     if item['source_index'] == group['source_index']])
        output[seed] = groups
    return output


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
    require_shared_eligible: bool = False,
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
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1
            or not isinstance(require_shared_eligible, bool)):
        raise ValueError('frozen seed and physical group rosters must be complete')
    digests = list(physical_group_sha256.values())
    if (any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x)
            for x in digests) or len(set(digests)) != len(digests)):
        raise ValueError('unique physical group digests are required')
    shared_by_source: dict[int, tuple[tuple[Any, Any, Any], ...]] = {}
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
            if require_shared_eligible:
                identities = group.get('eligible_candidate_identities')
                if (not isinstance(identities, (list, tuple))
                        or any(not isinstance(item, (list, tuple)) or len(item) != 3
                               or item[0] != source or not isinstance(item[1], str)
                               or not isinstance(item[2], str) or item[1] == 'anchor' for item in identities)):
                    raise ValueError('shared eligibility identities are invalid')
                canonical = tuple(sorted(tuple(item) for item in identities))
                if len(set(canonical)) != len(canonical) or len(canonical) != group.get('eligible_candidate_count'):
                    raise ValueError('shared eligibility count differs')
                if source in shared_by_source and shared_by_source[source] != canonical:
                    raise ValueError('shared eligibility differs across seeds')
                shared_by_source[source] = canonical
                if group.get('status') == 'ranked_descriptive':
                    ranking = group.get('ranking')
                    if (not isinstance(ranking, Mapping) or len(canonical) < 2
                            or (source, ranking.get('selected_label'), ranking.get('selected_parameter_hash')) not in canonical):
                        raise ValueError('selected candidate is outside shared eligibility')
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


def evaluate_locked_shared_choices(
    choices: Mapping[str, Any], manifest: Mapping[str, Any], physical: Mapping[str, Any], *,
    source_strata: Mapping[int, str], seeds: Sequence[int], physical_group_sha256: Mapping[int, str],
    expected_counts: Mapping[str, int], stratum_weights: Mapping[str, float],
    bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Evaluate pre-locked choices on independently verified physical records.

    The caller must verify lock chronology, hashes, predictions and solver
    provenance. Missing observations or any failed full-group gate prevent a
    summary. A statistical result alone is not physical acceptance.
    """
    entries = manifest['rows']
    measured = physical['rows']
    expected = {i: row for i, row in enumerate(entries)}
    observed = {row['case_index']: row for row in measured}
    if len(observed) != len(measured) or set(observed) - set(expected):
        raise ValueError('duplicate or unexpected physical candidate')
    if {row['source_index'] for row in entries} != set(source_strata):
        raise ValueError('manifest source roster differs')
    for index, row in observed.items():
        reference = expected[index]
        if any(row.get(k) != reference.get(k) for k in ('source_index', 'label', 'parameter_hash')):
            raise ValueError('physical candidate identity differs')
        value = row.get('target_margin')
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not isinstance(row.get('numerical_gate_passed'), bool)):
            raise ValueError('invalid physical outcome')
    locked = choices['sources']
    by_pair = {(row['source_index'], row['seed']): row for row in locked}
    roster = {(source, seed) for source in source_strata for seed in seeds}
    if len(by_pair) != len(locked) or set(by_pair) - roster:
        raise ValueError('duplicate or unexpected locked choice')
    modes = {'raw': {seed: [] for seed in seeds}, 'corrected': {seed: [] for seed in seeds}}
    for source, seed in sorted(roster):
        items = [(i, row) for i, row in expected.items() if row['source_index'] == source]
        anchors = [i for i, row in items if row['label'] == 'anchor']
        if len(anchors) != 1:
            raise ValueError('source requires exactly one anchor')
        choice = by_pair.get((source, seed))
        if choice is None:
            continue
        if choice.get('anchor_case_index') != anchors[0]:
            raise ValueError('locked anchor differs')
        status = choice.get('status')
        allowed = {'choices_locked', 'anchor_unscoreable', 'anchor_numerical_gate_failed',
                   'insufficient_shared_candidates'}
        if status not in allowed:
            raise ValueError('unknown locked choice status')
        eligible = choice.get('eligible', [])
        identities = []
        for candidate in eligible:
            index = candidate['case_index']
            if index not in expected or expected[index]['source_index'] != source or expected[index]['label'] == 'anchor':
                raise ValueError('eligible candidate outside source')
            if any(candidate.get(k) != expected[index].get(k) for k in ('label', 'parameter_hash')):
                raise ValueError('locked candidate identity differs')
            for score in ('raw_score', 'corrected_score'):
                value = candidate.get(score)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('locked score must be finite')
            identities.append((source, candidate['label'], candidate['parameter_hash']))
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate eligible candidate')
        if status == 'choices_locked':
            if len(eligible) < 2:
                raise ValueError('locked choice requires two eligible candidates')
            for mode in modes:
                winner = min(eligible, key=lambda x: (-x[f'{mode}_score'], x['label'], x['parameter_hash']))
                if choice.get(f'{mode}_choice') != winner['case_index']:
                    raise ValueError('locked winner differs from scores')
        complete = all(index in observed for index, _ in items)
        passed = complete and all(observed[index]['numerical_gate_passed'] for index, _ in items)
        for mode, groups in modes.items():
            ranking = None
            group_status = status
            if status == 'choices_locked' and complete:
                anchor = observed[anchors[0]]['target_margin']
                selected = observed[choice[f'{mode}_choice']]
                baseline = sum(observed[item['case_index']]['target_margin'] for item in eligible) / len(eligible)
                ranking = {'source_index': source, 'selected_label': selected['label'],
                           'selected_parameter_hash': selected['parameter_hash'],
                           'selected_gain': selected['target_margin'] - anchor,
                           'random_expected_gain': baseline - anchor}
                group_status = 'ranked_descriptive'
            elif status == 'choices_locked':
                group_status = 'missing_physical_candidate'
            groups[seed].append({'source_index': source, 'physical_group_sha256': physical_group_sha256[source],
                'status': group_status, 'ranking': ranking, 'all_numerical_gates_passed': passed,
                'eligible_candidate_identities': identities, 'eligible_candidate_count': len(identities),
                'unscoreable': [row['label'] for _, row in items
                                if row['label'] != 'anchor' and (source, row['label'], row['parameter_hash']) not in identities]})
    result = {mode: summarize_multiseed_eligible_stratified(groups, source_strata=source_strata, seeds=seeds,
              physical_group_sha256=physical_group_sha256, expected_counts=expected_counts,
              stratum_weights=stratum_weights, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
              require_shared_eligible=True) for mode, groups in modes.items()}
    return {'status': 'ready' if all(value['status'] == 'ranking_summary_ready' for value in result.values())
            else 'unavailable', 'physical_acceptance': 'not_evaluated', **result}


def adjudicate_locked_choice_development(
    evaluation: Mapping[str, Any], crosscheck: Mapping[str, Any] | None, *,
    bootstrap_seed: int, bootstrap_samples: int, model_seeds: Sequence[int],
    expected_source_count: int, expected_case_indices: Sequence[int], expected_offsets: Sequence[float],
    frequency_tolerance: float,
) -> dict[str, Any]:
    """Apply a predeclared statistical and finite-grid crosscheck criterion.

    Callers must authenticate the frozen protocol, all input artifacts and
    the calculations producing these summaries. A pass is conditional on
    that validation and does not decide the wider P2 gate or formal success.
    """
    if (isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1
            or not isinstance(model_seeds, Sequence) or isinstance(model_seeds, (str, bytes)) or not model_seeds
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in model_seeds)
            or len(set(model_seeds)) != len(model_seeds)
            or isinstance(expected_source_count, bool) or not isinstance(expected_source_count, int)
            or expected_source_count < 1
            or not isinstance(expected_case_indices, Sequence) or isinstance(expected_case_indices, (str, bytes))
            or not expected_case_indices or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                                for x in expected_case_indices)
            or len(set(expected_case_indices)) != len(expected_case_indices)
            or not isinstance(expected_offsets, Sequence) or isinstance(expected_offsets, (str, bytes))
            or not expected_offsets or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                           or not math.isfinite(x) for x in expected_offsets)
            or len(set(expected_offsets)) != len(expected_offsets)
            or isinstance(frequency_tolerance, bool) or not isinstance(frequency_tolerance, (int, float))
            or not math.isfinite(frequency_tolerance) or frequency_tolerance < 0):
        raise ValueError('invalid frozen decision contract')
    result = evaluation.get('locked_choice_evaluation')
    for key in ('manifest_sha256', 'archive_sha256'):
        value = evaluation.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('evaluation artifact digest required')
    if not isinstance(result, Mapping) or result.get('status') not in {'ready', 'unavailable'}:
        raise ValueError('invalid locked-choice evaluation')
    stat = 'unavailable'
    if result['status'] == 'ready':
        corrected = result.get('corrected')
        if not isinstance(corrected, Mapping) or corrected.get('status') != 'ranking_summary_ready':
            raise ValueError('ready evaluation needs a complete corrected summary')
        summary = corrected.get('summary')
        if (not isinstance(summary, Mapping) or summary.get('seed') != bootstrap_seed
                or summary.get('bootstrap_samples') != bootstrap_samples or summary.get('model_seeds') != list(model_seeds)
                or summary.get('source_seed_count') != expected_source_count * len(model_seeds)
                or summary.get('resampling_unit') != 'source_within_stratum'):
            raise ValueError('statistical summary differs from frozen contract')
        values = [summary.get(key) for key in ('selected_minus_random_mean', 'bootstrap_95_low', 'bootstrap_95_high')]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values) or values[1] > values[2]:
            raise ValueError('invalid statistical interval')
        stat = 'pass' if values[0] > 0 and values[1] > 0 else 'fail'
    physical = 'unavailable'
    if crosscheck is not None:
        if (not isinstance(crosscheck, Mapping) or crosscheck.get('manifest_sha256') != evaluation.get('manifest_sha256')
                or crosscheck.get('archive_sha256') != evaluation.get('archive_sha256')
                or crosscheck.get('frequency_absolute_tolerance') != frequency_tolerance
                or not isinstance(crosscheck.get('cases'), list)):
            raise ValueError('crosscheck identity or tolerance differs')
        cases = crosscheck['cases']
        indexed = {case['case_index']: case for case in cases}
        if len(indexed) != len(cases) or set(indexed) != set(expected_case_indices):
            raise ValueError('crosscheck case roster differs')
        checks = []
        for case in cases:
            offsets = case.get('offsets')
            if (not isinstance(offsets, list) or len(offsets) != len(expected_offsets)
                    or any(not isinstance(item, Mapping) for item in offsets)
                    or any(isinstance(item.get('offset'), bool) or not isinstance(item.get('offset'), (int, float))
                           or not math.isfinite(item['offset']) for item in offsets)
                    or {item.get('offset') for item in offsets} != set(expected_offsets)):
                raise ValueError('crosscheck offset roster differs')
            passed = []
            for offset in offsets:
                points, gates, solver = (offset.get(key) for key in ('k_points_match', 'gates_match', 'solver_identity_match'))
                diff = offset.get('max_frequency_difference')
                if any(not isinstance(flag, bool) for flag in (points, gates, solver, offset.get('passed'))):
                    raise ValueError('crosscheck flags must be boolean')
                if diff is not None and (isinstance(diff, bool) or not isinstance(diff, (int, float))
                                         or not math.isfinite(diff) or diff < 0):
                    raise ValueError('invalid crosscheck frequency difference')
                actual = points and gates and solver and diff is not None and diff <= frequency_tolerance
                if offset['passed'] is not actual:
                    raise ValueError('crosscheck offset decision differs')
                passed.append(actual)
            if case.get('passed') is not all(passed):
                raise ValueError('crosscheck case decision differs')
            checks.append(all(passed))
        if crosscheck.get('all_passed') is not all(checks):
            raise ValueError('crosscheck aggregate decision differs')
        physical = 'pass' if all(checks) else 'fail'
    overall = ('fail' if 'fail' in (stat, physical) else
               'unavailable' if 'unavailable' in (stat, physical) else 'pass')
    return {'status': overall, 'statistical_criterion': stat, 'cross_environment_criterion': physical,
            'scope': 'conditional_development_criteria_only', 'p2_gate': 'not_evaluated'}
