"""Evaluate separately locked inverse-design variants on shared physical groups.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from phc2d.inverse.audit_refined_reconcile import summarize_multiseed_eligible_stratified


def evaluate_locked_variants(
    choices: Mapping[str, Any], manifest: Mapping[str, Any], physical: Mapping[str, Any], *,
    source_strata: Mapping[int, str], seeds: Sequence[int], physical_group_sha256: Mapping[int, str],
    expected_counts: Mapping[str, int], stratum_weights: Mapping[str, float], bootstrap_seed: int,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compare three frozen method selections with matched uniform baselines.

    The caller must authenticate the lock, its chronology and scores, the
    physical observations and group digests. Missing or failed physical cases
    make a source unavailable; transformed predictions remain surrogate-only.
    """
    entries = manifest['rows']
    observed = physical['rows']
    by_index = {row['case_index']: row for row in observed}
    if (len(by_index) != len(observed) or any(isinstance(i, bool) or not isinstance(i, int)
                                             or i not in range(len(entries)) for i in by_index)):
        raise ValueError('duplicate or unexpected physical case')
    for i, row in by_index.items():
        if any(row.get(k) != entries[i].get(k) for k in ('source_index', 'label', 'parameter_hash')):
            raise ValueError('physical case identity mismatch')
        margin = row.get('target_margin')
        if (isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin)
                or not isinstance(row.get('numerical_gate_passed'), bool)):
            raise ValueError('invalid physical case outcome')
    sources = choices['sources']
    groups = {name: {seed: [] for seed in seeds} for name in ('raw', 'projected', 'projected_anchor')}
    source_ids = [row['source_index'] for row in sources]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(source_strata):
        raise ValueError('locked source roster mismatch')
    if {row['source_index'] for row in entries} != set(source_strata):
        raise ValueError('manifest source roster mismatch')
    for source in sources:
        sid = source['source_index']
        items = [(i, row) for i, row in enumerate(entries) if row['source_index'] == sid]
        anchors = [i for i, row in items if row['label'] == 'anchor']
        if len(anchors) != 1 or source['anchor_case_index'] != anchors[0]:
            raise ValueError('locked anchor identity mismatch')
        if (not isinstance(source.get('anchor_numerical_gate_passed'), bool)
                or (anchors[0] in by_index and source['anchor_numerical_gate_passed']
                    != by_index[anchors[0]]['numerical_gate_passed'])):
            raise ValueError('locked anchor numerical gate mismatch')
        candidates = [(i, row) for i, row in items if row['label'] != 'anchor']
        lookup = {row['parameter_hash']: (i, row) for i, row in candidates}
        if len(lookup) != len(candidates):
            raise ValueError('duplicate candidate identity')
        models = source['models']
        if len(models) != len(seeds) or {row['seed'] for row in models} != set(seeds):
            raise ValueError('locked model roster mismatch')
        complete = all(i in by_index for i, _ in items)
        gates = complete and all(by_index[i]['numerical_gate_passed'] for i, _ in items)
        for name, by_seed in groups.items():
            field = 'raw_shared_candidate_hashes' if name == 'raw' else 'projected_shared_candidate_hashes'
            hashes = source[field]
            count_field = 'raw_shared_count' if name == 'raw' else 'projected_shared_count'
            if (not isinstance(hashes, list) or len(set(hashes)) != len(hashes)
                    or source.get(count_field) != len(hashes)
                    or any(h not in lookup for h in hashes)
                    or hashes != [r['parameter_hash'] for _, r in candidates if r['parameter_hash'] in hashes]):
                raise ValueError('shared eligible candidate roster mismatch')
            identities = [(sid, lookup[h][1]['label'], h) for h in hashes]
            excluded = [r['label'] for _, r in candidates if r['parameter_hash'] not in hashes]
            for model in models:
                seed, method = model['seed'], model['methods'][name]
                status, ranking = method['status'], None
                if status == 'selected':
                    scores = method.get('scores')
                    if not isinstance(scores, list) or len(scores) != len(hashes) or len(hashes) < 2:
                        raise ValueError('locked scores do not cover eligible candidates')
                    for score, identity in zip(scores, identities, strict=True):
                        value = score.get('margin')
                        if ((score.get('label'), score.get('parameter_hash')) != identity[1:]
                                or isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value)):
                            raise ValueError('locked candidate score mismatch')
                    winner = max(scores, key=lambda x: x['margin'])
                    if (method.get('label'), method.get('parameter_hash')) != (winner['label'], winner['parameter_hash']):
                        raise ValueError('locked winner differs from scores')
                    if complete:
                        anchor_margin = by_index[anchors[0]]['target_margin']
                        chosen = by_index[lookup[winner['parameter_hash']][0]]
                        average = sum(by_index[lookup[h][0]]['target_margin'] for h in hashes) / len(hashes)
                        ranking = {'source_index': sid, 'selected_label': winner['label'],
                                   'selected_parameter_hash': winner['parameter_hash'],
                                   'selected_gain': chosen['target_margin'] - anchor_margin,
                                   'random_expected_gain': average - anchor_margin}
                        status = 'ranked_descriptive'
                    else:
                        status = 'missing_physical_candidate'
                elif status not in ('unscoreable_shared_eligibility', 'anchor_numerical_gate_failed'):
                    raise ValueError('unknown locked method status')
                by_seed[seed].append({'source_index': sid, 'physical_group_sha256': physical_group_sha256[sid],
                    'status': status, 'ranking': ranking, 'all_numerical_gates_passed': gates,
                    'eligible_candidate_identities': identities, 'eligible_candidate_count': len(identities),
                    'unscoreable': excluded})
    result = {name: summarize_multiseed_eligible_stratified(by_seed, source_strata=source_strata,
        seeds=seeds, physical_group_sha256=physical_group_sha256, expected_counts=expected_counts,
        stratum_weights=stratum_weights, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
        require_shared_eligible=True) for name, by_seed in groups.items()}
    return {'status': 'descriptive_variant_evaluation', 'physical_acceptance': 'not_evaluated', 'methods': result}
