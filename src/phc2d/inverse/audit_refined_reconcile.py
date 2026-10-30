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
from phc2d.inverse.audit_ranking import paired_candidate_ranking


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
