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
