"""Paired surrogate-versus-solver ranking for local inverse audit candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau


def paired_candidate_ranking(proxy: Sequence[Mapping[str, Any]],
                             solver: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare complete paired groups, selecting among non-anchor candidates.

    Random baseline is the exact average over the same eligible non-anchor
    candidates. Solver evidence may be coarse sampled; this operation does not
    infer numerical convergence or physical validation.
    """
    def key(row: Mapping[str, Any]) -> tuple[int, str, str]:
        return int(row['source_index']), str(row['label']), str(row['parameter_hash'])

    proxy_by_key = {key(row): row for row in proxy}
    solver_by_key = {key(row): row for row in solver}
    if len(proxy_by_key) != len(proxy) or len(solver_by_key) != len(solver):
        raise ValueError('duplicate candidate identity')
    if not proxy_by_key or proxy_by_key.keys() != solver_by_key.keys():
        raise ValueError('proxy and solver candidate identities differ')
    groups: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for identity in sorted(proxy_by_key):
        p, s = proxy_by_key[identity], solver_by_key[identity]
        if p.get('status') != 'surrogate_only' or s.get('status') != 'sampled_solver_only':
            raise ValueError('candidate has invalid proxy or solver status')
        if not np.isfinite([p['predicted_margin'], s['gap']['target_margin']]).all():
            raise ValueError('candidate scores must be finite')
        groups[identity[0]].append((p, s))
    result = []
    for source, entries in sorted(groups.items()):
        anchors = [item for item in entries if item[0]['label'] == 'anchor']
        candidates = [item for item in entries if item[0]['label'] != 'anchor']
        if len(anchors) != 1 or len(candidates) < 2:
            raise ValueError('each source needs one anchor and at least two candidates')
        # Stable ordering from identity sorting defines reproducible tie handling.
        selected = max(candidates, key=lambda item: float(item[0]['predicted_margin']))
        true = np.asarray([float(s['gap']['target_margin']) for _, s in candidates])
        predicted = np.asarray([float(p['predicted_margin']) for p, _ in candidates])
        tau = float(kendalltau(predicted, true).statistic)
        anchor = float(anchors[0][1]['gap']['target_margin'])
        selected_true = float(selected[1]['gap']['target_margin'])
        result.append({'source_index': source, 'candidate_count': len(candidates),
                       'selected_label': selected[0]['label'], 'selected_parameter_hash': selected[0]['parameter_hash'],
                       'anchor_solver_margin': anchor, 'selected_solver_margin': selected_true,
                       'selected_gain': selected_true - anchor,
                       'random_expected_gain': float(true.mean()) - anchor,
                       'kendall_tau': tau if np.isfinite(tau) else None,
                       'kendall_status': 'defined' if np.isfinite(tau) else 'undefined_constant_rank'})
    return result
