"""Decide a bounded paired-search development criterion from audited inputs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def adjudicate_paired_development(
    evaluation: Mapping[str, Any], crosscheck: Mapping[str, Any], *,
    expected_case_indices: Sequence[int], expected_offsets: Sequence[float],
    expected_source_count: int, expected_seeds: Sequence[int], expected_tolerance: float,
) -> dict[str, Any]:
    """Apply the frozen finite-grid development rule, leaving P2 unassessed.

    The caller independently recomputes both supplied reports and verifies
    provenance, lock timing, raw artifact hashes and scheduler accounting.
    """
    if (not expected_case_indices or len(set(expected_case_indices)) != len(expected_case_indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in expected_case_indices)
            or not expected_offsets or len(set(expected_offsets)) != len(expected_offsets)
            or not np.isfinite(expected_offsets).all() or isinstance(expected_source_count, bool)
            or not isinstance(expected_source_count, int) or expected_source_count < 2
            or not expected_seeds or len(set(expected_seeds)) != len(expected_seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in expected_seeds)
            or isinstance(expected_tolerance, bool) or not np.isfinite(expected_tolerance) or expected_tolerance < 0):
        raise ValueError('invalid frozen development contract')
    if (evaluation.get('p2_gate') != 'not_evaluated' or crosscheck.get('p2_gate') != 'not_evaluated'
            or any(evaluation.get(k) != crosscheck.get(k) for k in ('batch_sha256','manifest_sha256','protocol_sha256'))
            or crosscheck.get('frequency_absolute_tolerance') != expected_tolerance):
        raise ValueError('evaluation and crosscheck provenance mismatch')
    cases = crosscheck.get('cases')
    if not isinstance(cases, list) or [row['case_index'] for row in cases] != list(expected_case_indices):
        raise ValueError('crosscheck roster mismatch')
    hashes = evaluation.get('case_result_sha256')
    if not isinstance(hashes, Mapping):
        raise ValueError('missing physical result identities')
    checks = []
    for case in cases:
        index = case['case_index']
        if hashes.get(index, hashes.get(str(index))) != case['reference_sha256']:
            raise ValueError('crosscheck reference identity mismatch')
        if case['replay_exit_code'] != 0:
            passed = False
            if case.get('reason') != 'replay_process_failed' or case.get('offsets') != []:
                raise ValueError('failed replay record mismatch')
        else:
            offsets = case['offsets']
            if [item['offset'] for item in offsets] != list(expected_offsets):
                raise ValueError('crosscheck offsets mismatch')
            valid = []
            for item in offsets:
                flags = [item[k] for k in ('grid_match','k_points_match','gates_match')]
                if any(not isinstance(flag, bool) for flag in flags):
                    raise ValueError('crosscheck flags invalid')
                difference = item['max_frequency_difference']
                if difference is not None and (not np.isfinite(difference) or difference < 0):
                    raise ValueError('crosscheck frequency difference invalid')
                decision = all(flags) and difference is not None and difference <= expected_tolerance
                if item['passed'] is not decision:
                    raise ValueError('crosscheck offset decision mismatch')
                valid.append(decision)
            passed = all(valid)
        if case['passed'] is not passed:
            raise ValueError('crosscheck case decision mismatch')
        checks.append(passed)
    if crosscheck.get('all_passed') is not all(checks):
        raise ValueError('crosscheck aggregate mismatch')
    if evaluation.get('status') == 'unscoreable_numerical_gate_failures':
        if not evaluation.get('failed_case_indices') or evaluation.get('statistics') is not None:
            raise ValueError('invalid numerical-failure record')
        stat = 'unscoreable'
    elif evaluation.get('status') == 'paired_statistics_available':
        summary = evaluation['statistics']
        if (evaluation.get('failed_case_indices') != [] or summary.get('source_count') != expected_source_count
                or summary.get('model_seeds') != list(expected_seeds)
                or summary.get('source_seed_count') != expected_source_count*len(expected_seeds)
                or summary.get('seed_aggregation') != 'arithmetic_mean_within_source_before_resampling'
                or summary.get('status') != 'descriptive_only'
                or not np.isfinite([summary['selected_minus_random_mean'],summary['bootstrap_95_low'],
                                    summary['bootstrap_95_high']]).all()
                or summary['bootstrap_95_low'] > summary['bootstrap_95_high']):
            raise ValueError('paired statistical summary mismatch')
        stat = 'pass' if summary['bootstrap_95_low'] > 0 else 'fail'
    else:
        raise ValueError('unexpected paired evaluation status')
    cross = 'pass' if all(checks) else 'fail'
    decision = 'development_criterion_met' if stat == 'pass' and cross == 'pass' else 'development_criterion_not_met'
    return {'status':decision, 'statistical_criterion':stat, 'cross_environment_criterion':cross,
            'batch_sha256':evaluation['batch_sha256'], 'p2_gate':'not_evaluated',
            'scope':'conditional_finite_grid_development_not_p5'}
