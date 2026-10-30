"""Summarize inverse-design outcomes without hiding missing physical evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_STATES = frozenset({'verified', 'unresolved', 'failed', 'not_run'})


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    target, seed = row.get('target_id'), row.get('seed')
    if not isinstance(target, str) or not target.strip() or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('target and nonnegative integer seed are required')
    return target, seed


def _verified_margin(evidence: Mapping[str, Any], *, target: float) -> tuple[float, float] | None:
    status = evidence.get('status')
    if status not in _STATES:
        raise ValueError('invalid physical evidence status')
    if status != 'verified':
        if any(evidence.get(key) is not None for key in ('lower_edge', 'upper_edge')):
            raise ValueError('unverified evidence cannot supply decision edges')
        if evidence.get('numerical_gate_passed') is True:
            raise ValueError('unverified evidence cannot pass the numerical gate')
        if status in {'unresolved', 'failed'} and not str(evidence.get('reason') or '').strip():
            raise ValueError('failed or unresolved evidence needs a reason')
        return None
    identity = evidence.get('artifact_sha256')
    if evidence.get('numerical_gate_passed') is not True or not isinstance(identity, str) or not _SHA256.fullmatch(identity):
        raise ValueError('verified evidence needs a numerical gate and artifact identity')
    try:
        if isinstance(evidence['lower_edge'], bool) or isinstance(evidence['upper_edge'], bool):
            raise ValueError('boolean edge')
        lower, upper = float(evidence['lower_edge']), float(evidence['upper_edge'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('verified evidence requires finite physical edges') from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or min(lower, upper) < 0:
        raise ValueError('verified evidence requires finite nonnegative edges')
    return 2.0 * min(target - lower, upper - target) / target, upper - lower


def summarize_design_outcomes(expected: Sequence[Mapping[str, Any]],
                              observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one method over an explicitly frozen target/seed roster.

    The caller must first validate the underlying solver artifacts and unseen
    perturbation identity. Evidence is for the provided finite k/offset sets;
    this function never certifies continuous BZ or manufacturing yield.
    Missing records count as unsuccessful in all-roster rates while remaining
    distinct in coverage; if an evaluation type was never attempted, its
    all-roster rate is undefined. A false-positive denominator includes only predicted
    positive cases with numerically verified nominal physical evidence.
    """
    roster: dict[tuple[str, int], float] = {}
    target_frequencies: dict[str, float] = {}
    for item in expected:
        identity = _key(item)
        try:
            if isinstance(item['target_frequency'], bool):
                raise ValueError('boolean frequency')
            frequency = float(item['target_frequency'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('target frequency must be finite and positive') from exc
        if not math.isfinite(frequency) or frequency <= 0 or identity in roster:
            raise ValueError('target frequency or roster identity is invalid')
        if identity[0] in target_frequencies and target_frequencies[identity[0]] != frequency:
            raise ValueError('one target identity must have one frequency')
        target_frequencies[identity[0]] = frequency
        roster[identity] = frequency
    if not roster:
        raise ValueError('expected target/seed roster must not be empty')
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in observations:
        identity = _key(row)
        if identity not in roster or identity in indexed:
            raise ValueError('unexpected or duplicate target/seed observation')
        indexed[identity] = row

    counts = {'expected': len(roster), 'observed': len(indexed), 'missing': len(roster) - len(indexed),
              'nominal_verified': 0, 'nominal_success': 0, 'unseen_verified': 0,
              'unseen_success': 0, 'both_success': 0, 'predicted_positive': 0,
              'predictions_available': 0,
              'predicted_positive_verified': 0, 'predicted_positive_unverified': 0,
              'false_positive': 0}
    nominal_states = {key: 0 for key in _STATES}
    unseen_states = {key: 0 for key in _STATES}
    widths: list[float] = []
    for identity, row in indexed.items():
        frequency = roster[identity]
        nominal, unseen = row.get('nominal'), row.get('unseen')
        if not isinstance(nominal, Mapping) or not isinstance(unseen, Mapping):
            raise ValueError('nominal and unseen evidence are required')
        nominal_result = _verified_margin(nominal, target=frequency)
        unseen_result = _verified_margin(unseen, target=frequency)
        nominal_states[nominal['status']] += 1
        unseen_states[unseen['status']] += 1
        nominal_success = nominal_result is not None and nominal_result[0] > 0
        unseen_success = unseen_result is not None and unseen_result[0] > 0
        counts['nominal_verified'] += int(nominal_result is not None)
        counts['unseen_verified'] += int(unseen_result is not None)
        counts['nominal_success'] += int(nominal_success)
        counts['unseen_success'] += int(unseen_success)
        counts['both_success'] += int(nominal_success and unseen_success)
        if nominal_success:
            widths.append(nominal_result[1])
        prediction = row.get('prediction')
        if prediction is not None:
            if not isinstance(prediction, Mapping):
                raise ValueError('prediction must include checkpoint identity and margin')
            checkpoint = prediction.get('checkpoint_sha256')
            if not isinstance(checkpoint, str) or not _SHA256.fullmatch(checkpoint):
                raise ValueError('prediction checkpoint identity is required')
            try:
                if isinstance(prediction['target_margin'], bool):
                    raise ValueError('boolean prediction')
                margin = float(prediction['target_margin'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('prediction margin must be finite') from exc
            if not math.isfinite(margin):
                raise ValueError('prediction margin must be finite')
            counts['predictions_available'] += 1
            if margin > 0:
                counts['predicted_positive'] += 1
                if nominal_result is None:
                    counts['predicted_positive_unverified'] += 1
                else:
                    counts['predicted_positive_verified'] += 1
                    counts['false_positive'] += int(not nominal_success)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    nominal_attempted = sum(nominal_states[key] for key in ('verified', 'unresolved', 'failed'))
    unseen_attempted = sum(unseen_states[key] for key in ('verified', 'unresolved', 'failed'))
    return {'scope': 'provided_finite_solver_evidence_only', 'counts': counts,
            'nominal_status_counts': nominal_states, 'unseen_status_counts': unseen_states,
            'nominal_success_rate_all': ratio(counts['nominal_success'], counts['expected']) if nominal_attempted else None,
            'nominal_success_rate_verified': ratio(counts['nominal_success'], counts['nominal_verified']),
            'unseen_success_rate_all': ratio(counts['unseen_success'], counts['expected']) if unseen_attempted else None,
            'unseen_success_rate_verified': ratio(counts['unseen_success'], counts['unseen_verified']),
            'both_success_rate_all': ratio(counts['both_success'], counts['expected']) if nominal_attempted and unseen_attempted else None,
            'false_positive_rate_verified_predictions': ratio(counts['false_positive'], counts['predicted_positive_verified']),
            'mean_successful_nominal_common_width': sum(widths) / len(widths) if widths else None}
