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
        frequency = row.get('target_frequency')
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or frequency != roster[identity]:
            raise ValueError('observation target frequency differs from roster')
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
    checkpoint_by_seed: dict[int, str] = {}
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
            if identity[1] in checkpoint_by_seed and checkpoint_by_seed[identity[1]] != checkpoint:
                raise ValueError('checkpoint identity differs within a seed')
            checkpoint_by_seed[identity[1]] = checkpoint
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


def compare_paired_success(expected: Sequence[Mapping[str, Any]], *,
                           left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
                           left_budget: Mapping[str, float], right_budget: Mapping[str, float],
                           metric: str, bootstrap_seed: int, bootstrap_samples: int = 10000) -> dict[str, Any]:
    """Compare paired success under the same declared CPU/GPU resource caps.

    Bootstrap units are target IDs; seeds for one target are averaged first.
    Missing or failed attempts count as unsuccessful in the all-roster view.
    The optional verified-pair view is conditional and may be selection biased.
    The caller must independently verify actual resource use against those caps.
    This is descriptive unless the target sampling design supports inference.
    """
    import numpy as np

    if metric not in {'nominal', 'unseen', 'both'}:
        raise ValueError('metric must be nominal, unseen, or both')
    if (isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1):
        raise ValueError('bootstrap seed and sample count are invalid')

    def budget(value: Mapping[str, float]) -> tuple[float, float]:
        try:
            if any(isinstance(value[key], bool) for key in ('cpu_core_seconds', 'gpu_seconds')):
                raise ValueError('boolean budget')
            values = tuple(float(value[key]) for key in ('cpu_core_seconds', 'gpu_seconds'))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('CPU and GPU budget caps are required') from exc
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError('budget caps must be finite and nonnegative')
        return values

    if budget(left_budget) != budget(right_budget):
        raise ValueError('paired methods require identical resource caps')
    left_summary = summarize_design_outcomes(expected, left)
    right_summary = summarize_design_outcomes(expected, right)
    seed_sets: dict[str, set[int]] = {}
    for item in expected:
        target, seed = _key(item)
        seed_sets.setdefault(target, set()).add(seed)
    if len({frozenset(values) for values in seed_sets.values()}) != 1:
        raise ValueError('each target must have the same frozen seed roster')
    rate_key = {'nominal': 'nominal_success_rate_all', 'unseen': 'unseen_success_rate_all',
                'both': 'both_success_rate_all'}[metric]
    if left_summary[rate_key] is None or right_summary[rate_key] is None:
        raise ValueError('both methods must have attempted the requested physical evaluation')
    left_by_key = {_key(row): row for row in left}
    right_by_key = {_key(row): row for row in right}

    def outcome(row: Mapping[str, Any] | None, frequency: float) -> tuple[bool, bool]:
        if row is None:
            return False, False
        nominal = _verified_margin(row['nominal'], target=frequency)
        unseen = _verified_margin(row['unseen'], target=frequency)
        if metric == 'nominal':
            return nominal is not None and nominal[0] > 0, nominal is not None
        if metric == 'unseen':
            return unseen is not None and unseen[0] > 0, unseen is not None
        return (nominal is not None and nominal[0] > 0 and unseen is not None and unseen[0] > 0,
                nominal is not None and unseen is not None)

    grouped: dict[str, list[float]] = {}
    verified_grouped: dict[str, list[float]] = {}
    for item in expected:
        identity = _key(item)
        left_success, left_verified = outcome(left_by_key.get(identity), float(item['target_frequency']))
        right_success, right_verified = outcome(right_by_key.get(identity), float(item['target_frequency']))
        difference = float(left_success) - float(right_success)
        grouped.setdefault(identity[0], []).append(difference)
        if left_verified and right_verified:
            verified_grouped.setdefault(identity[0], []).append(difference)

    def describe(groups: Mapping[str, list[float]], *, seed: int) -> dict[str, Any]:
        values = np.asarray([np.mean(groups[key]) for key in sorted(groups)], dtype=np.float64)
        if not len(values):
            return {'target_count': 0, 'pair_count': 0, 'mean_difference': None,
                    'bootstrap_95_low': None, 'bootstrap_95_high': None}
        low = high = None
        if len(values) >= 2:
            rng = np.random.default_rng(seed)
            samples = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            low, high = (float(x) for x in np.quantile(values[samples].mean(axis=1), [.025, .975]))
        return {'target_count': len(values), 'pair_count': sum(len(v) for v in groups.values()),
                'mean_difference': float(values.mean()), 'bootstrap_95_low': low, 'bootstrap_95_high': high}

    return {'scope': 'paired_descriptive_finite_solver_evidence', 'metric': metric,
            'resampling_unit': 'target_id', 'bootstrap_seed': bootstrap_seed,
            'bootstrap_samples': bootstrap_samples, 'left': left_summary, 'right': right_summary,
            'all_roster': describe(grouped, seed=bootstrap_seed),
            'verified_pairs_conditional': describe(verified_grouped, seed=bootstrap_seed)}


def compare_paired_success_audited(
    expected: Sequence[Mapping[str, Any]], *, left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]], budget: Mapping[str, float],
    left_ledgers: Mapping[tuple[str, int], Any], right_ledgers: Mapping[tuple[str, int], Any],
    left_accounting: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    right_accounting: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    metric: str, bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compare paired trials only after replaying every Slurm budget ledger.

    Ledgers and raw accounting must cover the entire target/seed roster for
    each method, including empty ledgers for genuinely unrun trials. Duplicate
    allocations across trials or methods fail closed. Callers still verify
    authentic scheduler records, trial-work completeness, no requeue, physical
    artifacts and excluded costs such as pretraining and local WSL validation.
    """
    from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger, audit_slurm_ledger

    comparison = compare_paired_success(expected, left=left, right=right, left_budget=budget,
        right_budget=budget, metric=metric, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    roster = {_key(item) for item in expected}
    if (not all(isinstance(v, Mapping) for v in (left_ledgers, right_ledgers, left_accounting, right_accounting))
            or any(set(v) != roster for v in (left_ledgers, right_ledgers, left_accounting, right_accounting))):
        raise ValueError('budget evidence must cover the frozen roster for both methods')
    cap = ResourceAmount(float(budget['cpu_core_seconds']), float(budget['gpu_seconds']))
    used_allocations: set[str] = set()
    resource_summary: dict[str, Any] = {}
    for name, observations, ledgers, accounting in (
            ('left', left, left_ledgers, left_accounting), ('right', right, right_ledgers, right_accounting)):
        by_key = {_key(row): row for row in observations}
        trials = []
        for identity in sorted(roster):
            ledger = ledgers[identity]
            if not isinstance(ledger, ResourceLedger) or ledger.limits != cap:
                raise ValueError('trial ledger limits must match the shared budget')
            record = by_key.get(identity)
            attempted = record is not None and any(record[k]['status'] != 'not_run' for k in ('nominal', 'unseen'))
            audited = audit_slurm_ledger(ledger, accounting_by_task=accounting[identity],
                                         require_settled=attempted)
            identities = set(audited['allocation_ids'])
            if used_allocations.intersection(identities):
                raise ValueError('scheduler allocation is shared or counted more than once')
            used_allocations.update(identities)
            trials.append({'target_id': identity[0], 'seed': identity[1], **audited})
        resource_summary[name] = {'cpu_core_seconds': sum(row['cpu_core_seconds'] for row in trials),
                                  'gpu_seconds': sum(row['gpu_seconds'] for row in trials), 'trials': trials}
    return {'scope': 'paired_finite_solver_evidence_with_replayed_slurm_ledgers',
            'comparison': comparison, 'resources': resource_summary,
            'limitations': 'caller_verifies_complete_authentic_work_no_requeue_and_excluded_costs'}
