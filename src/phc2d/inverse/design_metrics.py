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


def _joint_margin(nominal: Mapping[str, Any], unseen: Mapping[str, Any], *,
                  target: float) -> tuple[float, float] | None:
    """Intersect verified evidence only when both identify the same base design."""
    if _verified_margin(nominal, target=target) is None or _verified_margin(unseen, target=target) is None:
        return None
    identities = (nominal.get('design_sha256'), unseen.get('design_sha256'))
    if any(value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value))
           for value in identities):
        raise ValueError('invalid base design identity')
    if any(value is None for value in identities):
        return None
    if identities[0] != identities[1]:
        raise ValueError('nominal and unseen evidence refer to different base designs')
    lower = max(float(nominal['lower_edge']), float(unseen['lower_edge']))
    upper = min(float(nominal['upper_edge']), float(unseen['upper_edge']))
    return 2.0 * min(target - lower, upper - target) / target, upper - lower


def summarize_design_outcomes(expected: Sequence[Mapping[str, Any]],
                              observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one method over an explicitly frozen target/seed roster.

    The caller must first validate the underlying solver artifacts, unseen
    perturbation identity, band pair and frequency units. Evidence is for the provided finite k/offset sets;
    this function never certifies continuous BZ or manufacturing yield.
    Joint nominal/unseen evidence requires matching ``design_sha256`` base
    identities; its common interval is the intersection of both intervals.
    When either identity is missing, joint success is unverified and counts as
    unsuccessful in all-roster rates; ``both_identity_unavailable`` records it.
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
              'unseen_success': 0, 'both_verified': 0, 'both_identity_unavailable': 0,
              'both_success': 0, 'predicted_positive': 0,
              'predictions_available': 0,
              'predicted_positive_verified': 0, 'predicted_positive_unverified': 0,
              'false_positive': 0}
    nominal_states = {key: 0 for key in _STATES}
    unseen_states = {key: 0 for key in _STATES}
    widths: list[float] = []
    joint_widths: list[float] = []
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
        joint_result = _joint_margin(nominal, unseen, target=frequency)
        counts['both_verified'] += int(joint_result is not None)
        counts['both_identity_unavailable'] += int(nominal_result is not None and unseen_result is not None
                                                   and joint_result is None)
        counts['nominal_verified'] += int(nominal_result is not None)
        counts['unseen_verified'] += int(unseen_result is not None)
        counts['nominal_success'] += int(nominal_success)
        counts['unseen_success'] += int(unseen_success)
        joint_success = joint_result is not None and joint_result[0] > 0
        counts['both_success'] += int(joint_success)
        if joint_success:
            joint_widths.append(joint_result[1])
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
            'mean_successful_nominal_common_width': sum(widths) / len(widths) if widths else None,
            'mean_successful_joint_common_width': sum(joint_widths) / len(joint_widths) if joint_widths else None}


def summarize_verified_budget_curve(
    attempts: Sequence[Mapping[str, Any]], *, target_frequency: float,
    cpu_milestones: Sequence[float], gpu_cap_seconds: float, target_margin: float = 0.0,
) -> dict[str, Any]:
    """Summarize one ordered trajectory at CPU budgets and a fixed GPU cap.

    Each attempt supplies cumulative settled CPU core-seconds and GPU-seconds,
    a unique attempt ID, a base design hash and physical evidence. Failed or
    unresolved attempts consume budget but cannot improve the incumbent.
    A target is reached only when a verified margin is strictly greater than
    ``target_margin`` and lies within the largest supplied CPU milestone and
    GPU cap. This operation does not authenticate artifacts or
    scheduler accounts: the caller must verify provenance, costs, completion
    order, and that all concurrent and excluded work is accounted for.
    """
    def number(value: Any, *, nonnegative: bool, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f'{name} must be finite')
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be finite') from exc
        if not math.isfinite(result) or (nonnegative and result < 0):
            raise ValueError(f'{name} must be finite and nonnegative' if nonnegative else f'{name} must be finite')
        return result

    target = number(target_frequency, nonnegative=True, name='target frequency')
    if target == 0:
        raise ValueError('target frequency must be positive')
    gpu_cap = number(gpu_cap_seconds, nonnegative=True, name='GPU cap')
    threshold = number(target_margin, nonnegative=False, name='target margin')
    if (not isinstance(cpu_milestones, Sequence) or isinstance(cpu_milestones, (str, bytes))
            or not cpu_milestones):
        raise ValueError('CPU milestones must be a nonempty sequence')
    milestones = [number(x, nonnegative=True, name='CPU milestone') for x in cpu_milestones]
    if any(a >= b for a, b in zip(milestones, milestones[1:])):
        raise ValueError('CPU milestones must be strictly increasing')
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise ValueError('attempts must be an ordered sequence')
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_cpu = previous_gpu = 0.0
    for item in attempts:
        if not isinstance(item, Mapping):
            raise ValueError('attempt must be an object')
        attempt_id, design = item.get('attempt_id'), item.get('design_sha256')
        if not isinstance(attempt_id, str) or not attempt_id.strip() or attempt_id in seen:
            raise ValueError('attempt IDs must be unique nonempty strings')
        if not isinstance(design, str) or not _SHA256.fullmatch(design):
            raise ValueError('attempt needs a base design identity')
        seen.add(attempt_id)
        cpu = number(item.get('cumulative_cpu_core_seconds'), nonnegative=True, name='cumulative CPU cost')
        gpu = number(item.get('cumulative_gpu_seconds'), nonnegative=True, name='cumulative GPU cost')
        if cpu < previous_cpu or gpu < previous_gpu:
            raise ValueError('cumulative resource costs cannot decrease')
        previous_cpu, previous_gpu = cpu, gpu
        physical = item.get('physical')
        if not isinstance(physical, Mapping):
            raise ValueError('attempt needs physical evidence status')
        if physical.get('status') == 'not_run':
            raise ValueError('an unrun candidate is not a completed attempt')
        margin = _verified_margin(physical, target=target)
        physical_design = physical.get('design_sha256')
        if physical_design is not None and physical_design != design:
            raise ValueError('physical evidence belongs to a different design')
        if margin is not None and physical_design != design:
            raise ValueError('verified evidence must identify the attempted design')
        verified.append({'attempt_id': attempt_id, 'design_sha256': design, 'cpu': cpu, 'gpu': gpu,
                         'artifact_sha256': physical.get('artifact_sha256') if margin is not None else None,
                         'margin': None if margin is None else margin[0]})

    points = []
    cursor, best, first_reached = 0, None, None
    for limit in milestones:
        while cursor < len(verified) and verified[cursor]['cpu'] <= limit and verified[cursor]['gpu'] <= gpu_cap:
            row = verified[cursor]
            if row['margin'] is not None and (best is None or row['margin'] > best['margin']):
                best = row
            if first_reached is None and row['margin'] is not None and row['margin'] > threshold:
                first_reached = {'attempt_id': row['attempt_id'], 'design_sha256': row['design_sha256'],
                                 'artifact_sha256': row['artifact_sha256'],
                                 'cpu_core_seconds': row['cpu'], 'gpu_seconds': row['gpu'], 'margin': row['margin']}
            cursor += 1
        points.append({'cpu_budget': limit, 'attempts_completed': cursor,
                       'best_verified_margin': None if best is None else best['margin'],
                       'best_design_sha256': None if best is None else best['design_sha256'],
                       'best_attempt_id': None if best is None else best['attempt_id'],
                       'best_artifact_sha256': None if best is None else best['artifact_sha256']})
    return {'scope': 'single_ordered_trajectory_caller_verifies_evidence_and_cumulative_costs',
            'target_frequency': target, 'gpu_cap_seconds': gpu_cap, 'target_margin_strictly_above': threshold,
            'points': points, 'first_reached': first_reached}


def compare_paired_budget_curves(
    expected: Sequence[Mapping[str, Any]], *,
    left: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    right: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    cpu_milestones: Sequence[float], gpu_cap_seconds: float, target_margin: float = 0.0,
    bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compare target attainment across paired trajectories at shared budgets.

    Each mapping contains at most one ordered trajectory per target/seed;
    missing trajectories and attempts without verified margins do not attain
    the target. Margin differences are conditional on both methods having
    verified margins and may be selection biased. Seeds are averaged within
    each target before pointwise target bootstrap. The caller must authenticate
    and account for complete work, artifacts, ordering and actual cumulative
    costs for both methods.
    """
    import numpy as np

    if (isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1):
        raise ValueError('bootstrap seed and sample count are invalid')
    summarize_design_outcomes(expected, [])  # validate roster and target frequencies
    roster = {_key(item): float(item['target_frequency']) for item in expected}
    seed_sets: dict[str, set[int]] = {}
    for target, seed in roster:
        seed_sets.setdefault(target, set()).add(seed)
    if len({frozenset(values) for values in seed_sets.values()}) != 1:
        raise ValueError('each target must have the same frozen seed roster')
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError('paired trajectory mappings are required')
    for source in (left, right):
        for identity in source:
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise ValueError('trajectory identity must be a target/seed pair')
            _key({'target_id': identity[0], 'seed': identity[1]})
    if set(left) - set(roster) or set(right) - set(roster):
        raise ValueError('trajectory identity is outside the frozen roster')
    curves: dict[str, dict[tuple[str, int], dict[str, Any]]] = {'left': {}, 'right': {}}
    for name, source in (('left', left), ('right', right)):
        for identity, frequency in roster.items():
            curves[name][identity] = summarize_verified_budget_curve(source.get(identity, ()),
                target_frequency=frequency, cpu_milestones=cpu_milestones,
                gpu_cap_seconds=gpu_cap_seconds, target_margin=target_margin)
    targets = sorted(seed_sets)
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(0, len(targets), size=(bootstrap_samples, len(targets))) if len(targets) >= 2 else None
    points = []
    for index, budget in enumerate(cpu_milestones):
        groups: dict[str, list[float]] = {target: [] for target in targets}
        margin_groups: dict[str, list[float]] = {}
        stats = {name: {'attained': 0, 'verified': 0, 'no_completed_attempt': 0,
                        'missing_trajectories': len(set(roster) - set(source))}
                 for name, source in (('left', left), ('right', right))}
        for identity in sorted(roster):
            outcomes, margins = {}, {}
            for name in ('left', 'right'):
                point = curves[name][identity]['points'][index]
                margin = point['best_verified_margin']
                attained = margin is not None and margin > float(target_margin)
                stats[name]['attained'] += int(attained)
                stats[name]['verified'] += int(margin is not None)
                stats[name]['no_completed_attempt'] += int(point['attempts_completed'] == 0)
                outcomes[name] = attained
                margins[name] = margin
            groups[identity[0]].append(float(outcomes['left']) - float(outcomes['right']))
            if margins['left'] is not None and margins['right'] is not None:
                margin_groups.setdefault(identity[0], []).append(margins['left'] - margins['right'])
        attempted = {name: len(roster) - stats[name]['no_completed_attempt'] for name in ('left', 'right')}
        for name in ('left', 'right'):
            stats[name]['attainment_rate_all'] = stats[name]['attained'] / len(roster) if attempted[name] else None
        values = np.asarray([np.mean(groups[target]) for target in targets], dtype=np.float64)
        difference = float(values.mean()) if all(attempted.values()) else None
        low = high = None
        if difference is not None and samples is not None:
            low, high = (float(x) for x in np.quantile(values[samples].mean(axis=1), [.025, .975]))
        margin_values = np.asarray([np.mean(margin_groups[target]) for target in sorted(margin_groups)], dtype=np.float64)
        margin_mean = float(margin_values.mean()) if len(margin_values) else None
        margin_low = margin_high = None
        if len(margin_values) >= 2:
            margin_samples = np.random.default_rng(bootstrap_seed).integers(
                0, len(margin_values), size=(bootstrap_samples, len(margin_values)))
            margin_low, margin_high = (float(x) for x in
                np.quantile(margin_values[margin_samples].mean(axis=1), [.025, .975]))
        points.append({'cpu_budget': float(budget), 'left': stats['left'], 'right': stats['right'],
                       'paired_attainment_difference': difference, 'bootstrap_95_low': low,
                       'bootstrap_95_high': high, 'verified_margin_difference_conditional': {
                           'target_count': len(margin_values),
                           'pair_count': sum(len(values) for values in margin_groups.values()),
                           'mean_difference': margin_mean, 'bootstrap_95_low': margin_low,
                           'bootstrap_95_high': margin_high,
                           'scope': 'both_verified_only_potential_selection_bias'}})
    return {'scope': 'paired_trajectory_descriptive_caller_verifies_evidence_and_cumulative_costs',
            'target_count': len(targets), 'target_seed_count': len(roster),
            'resampling_unit': 'target_id', 'bootstrap_seed': bootstrap_seed,
            'bootstrap_samples': bootstrap_samples, 'interval_type': 'pointwise_not_simultaneous',
            'gpu_cap_seconds': float(gpu_cap_seconds), 'target_margin_strictly_above': float(target_margin),
            'points': points}


def compare_paired_budget_curves_audited(
    expected: Sequence[Mapping[str, Any]], *,
    left: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    right: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    budget: Mapping[str, float], cpu_milestones: Sequence[float],
    left_ledgers: Mapping[tuple[str, int], Any], right_ledgers: Mapping[tuple[str, int], Any],
    left_accounting: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    right_accounting: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    left_setup_tasks: Mapping[tuple[str, int], Sequence[str]],
    right_setup_tasks: Mapping[tuple[str, int], Sequence[str]],
    target_margin: float = 0.0, bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Recompute paired budget curves from settled Slurm task allocations.

    Setup task IDs and each ordered attempt's ``task_ids`` must partition the
    trial ledger exactly. Each attempt also has ``attempt_id``, ``design_sha256``
    and ``physical``; cumulative costs are derived here, not supplied by the
    caller. This does not prove scheduler authenticity, completion ordering,
    absent concurrent work, requeue exclusion or excluded pretraining/WSL cost.
    """
    from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger, audit_slurm_ledger

    summarize_design_outcomes(expected, [])
    roster = {_key(item) for item in expected}
    if not isinstance(budget, Mapping):
        raise ValueError('shared resource budget is required')
    try:
        cap = ResourceAmount(budget['cpu_core_seconds'], budget['gpu_seconds'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('valid CPU and GPU budget caps are required') from exc
    empty_curve = summarize_verified_budget_curve((), target_frequency=1., cpu_milestones=cpu_milestones,
                                                   gpu_cap_seconds=cap.gpu_seconds, target_margin=target_margin)
    if empty_curve['points'][-1]['cpu_budget'] > cap.cpu_core_seconds:
        raise ValueError('CPU milestone exceeds shared budget')
    evidence_maps = (left_ledgers, right_ledgers, left_accounting, right_accounting,
                     left_setup_tasks, right_setup_tasks)
    if (not all(isinstance(value, Mapping) for value in evidence_maps)
            or any(set(value) != roster for value in evidence_maps)):
        raise ValueError('ledger, accounting and setup tasks must cover the full roster')
    for evidence in evidence_maps:
        for identity in evidence:
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise ValueError('evidence identity must be a target/seed pair')
            _key({'target_id': identity[0], 'seed': identity[1]})
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError('paired trajectory mappings are required')
    for source in (left, right):
        for identity in source:
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise ValueError('trajectory identity must be a target/seed pair')
            _key({'target_id': identity[0], 'seed': identity[1]})
        if set(source) - roster:
            raise ValueError('trajectory identity is outside the frozen roster')

    frequencies = {_key(item): float(item['target_frequency']) for item in expected}
    used_allocations: set[str] = set()
    converted: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {'left': {}, 'right': {}}
    resources: dict[str, Any] = {}
    for name, source, ledgers, accounting, setup in (
        ('left', left, left_ledgers, left_accounting, left_setup_tasks),
        ('right', right, right_ledgers, right_accounting, right_setup_tasks),
    ):
        trials = []
        for identity in sorted(roster):
            ledger = ledgers[identity]
            if not isinstance(ledger, ResourceLedger) or ledger.limits != cap:
                raise ValueError('trial ledger limits must match the shared budget')
            attempts = source.get(identity, ())
            if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
                raise ValueError('attempts must be an ordered sequence')
            audited = audit_slurm_ledger(ledger, accounting_by_task=accounting[identity],
                                         require_settled=bool(attempts))
            allocations = set(audited['allocation_ids'])
            if used_allocations.intersection(allocations):
                raise ValueError('scheduler allocation is shared or counted more than once')
            used_allocations.update(allocations)
            records = {record.task_id: record for record in ledger.records}
            consumed: set[str] = set()

            def use_tasks(task_ids: Sequence[str]) -> ResourceAmount:
                if (not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes))
                        or any(not isinstance(task, str) or not task.strip() for task in task_ids)
                        or len(set(task_ids)) != len(task_ids) or consumed.intersection(task_ids)
                        or set(task_ids) - set(records)):
                    raise ValueError('task IDs must be unique, disjoint and present in the trial ledger')
                consumed.update(task_ids)
                return ResourceAmount(sum(records[task].actual.cpu_core_seconds for task in task_ids),
                                      sum(records[task].actual.gpu_seconds for task in task_ids))

            total = use_tasks(setup[identity])
            trajectory = []
            for attempt in attempts:
                if not isinstance(attempt, Mapping) or set(attempt) != {'attempt_id', 'design_sha256', 'physical', 'task_ids'}:
                    raise ValueError('attempt must provide identity, physical evidence and task IDs')
                allocated = use_tasks(attempt['task_ids'])
                total = ResourceAmount(total.cpu_core_seconds + allocated.cpu_core_seconds,
                                       total.gpu_seconds + allocated.gpu_seconds)
                trajectory.append({'attempt_id': attempt['attempt_id'], 'design_sha256': attempt['design_sha256'],
                    'physical': attempt['physical'], 'cumulative_cpu_core_seconds': total.cpu_core_seconds,
                    'cumulative_gpu_seconds': total.gpu_seconds})
            if consumed != set(records):
                raise ValueError('all settled tasks must be assigned to setup or an attempt')
            if identity in source:
                converted[name][identity] = trajectory
            curve = summarize_verified_budget_curve(trajectory, target_frequency=frequencies[identity],
                cpu_milestones=cpu_milestones, gpu_cap_seconds=cap.gpu_seconds, target_margin=target_margin)
            trials.append({'target_id': identity[0], 'seed': identity[1], **audited,
                           'first_reached': curve['first_reached']})
        resources[name] = {'cpu_core_seconds': sum(x['cpu_core_seconds'] for x in trials),
                           'gpu_seconds': sum(x['gpu_seconds'] for x in trials), 'trials': trials}
    comparison = compare_paired_budget_curves(expected, left=converted['left'], right=converted['right'],
        cpu_milestones=cpu_milestones, gpu_cap_seconds=cap.gpu_seconds, target_margin=target_margin,
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    return {'scope': 'paired_budget_curves_with_replayed_slurm_ledgers', 'comparison': comparison,
            'resources': resources,
            'limitations': 'caller_verifies_authentic_complete_ordered_work_no_requeue_and_excluded_costs'}


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
        joint = _joint_margin(row['nominal'], row['unseen'], target=frequency)
        return joint is not None and joint[0] > 0, joint is not None

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
