"""Pure state transitions for solver-gated inverse design feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.inverse.refinement_feedback import compare_refined_candidates


@dataclass(frozen=True)
class PendingCandidate:
    parameter_hash: str
    parameters: tuple[float, ...]
    predicted_center: float
    predicted_candidate: float
    surrogate_contract_sha256: str


@dataclass(frozen=True)
class FeedbackState:
    """One candidate-evaluation budget; the initial incumbent costs are separate."""

    incumbent_hash: str
    incumbent_parameters: tuple[float, ...]
    incumbent_observation: Mapping[str, Any]
    target_frequency: float
    radius: float
    min_radius: float
    max_radius: float
    total_candidate_budget: int
    remaining_evaluations: int
    seen_hashes: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...] = ()
    pending: PendingCandidate | None = None


def _parameters(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError('parameters must be finite and nonempty')
    return result


def start_feedback(*, incumbent_hash: str, incumbent_parameters: Sequence[float],
                   incumbent_observation: Mapping[str, Any], target_frequency: float,
                   radius: float, min_radius: float, max_radius: float,
                   candidate_budget: int) -> FeedbackState:
    """Start from an independently solver-verified incumbent.

    The candidate budget excludes the initial incumbent solve. A campaign's
    resource ledger must account for that cost and each offset separately.
    """
    point = _parameters(incumbent_parameters)
    if not incumbent_hash or incumbent_observation.get('provenance', {}).get('parameter_hash') != incumbent_hash:
        raise ValueError('incumbent identity mismatch')
    if (not math.isfinite(target_frequency) or target_frequency <= 0
            or not all(math.isfinite(x) for x in (radius, min_radius, max_radius))
            or not 0 < min_radius <= radius <= max_radius):
        raise ValueError('invalid target or radius')
    if not isinstance(candidate_budget, int) or isinstance(candidate_budget, bool) or candidate_budget < 0:
        raise ValueError('candidate budget must be a nonnegative integer')
    row = incumbent_observation['row']
    if not all(row.get(k) is True for k in ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')):
        raise ValueError('incumbent numerical gate not passed')
    if not incumbent_observation['provenance'].get('comparison_contract_sha256'):
        raise ValueError('incumbent solver contract missing')
    if (not isinstance(row.get('lower_band_one_based'), int) or row['lower_band_one_based'] < 1
            or not all(math.isfinite(float(row[k])) for k in ('refined_lower_edge_frequency', 'refined_upper_edge_frequency'))):
        raise ValueError('incumbent band pair or edges invalid')
    return FeedbackState(incumbent_hash, point, copy.deepcopy(incumbent_observation), float(target_frequency),
                         float(radius), float(min_radius), float(max_radius), candidate_budget,
                         candidate_budget, (incumbent_hash,))


def restore_feedback(payload: Mapping[str, Any]) -> FeedbackState:
    """Restore a serialized state without rerunning a pending physical attempt."""
    total = payload['total_candidate_budget']
    remaining = payload['remaining_evaluations']
    if (not isinstance(total, int) or isinstance(total, bool) or not isinstance(remaining, int)
            or isinstance(remaining, bool) or not 0 <= remaining <= total):
        raise ValueError('invalid saved budget')
    history = tuple(copy.deepcopy(payload['history']))
    seen = tuple(payload['seen_hashes'])
    if (len(history) != total - remaining or len(seen) != len(history) + 1
            or len(set(seen)) != len(seen) or not all(isinstance(x, str) and x for x in seen)
            or tuple(h['parameter_hash'] for h in history) != seen[1:]
            or payload['incumbent_hash'] not in seen):
        raise ValueError('saved history or identities are inconsistent')
    state = start_feedback(incumbent_hash=payload['incumbent_hash'],
                           incumbent_parameters=payload['incumbent_parameters'],
                           incumbent_observation=payload['incumbent_observation'],
                           target_frequency=payload['target_frequency'], radius=payload['radius'],
                           min_radius=payload['min_radius'], max_radius=payload['max_radius'],
                           candidate_budget=total)
    state = replace(state, remaining_evaluations=remaining, seen_hashes=seen, history=history)
    pending = payload.get('pending')
    if pending is not None:
        state = propose_feedback(state, parameter_hash=pending['parameter_hash'], parameters=pending['parameters'],
                                 predicted_center=pending['predicted_center'],
                                 predicted_candidate=pending['predicted_candidate'],
                                 surrogate_contract_sha256=pending['surrogate_contract_sha256'])
    return state


def propose_feedback(state: FeedbackState, *, parameter_hash: str, parameters: Sequence[float],
                     predicted_center: float, predicted_candidate: float,
                     surrogate_contract_sha256: str) -> FeedbackState:
    """Reserve one new local candidate before a physical evaluation."""
    point = _parameters(parameters)
    if state.pending is not None or state.remaining_evaluations <= 0:
        raise ValueError('pending candidate or evaluation budget exhausted')
    if not parameter_hash or parameter_hash in state.seen_hashes:
        raise ValueError('candidate identity is missing or already evaluated')
    if len(point) != len(state.incumbent_parameters) or np.linalg.norm(np.asarray(point) - state.incumbent_parameters) > state.radius + 1e-12:
        raise ValueError('candidate outside local trust region')
    if not surrogate_contract_sha256 or not all(math.isfinite(x) for x in (predicted_center, predicted_candidate)):
        raise ValueError('candidate prediction or surrogate identity invalid')
    pending = PendingCandidate(parameter_hash, point, float(predicted_center), float(predicted_candidate), surrogate_contract_sha256)
    return replace(state, pending=pending)


def finish_feedback(state: FeedbackState, *, observation: Mapping[str, Any] | None = None,
                    failure_reason: str | None = None) -> FeedbackState:
    """Consume one physical attempt and retain the incumbent unless improvement is verified."""
    pending = state.pending
    if pending is None or (observation is None) == (failure_reason is None):
        raise ValueError('provide exactly one observation or failure for the pending candidate')
    if failure_reason is not None and not failure_reason.strip():
        raise ValueError('failure reason must be nonempty')
    base = {'parameter_hash': pending.parameter_hash, 'parameters': list(pending.parameters),
            'surrogate_contract_sha256': pending.surrogate_contract_sha256,
            'predicted_center': pending.predicted_center, 'predicted_candidate': pending.predicted_candidate}
    incumbent_hash, incumbent_point = state.incumbent_hash, state.incumbent_parameters
    incumbent_observation = state.incumbent_observation
    radius = state.radius
    if failure_reason is not None:
        entry = {**base, 'status': 'solver_failure', 'reason': failure_reason, 'accepted': False}
    else:
        if observation.get('provenance', {}).get('parameter_hash') != pending.parameter_hash:
            raise ValueError('candidate observation identity mismatch')
        comparison = compare_refined_candidates(
            state.incumbent_observation, observation, target_frequency=state.target_frequency,
            predicted_center=pending.predicted_center, predicted_candidate=pending.predicted_candidate,
            radius=state.radius, min_radius=state.min_radius, max_radius=state.max_radius)
        decision = comparison.get('decision')
        accepted = bool(decision and decision['accepted'])
        entry = {**base, 'status': comparison['status'], 'accepted': accepted, 'comparison': comparison}
        radius = float(comparison['radius'])
        if accepted:
            incumbent_hash, incumbent_point = pending.parameter_hash, pending.parameters
            incumbent_observation = copy.deepcopy(observation)
    return replace(state, incumbent_hash=incumbent_hash, incumbent_parameters=incumbent_point,
                   incumbent_observation=incumbent_observation, radius=radius,
                   remaining_evaluations=state.remaining_evaluations - 1,
                   seen_hashes=state.seen_hashes + (pending.parameter_hash,),
                   history=state.history + (entry,), pending=None)
