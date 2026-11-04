"""Bind a frozen feedback proposal to its later physical observation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.inverse.feedback_state import FeedbackState, finish_feedback, propose_feedback, restore_feedback


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def freeze_round(state: FeedbackState, proposal: Mapping[str, Any],
                 observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze prediction and pending state before the new MPB result is available."""
    if state.acceptance_policy != 'strict_physical_improvement' or state.pending is not None:
        raise ValueError('strict feedback state without pending work required')
    if proposal.get('status') != 'proposal_ready' or proposal.get('observation_sha256') != _digest(list(observations)):
        raise ValueError('proposal or prefit observation identity mismatch')
    index = proposal['selected_index']
    history = proposal['history']
    candidates = proposal['hard_screening']['candidates']
    if not isinstance(index, int) or not 0 <= index < len(history) or len(history) != 20 or len(candidates) != len(history):
        raise ValueError('invalid selected proposal')
    candidate, prediction = candidates[index], proposal['prefit_prediction']
    if (candidate['status'] != 'surrogate_only' or candidate['index'] != index
            or prediction['point'] != history[index]['parameters']
            or history[0]['parameters'] != list(state.incumbent_parameters)
            or proposal['correction']['center'] != list(state.incumbent_parameters)
            or float(proposal['correction']['radius']) != state.radius
            or proposal['correction']['comparison_contract_sha256'] != state.incumbent_observation['provenance']['comparison_contract_sha256']):
        raise ValueError('proposal candidate, center or radius mismatch')
    pending = propose_feedback(state, parameter_hash=candidate['parameter_hash'],
        parameters=prediction['point'], predicted_center=float(history[0]['objective']),
        predicted_candidate=float(history[index]['objective']),
        surrogate_contract_sha256=proposal['correction']['surrogate_contract_sha256'])
    result = {'schema': 'differentiable_feedback_round_v1', 'pending_state': asdict(pending),
            'proposal_sha256': _digest(proposal), 'prior_observations_sha256': _digest(list(observations)),
            'candidate': copy.deepcopy(candidate), 'prefit_prediction': copy.deepcopy(prediction),
            'correction': copy.deepcopy(proposal['correction'])}
    result['frozen_sha256'] = _digest(result)
    return result


def complete_round(frozen: Mapping[str, Any], *, physical: Mapping[str, Any],
                   composite: Mapping[str, Any]) -> dict[str, Any]:
    """Compare against frozen prediction before returning the new fit observation."""
    if frozen.get('schema') != 'differentiable_feedback_round_v1':
        raise ValueError('invalid frozen feedback round')
    content = {key: value for key, value in frozen.items() if key != 'frozen_sha256'}
    if frozen.get('frozen_sha256') != _digest(content):
        raise ValueError('frozen feedback round integrity mismatch')
    state = restore_feedback(frozen['pending_state'])
    if state.pending is None:
        raise ValueError('round has no pending candidate')
    pred = frozen['prefit_prediction']
    if (composite.get('parameter_hash') != state.pending.parameter_hash
            or composite.get('numerical_gate_passed') is not True
            or composite.get('surrogate_contract_sha256') != state.pending.surrogate_contract_sha256
            or composite.get('comparison_contract_sha256') != state.incumbent_observation['provenance']['comparison_contract_sha256']
            or composite.get('parameters') != list(state.pending.parameters)):
        raise ValueError('feedback observation identity or contract mismatch')
    pairs = (('hard_edges', 'hard_edges'), ('soft_exact_edges', 'soft_exact_edges'),
             ('predicted_edges', 'soft_smooth_edges'))
    for a, b in pairs:
        left, right = np.asarray(composite[a], dtype=float), np.asarray(pred[b], dtype=float)
        if (left.shape != (2,) or right.shape != (2,) or not np.isfinite(left).all()
                or not np.isfinite(right).all() or not np.allclose(left, right, rtol=0, atol=1e-8)):
            raise ValueError('observation differs from frozen prefit prediction')
    row = physical['row']
    edges = np.asarray([row['refined_lower_edge_frequency'], row['refined_upper_edge_frequency']], dtype=float)
    if (edges.shape != (2,) or not np.isfinite(edges).all()
            or not np.allclose(edges, composite['physical_edges'], rtol=0, atol=1e-8)):
        raise ValueError('physical and composite edge observations differ')
    result = finish_feedback(state, observation=physical)
    if result.history[-1]['status'] != 'compared':
        raise ValueError('physical result is not numerically admissible')
    corrected = np.asarray(pred['corrected_smooth_edges'], dtype=float)
    if corrected.shape != (2,) or not np.isfinite(corrected).all():
        raise ValueError('invalid frozen corrected prediction')
    return {'schema': 'differentiable_feedback_round_completed_v1', 'state': asdict(result),
            'prefit_edge_error': (corrected - edges).tolist(),
            'accepted': result.history[-1]['accepted'], 'observation_for_next_round': dict(composite),
            'prior_observations_sha256': frozen['prior_observations_sha256'],
            'proposal_sha256': frozen['proposal_sha256']}


def save_round_request(directory: Path, *, frozen: Mapping[str, Any],
                       observations: Sequence[Mapping[str, Any]]) -> None:
    """Persist the frozen prediction and existing observation bank before dispatch."""
    if (frozen.get('schema') != 'differentiable_feedback_round_v1'
            or frozen.get('prior_observations_sha256') != _digest(list(observations))
            or frozen.get('frozen_sha256') != _digest({k: v for k, v in frozen.items() if k != 'frozen_sha256'})):
        raise ValueError('frozen request or prior observation bank mismatch')
    directory.mkdir(parents=False, exist_ok=False)
    request = {'frozen': dict(frozen), 'observations': list(observations)}
    with (directory/'request.json').open('x', encoding='utf8') as stream:
        json.dump(request, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def save_round_completion(directory: Path, *, physical: Mapping[str, Any],
                          composite: Mapping[str, Any]) -> dict[str, Any]:
    """Resume a pending round without changing its request or repeating a result."""
    request = json.loads((directory/'request.json').read_bytes())
    frozen, prior = request['frozen'], request['observations']
    if frozen['prior_observations_sha256'] != _digest(prior):
        raise ValueError('saved observation bank identity mismatch')
    completed = complete_round(frozen, physical=physical, composite=composite)
    record = {'completion': completed, 'physical': dict(physical), 'composite': dict(composite),
              'next_observations': prior + [dict(composite)]}
    record = json.loads(json.dumps(record, allow_nan=False))
    destination = directory/'result.json'
    if destination.exists():
        previous = json.loads(destination.read_bytes())
        if previous != record:
            raise ValueError('existing round result differs; reconcile instead of replacing')
        return previous
    with destination.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return record
