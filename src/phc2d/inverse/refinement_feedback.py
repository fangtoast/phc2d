"""Gate trust-region feedback using paired refined solver observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from phc2d.inverse.trust_region import update_trust_region


def assess_refinement_group(
    observations: Mapping[str, Mapping[str, Any] | None], *,
    expected_members: Mapping[str, str], comparison_contract_sha256: str,
    stage_index: int, stage_count: int,
) -> dict[str, Any]:
    """Classify a paired group without mixing numerical refinement contracts.

    ``expected_members`` maps stable member IDs to their parameter hashes;
    repeated hashes are permitted for identical structures in separate arms.
    An absent member is pending, while ``None`` records a confirmed solver
    failure. If any numeric gate fails and another stage exists, the caller
    must evaluate *every* member at that stage before comparing the group.

    This pure policy does not dispatch work, verify raw solver evidence,
    establish roster fairness, or account for compute. The caller must freeze
    the roster and stages, independently validate observations and solver
    failures, and enforce budget and no-automatic-retry rules.
    """
    if (not isinstance(expected_members, Mapping) or not expected_members
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in expected_members.items())
            or not isinstance(observations, Mapping)
            or not isinstance(comparison_contract_sha256, str) or not comparison_contract_sha256
            or isinstance(stage_index, bool) or not isinstance(stage_index, int)
            or isinstance(stage_count, bool) or not isinstance(stage_count, int)
            or stage_count < 1 or not 0 <= stage_index < stage_count):
        raise ValueError('valid group roster, contract and stage are required')
    if not set(observations).issubset(expected_members):
        raise ValueError('observation is not in the frozen group roster')
    pending = [member for member in expected_members if member not in observations]
    failed, unstable, bands = [], [], set()
    gates = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
    for member, obs in observations.items():
        if obs is None:
            failed.append(member)
            continue
        if not isinstance(obs, Mapping) or not isinstance(obs.get('row'), Mapping) or not isinstance(obs.get('provenance'), Mapping):
            raise ValueError('group observation must include row and provenance')
        row, provenance = obs['row'], obs['provenance']
        if (provenance.get('parameter_hash') != expected_members[member]
                or provenance.get('comparison_contract_sha256') != comparison_contract_sha256):
            raise ValueError('group candidate identity or numerical contract differs')
        band = row.get('lower_band_one_based')
        if isinstance(band, bool) or not isinstance(band, int) or band < 1:
            raise ValueError('invalid group band pair')
        bands.add(band)
        if not all(isinstance(row.get(k), bool) for k in gates):
            raise ValueError('group numerical gates must be explicit booleans')
        try:
            edges = (float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency']))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid group edges') from exc
        if not all(math.isfinite(x) for x in edges):
            raise ValueError('invalid group edges')
        if not all(row[k] is True for k in gates):
            unstable.append(member)
    if len(bands) > 1:
        raise ValueError('group band pairs differ')
    result = {'stage_index': stage_index, 'stage_count': stage_count,
              'pending_members': pending, 'failed_members': failed, 'unstable_members': unstable,
              'next_stage': None, 'reevaluate_members': []}
    if pending:
        result['status'] = 'pending'
    elif failed:
        result['status'] = 'solver_failure'
    elif not unstable:
        result['status'] = 'admissible'
    elif stage_index + 1 < stage_count:
        result.update(status='group_escalation_required', next_stage=stage_index + 1,
                      reevaluate_members=list(expected_members))
    else:
        result['status'] = 'numerically_unresolved'
    return result


def aggregate_refined_perturbations(
    records: Sequence[Mapping[str, Any]], *, expected_offsets: Sequence[float],
) -> dict[str, Any]:
    """Aggregate one candidate's refined perturbations without promoting failures.

    Each record contains offset, row and provenance from a grid refinement.
    The complete finite offset set and numerical protocol become part of the
    comparison identity; individual solver records remain the evidence source.
    """
    expected = [float(x) for x in expected_offsets]
    if not expected or not all(math.isfinite(x) for x in expected) or len(set(expected)) != len(expected) or 0.0 not in expected:
        raise ValueError("expected offsets must be finite, unique and include zero")
    actual = [float(r['offset']) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("perturbation results do not match expected offsets")
    ordered = sorted(records, key=lambda x: float(x['offset']))
    provenance = [r['provenance'] for r in ordered]
    rows = [r['row'] for r in ordered]
    signatures = {p.get('comparison_contract_sha256') for p in provenance}
    parents = {p.get('parameter_hash') for p in provenance}
    bands = {r.get('lower_band_one_based') for r in rows}
    if len(signatures) != 1 or None in signatures or len(parents) != 1 or None in parents or len(bands) != 1:
        raise ValueError("perturbations differ in protocol, parent or band pair")
    if any(p.get('evaluated_geometry') != 'periodic_binary_pixel_cells_v1' or not p.get('evaluated_grid_sha256') for p in provenance):
        raise ValueError("perturbation grid identity missing")
    edges = [(float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])) for row in rows]
    if not all(math.isfinite(x) for pair in edges for x in pair):
        raise ValueError("refined edges must be finite")
    passed = all(row.get('last_refinement_stable') is True and row.get('resolution_stable') is True
                 and row.get('adaptive_refinement_converged') is True for row in rows)
    contract = {'refinement': next(iter(signatures)), 'offsets': sorted(expected)}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'row': {'lower_band_one_based': next(iter(bands)),
                    'refined_lower_edge_frequency': max(x[0] for x in edges),
                    'refined_upper_edge_frequency': min(x[1] for x in edges),
                    'last_refinement_stable': passed, 'resolution_stable': passed,
                    'adaptive_refinement_converged': passed},
            'provenance': {'comparison_contract_sha256': signature, 'parameter_hash': next(iter(parents)),
                           'offsets': sorted(expected),
                           'grid_sha256': [p['evaluated_grid_sha256'] for p in provenance]},
            'status': 'numerically_stable' if passed else 'numerical_gate_failed'}


def compare_refined_candidates(
    center: Mapping[str, Any], candidate: Mapping[str, Any], *,
    target_frequency: float, predicted_center: float, predicted_candidate: float,
    radius: float, min_radius: float, max_radius: float, acceptance_policy: str = 'ratio',
) -> dict[str, Any]:
    """Compare nominal candidates only after equivalent numerical gates pass.

    Each observation contains one refinement row and its provenance, as emitted
    by ``refine_inverse_candidate``. Inconclusive solves leave the radius and
    incumbent unchanged; caller retains the full solver records separately.
    """
    if not math.isfinite(target_frequency) or target_frequency <= 0:
        raise ValueError("target frequency must be finite and positive")
    if not all(math.isfinite(x) for x in (radius, min_radius, max_radius, predicted_center, predicted_candidate)) or not 0 < min_radius <= radius <= max_radius:
        raise ValueError("invalid trust radius or prediction")
    p0, p1 = center['provenance'], candidate['provenance']
    signature = p0.get('comparison_contract_sha256')
    if not signature or signature != p1.get('comparison_contract_sha256'):
        raise ValueError("refinement comparison contracts do not match")
    r0, r1 = center['row'], candidate['row']
    if r0['lower_band_one_based'] != r1['lower_band_one_based']:
        raise ValueError("refinement band pairs do not match")
    for row in (r0, r1):
        if not (row.get('last_refinement_stable') is True and row.get('resolution_stable') is True
                and row.get('adaptive_refinement_converged') is True):
            return {'status': 'numerical_gate_failed', 'decision': None, 'radius': radius}
    def margin(row: Mapping[str, Any]) -> float:
        lower, upper = float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("refined edges must be finite")
        return 2 * min(target_frequency - lower, upper - target_frequency) / target_frequency
    center_margin, candidate_margin = margin(r0), margin(r1)
    if acceptance_policy not in ('ratio', 'strict_physical_improvement'):
        raise ValueError('unknown acceptance policy')
    if acceptance_policy == 'strict_physical_improvement':
        accepted = candidate_margin > center_margin
        new_radius = radius if accepted else max(min_radius, radius * 0.5)
        decision = {'accepted': accepted, 'ratio': None, 'radius': new_radius,
                    'reason': 'physical_improvement' if accepted else 'physical_nonimprovement'}
        return {'status': 'compared', 'center_margin': center_margin, 'candidate_margin': candidate_margin,
                'decision': decision, 'radius': new_radius}
    decision = update_trust_region(predicted_center=predicted_center, predicted_candidate=predicted_candidate,
                                   actual_center=center_margin, actual_candidate=candidate_margin,
                                   radius=radius, min_radius=min_radius, max_radius=max_radius)
    return {'status': 'compared', 'center_margin': center_margin, 'candidate_margin': candidate_margin,
            'decision': asdict(decision), 'radius': decision.radius}
