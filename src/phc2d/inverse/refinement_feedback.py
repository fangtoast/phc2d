"""Gate trust-region feedback using paired refined solver observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

from phc2d.inverse.trust_region import update_trust_region


def compare_refined_candidates(
    center: Mapping[str, Any], candidate: Mapping[str, Any], *,
    target_frequency: float, predicted_center: float, predicted_candidate: float,
    radius: float, min_radius: float, max_radius: float,
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
    decision = update_trust_region(predicted_center=predicted_center, predicted_candidate=predicted_candidate,
                                   actual_center=center_margin, actual_candidate=candidate_margin,
                                   radius=radius, min_radius=min_radius, max_radius=max_radius)
    return {'status': 'compared', 'center_margin': center_margin, 'candidate_margin': candidate_margin,
            'decision': asdict(decision), 'radius': decision.radius}
