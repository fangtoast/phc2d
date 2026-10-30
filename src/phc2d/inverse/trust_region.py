"""Trust-region decisions for solver-checked inverse optimization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrustDecision:
    """Decision for a maximization step evaluated at one consistent fidelity."""

    accepted: bool
    ratio: float | None
    radius: float
    reason: str


def update_trust_region(
    *, predicted_center: float, predicted_candidate: float,
    actual_center: float, actual_candidate: float, radius: float,
    min_radius: float, max_radius: float, predicted_tolerance: float = 1e-12,
    acceptance_ratio: float = 0.1, expansion_ratio: float = 0.75,
    shrink_factor: float = 0.5, expansion_factor: float = 2.0,
) -> TrustDecision:
    """Compare physical and predicted gains, retaining the verified incumbent on rejection.

    Center and candidate physical scores must use the same solver fidelity,
    target and perturbation set. Radius is normalized parameter-space distance.
    """
    numbers = (predicted_center, predicted_candidate, actual_center, actual_candidate,
               radius, min_radius, max_radius, predicted_tolerance, acceptance_ratio,
               expansion_ratio, shrink_factor, expansion_factor)
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("trust-region inputs must be finite")
    if not 0 < min_radius <= radius <= max_radius:
        raise ValueError("invalid radius bounds")
    if not (0 <= acceptance_ratio < expansion_ratio and predicted_tolerance >= 0
            and 0 < shrink_factor < 1 and expansion_factor > 1):
        raise ValueError("invalid trust-region policy")
    predicted_gain = predicted_candidate - predicted_center
    actual_gain = actual_candidate - actual_center
    smaller = max(min_radius, radius * shrink_factor)
    if predicted_gain <= predicted_tolerance:
        return TrustDecision(False, None, smaller, "nonpositive_predicted_gain")
    ratio = actual_gain / predicted_gain
    if not math.isfinite(ratio):
        return TrustDecision(False, None, smaller, "nonfinite_ratio")
    if actual_gain <= 0 or ratio < acceptance_ratio:
        return TrustDecision(False, ratio, smaller, "physical_improvement_not_confirmed")
    if ratio >= expansion_ratio:
        return TrustDecision(True, ratio, min(max_radius, radius * expansion_factor), "accepted_expand")
    return TrustDecision(True, ratio, radius, "accepted_retain")
