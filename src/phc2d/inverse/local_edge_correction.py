"""Local correction of surrogate band edges using solver observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalEdgeCorrection:
    """One locally weighted affine residual fit in normalized coordinates."""

    center: np.ndarray
    radius: float
    coefficients: np.ndarray
    comparison_contract_sha256: str
    surrogate_contract_sha256: str
    sample_count: int

    def correct(self, point: Any, predicted_edges: Any) -> np.ndarray:
        """Return corrected lower/upper edges inside the fitted trust radius."""
        position = np.asarray(point, dtype=np.float64)
        edges = np.asarray(predicted_edges, dtype=np.float64)
        if position.shape != self.center.shape or edges.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(edges).all():
            raise ValueError('invalid point or predicted edges')
        displacement = position - self.center
        if np.linalg.norm(displacement) > self.radius + 1e-12:
            raise ValueError('point lies outside the local correction radius')
        return edges + np.r_[1.0, displacement] @ self.coefficients


def fit_local_edge_correction(observations: Sequence[Mapping[str, Any]], *, center: Any,
                              radius: float, ridge: float) -> LocalEdgeCorrection:
    """Fit lower and upper edge residuals only from comparable gated observations.

    Features are caller-normalized parameters. Distance weights and ridge
    regularization are local modeling choices, not uncertainty bounds.
    """
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.ndim != 1 or not center_array.size or not np.isfinite(center_array).all():
        raise ValueError('center must be a finite nonempty vector')
    if not np.isfinite([radius, ridge]).all() or radius <= 0 or ridge <= 0 or not observations:
        raise ValueError('positive radius, ridge and observations required')
    signatures = {r.get('comparison_contract_sha256') for r in observations}
    if len(signatures) != 1 or not next(iter(signatures)):
        raise ValueError('observations require one comparison contract')
    surrogate_signatures = {r.get('surrogate_contract_sha256') for r in observations}
    if len(surrogate_signatures) != 1 or not next(iter(surrogate_signatures)):
        raise ValueError('observations require one surrogate contract')
    design, residuals, distances = [], [], []
    identities = set()
    for row in observations:
        if row.get('numerical_gate_passed') is not True:
            raise ValueError('all correction observations must pass numerical gates')
        identity = row.get('parameter_hash')
        if not identity or identity in identities:
            raise ValueError('observation identities must be unique')
        identities.add(identity)
        point = np.asarray(row['parameters'], dtype=np.float64)
        predicted = np.asarray(row['predicted_edges'], dtype=np.float64)
        actual = np.asarray(row['physical_edges'], dtype=np.float64)
        if point.shape != center_array.shape or predicted.shape != (2,) or actual.shape != (2,) or not all(np.isfinite(v).all() for v in (point, predicted, actual)):
            raise ValueError('invalid observation shapes or values')
        delta = point - center_array
        distance = float(np.linalg.norm(delta))
        if distance > radius + 1e-12:
            raise ValueError('observation lies outside local radius')
        design.append(np.r_[1., delta])
        residuals.append(actual - predicted)
        distances.append(distance)
    x = np.asarray(design)
    y = np.asarray(residuals)
    weights = np.exp(-0.5 * (np.asarray(distances) / radius) ** 2)
    regularizer = np.eye(x.shape[1]) * ridge
    regularizer[0, 0] = 0.
    coefficients = np.linalg.solve(x.T @ (weights[:, None] * x) + regularizer,
                                   x.T @ (weights[:, None] * y))
    return LocalEdgeCorrection(center_array.copy(), float(radius), coefficients,
                               str(next(iter(signatures))), str(next(iter(surrogate_signatures))), len(observations))
