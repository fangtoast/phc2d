"""Bind physical observations to the smooth-edge correction contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def composite_edge_observation(*, parameter_hash: str, parameters: Any, physical_edges: Any,
                               hard_edges: Any, soft_exact_edges: Any, soft_smooth_edges: Any,
                               comparison_contract_sha256: str, surrogate_contract_sha256: str,
                               numerical_gate_passed: bool) -> dict[str, Any]:
    """Create one fit-compatible observation and its three-part residual decomposition.

    Physical bands must come from the same hard geometry as ``hard_edges``.
    The caller authenticates the physical result, source geometry, temperature,
    smoothing, model version and comparison contract before this conversion.
    """
    point = np.asarray(parameters, dtype=np.float64)
    arrays = [np.asarray(item, dtype=np.float64) for item in
              (physical_edges, hard_edges, soft_exact_edges, soft_smooth_edges)]
    if (not parameter_hash or not comparison_contract_sha256 or not surrogate_contract_sha256
            or numerical_gate_passed is not True or point.ndim != 1 or not point.size
            or not np.isfinite(point).all() or any(a.shape != (2,) or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('invalid or ungated composite edge observation')
    physical, hard, exact_soft, smooth = arrays
    return {'parameter_hash': parameter_hash, 'parameters': point.tolist(),
            'physical_edges': physical.tolist(), 'predicted_edges': smooth.tolist(),
            'hard_edges': hard.tolist(), 'soft_exact_edges': exact_soft.tolist(),
            'residual_physical_minus_hard': (physical - hard).tolist(),
            'residual_hard_minus_soft_exact': (hard - exact_soft).tolist(),
            'residual_soft_exact_minus_smooth': (exact_soft - smooth).tolist(),
            'composite_residual': (physical - smooth).tolist(),
            'comparison_contract_sha256': comparison_contract_sha256,
            'surrogate_contract_sha256': surrogate_contract_sha256,
            'numerical_gate_passed': True}
