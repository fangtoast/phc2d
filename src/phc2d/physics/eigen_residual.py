"""Schema-derived TE Bloch eigenfield residuals.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


def _spatial_shape(field: np.ndarray) -> tuple[int, int]:
    values = np.asarray(field)
    if values.ndim < 3 or values.shape[-1] < 2:
        raise ValueError("TE eigenfield must expose two spatial axes and at least Ex/Ey components.")
    nx, ny = int(values.shape[0]), int(values.shape[1])
    if nx <= 1 or ny <= 1:
        raise ValueError("TE residual requires at least two field cells per spatial axis.")
    return nx, ny


def derive_epsilon_on_field_grid(epsilon: np.ndarray, field: np.ndarray | Sequence[int]) -> np.ndarray:
    """Lift a 2D material grid to the serialized field grid by integer cell replication."""

    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim != 2 or min(eps.shape) <= 0:
        raise ValueError("epsilon must be a non-empty two-dimensional material grid.")
    shape = tuple(int(value) for value in (field if isinstance(field, tuple) else _spatial_shape(np.asarray(field))))
    if len(shape) != 2:
        raise ValueError("field spatial shape must contain exactly two axes.")
    nx, ny = shape
    if nx % eps.shape[0] or ny % eps.shape[1]:
        raise ValueError(f"Cannot derive field epsilon grid {shape} from material grid {eps.shape}: non-integer ratio.")
    return np.repeat(np.repeat(eps, nx // eps.shape[0], axis=0), ny // eps.shape[1], axis=1)


def _bloch_central_difference(values: np.ndarray, *, axis: int, spacing: float, bloch_phase: complex) -> np.ndarray:
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Field spacing must be positive and finite.")
    data = np.asarray(values, dtype=np.complex128)
    forward = np.roll(data, -1, axis=axis)
    backward = np.roll(data, 1, axis=axis)
    edge_forward = [slice(None)] * data.ndim
    edge_backward = [slice(None)] * data.ndim
    edge_forward[axis] = -1
    edge_backward[axis] = 0
    # F(x + L) = exp(i*2pi*k) F(x); the matching negative wrap is its inverse.
    forward[tuple(edge_forward)] = data.take(indices=0, axis=axis) * bloch_phase
    backward[tuple(edge_backward)] = data.take(indices=-1, axis=axis) * np.conjugate(bloch_phase)
    return (forward - backward) / (2.0 * float(spacing))


def canonicalize_complex_efield(efield: np.ndarray, *, epsilon: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Normalize one E field under ``integral E* epsilon E`` and fix its phase."""

    field = np.asarray(efield, dtype=np.complex128)
    nx, ny = _spatial_shape(field)
    eps = derive_epsilon_on_field_grid(epsilon, (nx, ny))
    while eps.ndim < field.ndim:
        eps = eps[..., None]
    l2 = float(np.sqrt(np.sum(eps * np.abs(field) ** 2) * float(dx) * float(dy)))
    if not np.isfinite(l2) or l2 <= 0.0:
        raise ValueError("Cannot canonicalize a zero or non-finite E field.")
    normalized = field / l2
    anchor = normalized.reshape(-1)[int(np.argmax(np.abs(normalized.reshape(-1))))]
    return normalized * np.exp(-1j * np.angle(anchor))


def make_deterministic_random_efield(
    shape: Sequence[int], *, record_id: str, epsilon: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Create a record-keyed random complex field under the frozen normalization."""

    dims = tuple(int(value) for value in shape)
    if len(dims) < 3 or dims[-1] < 2:
        raise ValueError("Random TE E field shape must include Ex/Ey components.")
    seed = int.from_bytes(hashlib.sha256(str(record_id).encode("utf-8")).digest()[:8], "little", signed=False)
    generator = np.random.default_rng(seed)
    values = generator.standard_normal(dims) + 1j * generator.standard_normal(dims)
    if dims[-1] > 2:
        values[..., 2:] = 0.0
    return canonicalize_complex_efield(values, epsilon=epsilon, dx=dx, dy=dy)


def evaluate_te_bloch_eigen_residual(
    efield: np.ndarray,
    *,
    epsilon: np.ndarray,
    eigenfrequency: float,
    dx: float,
    dy: float,
    bloch_k_point: Sequence[float],
) -> dict[str, Any]:
    """Evaluate the dimensionless TE equation ``curl curl E - omega^2 eps E``.

    The finite difference is intentionally evaluated on the stored field grid;
    it never assumes a particular material or field resolution.
    """

    field = np.asarray(efield, dtype=np.complex128)
    nx, ny = _spatial_shape(field)
    k_point = np.asarray(bloch_k_point, dtype=np.float64).reshape(-1)
    if k_point.size < 2 or not np.all(np.isfinite(k_point[:2])):
        raise ValueError("Bloch k-point must provide two finite reduced coordinates.")
    frequency = float(eigenfrequency)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("Eigenfrequency must be positive and finite.")
    eps = derive_epsilon_on_field_grid(epsilon, (nx, ny))
    phase_x = np.exp(2j * np.pi * k_point[0])
    phase_y = np.exp(2j * np.pi * k_point[1])
    ex, ey = field[..., 0], field[..., 1]
    epsilon_for_field = eps
    while epsilon_for_field.ndim < ex.ndim:
        epsilon_for_field = epsilon_for_field[..., None]
    curl_z = _bloch_central_difference(ey, axis=0, spacing=dx, bloch_phase=phase_x) - _bloch_central_difference(ex, axis=1, spacing=dy, bloch_phase=phase_y)
    omega_squared = (2.0 * np.pi * frequency) ** 2
    residual_x = _bloch_central_difference(curl_z, axis=1, spacing=dy, bloch_phase=phase_y) - omega_squared * epsilon_for_field * ex
    residual_y = -_bloch_central_difference(curl_z, axis=0, spacing=dx, bloch_phase=phase_x) - omega_squared * epsilon_for_field * ey
    residual_energy = float(np.sum(np.abs(residual_x) ** 2 + np.abs(residual_y) ** 2) * float(dx) * float(dy))
    scale_energy = float(np.sum(np.abs(omega_squared * epsilon_for_field * ex) ** 2 + np.abs(omega_squared * epsilon_for_field * ey) ** 2) * float(dx) * float(dy))
    relative = float(np.sqrt(residual_energy) / max(np.sqrt(scale_energy), 1.0e-12))
    boundary_mask = np.zeros((nx, ny), dtype=bool)
    boundary_mask[[0, -1], :] = True
    boundary_mask[:, [0, -1]] = True
    boundary_energy = float(np.sum((np.abs(residual_x) ** 2 + np.abs(residual_y) ** 2)[boundary_mask]) * float(dx) * float(dy))
    boundary_scale = float(np.sum((np.abs(omega_squared * epsilon_for_field * ex) ** 2 + np.abs(omega_squared * epsilon_for_field * ey) ** 2)[boundary_mask]) * float(dx) * float(dy))
    return {
        "relative_residual": relative,
        "boundary_relative_residual": float(np.sqrt(boundary_energy) / max(np.sqrt(boundary_scale), 1.0e-12)),
        "residual_l2": float(np.sqrt(residual_energy)),
        "omega": float(2.0 * np.pi * frequency),
        "field_spatial_shape": [nx, ny],
        "epsilon_spatial_shape": [int(epsilon.shape[0]), int(epsilon.shape[1])],
        "epsilon_upsample_ratio": [nx // int(epsilon.shape[0]), ny // int(epsilon.shape[1])],
        "bloch_phase": [[float(np.real(phase_x)), float(np.imag(phase_x))], [float(np.real(phase_y)), float(np.imag(phase_y))]],
    }
