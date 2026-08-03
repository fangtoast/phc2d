"""Tests for schema-derived TE Bloch eigenfield residuals.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from phc2d.physics import derive_epsilon_on_field_grid, evaluate_te_bloch_eigen_residual, make_deterministic_random_efield


def _discrete_bloch_plane_wave(nx: int = 29, ny: int = 17) -> tuple[np.ndarray, float, np.ndarray]:
    dx, dy = 1.0 / nx, 1.0 / ny
    reduced_kx = 0.23
    x = np.arange(nx, dtype=np.float64)[:, None] * dx
    field = np.zeros((nx, ny, 1, 3), dtype=np.complex128)
    field[..., 1] = np.exp(2j * np.pi * reduced_kx * x)[:, :, None]
    discrete_omega = abs(np.sin(2.0 * np.pi * reduced_kx / nx) / dx)
    return field, discrete_omega / (2.0 * np.pi), np.asarray([reduced_kx, 0.0])


def test_bloch_plane_wave_residual_is_at_discrete_floor_and_wrong_phase_rises() -> None:
    field, frequency, k_point = _discrete_bloch_plane_wave()
    epsilon = np.ones((field.shape[0], field.shape[1]), dtype=np.float32)
    correct = evaluate_te_bloch_eigen_residual(field, epsilon=epsilon, eigenfrequency=frequency, dx=1.0 / field.shape[0], dy=1.0 / field.shape[1], bloch_k_point=k_point)
    wrong = evaluate_te_bloch_eigen_residual(field, epsilon=epsilon, eigenfrequency=frequency, dx=1.0 / field.shape[0], dy=1.0 / field.shape[1], bloch_k_point=(0.0, 0.0))
    assert correct["relative_residual"] < 1.0e-10
    assert wrong["relative_residual"] > correct["relative_residual"] * 1.0e6


def test_epsilon_grid_derivation_is_shape_driven_and_rejects_noninteger_ratio() -> None:
    epsilon = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    lifted = derive_epsilon_on_field_grid(epsilon, (6, 8))
    assert lifted.shape == (6, 8)
    assert np.all(lifted[:3, :4] == 1.0)
    with pytest.raises(ValueError, match="non-integer ratio"):
        derive_epsilon_on_field_grid(epsilon, (5, 8))


def test_random_field_baseline_is_record_deterministic_and_normalized() -> None:
    epsilon = np.full((7, 9), 4.0)
    first = make_deterministic_random_efield((7, 9, 1, 3), record_id="field-a", epsilon=epsilon, dx=1.0 / 7, dy=1.0 / 9)
    second = make_deterministic_random_efield((7, 9, 1, 3), record_id="field-a", epsilon=epsilon, dx=1.0 / 7, dy=1.0 / 9)
    assert np.array_equal(first, second)
    assert np.sqrt(np.sum(epsilon[..., None, None] * np.abs(first) ** 2) / (7 * 9)) == pytest.approx(1.0)
