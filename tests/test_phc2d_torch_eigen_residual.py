"""Tests for differentiable TE Bloch residuals used by adapted PINO.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from phc2d.forward.field_eigen_v2 import phase_invariant_field_loss
from phc2d.models import build_model
from phc2d.physics import evaluate_te_bloch_eigen_residual, evaluate_te_bloch_eigen_residual_torch, pino_te_physics_loss


torch = pytest.importorskip("torch")


def _plane_wave(nx: int = 19, ny: int = 13):
    dx, dy, kx = 1.0 / nx, 1.0 / ny, 0.23
    x = np.arange(nx, dtype=np.float64)[:, None] * dx
    field = np.zeros((nx, ny, 1, 3), dtype=np.complex128)
    field[..., 1] = np.exp(2j * np.pi * kx * x)[:, :, None]
    frequency = abs(np.sin(2.0 * np.pi * kx / nx) / dx) / (2.0 * np.pi)
    return field, frequency, dx, dy, np.asarray([kx, 0.0])


def test_torch_residual_matches_numpy_and_backpropagates() -> None:
    field, frequency, dx, dy, k = _plane_wave()
    expected = evaluate_te_bloch_eigen_residual(field, epsilon=np.ones(field.shape[:2]), eigenfrequency=frequency, dx=dx, dy=dy, bloch_k_point=k)
    active = torch.tensor(field[None, ..., 0, :2], dtype=torch.complex64, requires_grad=True)
    actual = evaluate_te_bloch_eigen_residual_torch(
        active,
        epsilon=torch.ones((1, field.shape[0], field.shape[1])),
        eigenfrequency=torch.tensor([frequency]),
        dx=torch.tensor([dx]),
        dy=torch.tensor([dy]),
        bloch_k_point=torch.tensor(k[None], dtype=torch.float32),
    )
    assert float(actual["relative_residual"][0].detach()) == pytest.approx(expected["relative_residual"], abs=2.0e-5)
    loss, _, _ = pino_te_physics_loss(actual)
    loss.backward()
    assert active.grad is not None
    assert torch.isfinite(active.grad).all()


def test_wrong_bloch_phase_rises_and_zero_field_is_penalized() -> None:
    field, frequency, dx, dy, k = _plane_wave()
    active = torch.tensor(field[None, ..., 0, :2], dtype=torch.complex64)
    kwargs = dict(
        epsilon=torch.ones((1, field.shape[0], field.shape[1])),
        eigenfrequency=torch.tensor([frequency]),
        dx=torch.tensor([dx]),
        dy=torch.tensor([dy]),
    )
    correct = evaluate_te_bloch_eigen_residual_torch(active, bloch_k_point=torch.tensor(k[None], dtype=torch.float32), **kwargs)
    wrong = evaluate_te_bloch_eigen_residual_torch(active, bloch_k_point=torch.zeros((1, 2)), **kwargs)
    assert float(wrong["relative_residual"][0]) > float(correct["relative_residual"][0])
    zero = evaluate_te_bloch_eigen_residual_torch(torch.zeros_like(active), bloch_k_point=torch.tensor(k[None], dtype=torch.float32), **kwargs)
    _, _, normalization = pino_te_physics_loss(zero)
    assert float(normalization) == pytest.approx(1.0)


def test_field_norm_and_rayleigh_frequency_use_epsilon_metric() -> None:
    field, frequency, dx, dy, k = _plane_wave()
    active = torch.tensor(field[None, ..., 0, :2], dtype=torch.complex64)
    residual = evaluate_te_bloch_eigen_residual_torch(
        active,
        epsilon=torch.full((1, field.shape[0], field.shape[1]), 4.0),
        eigenfrequency=torch.tensor([frequency / 2.0]),
        dx=torch.tensor([dx]),
        dy=torch.tensor([dy]),
        bloch_k_point=torch.tensor(k[None], dtype=torch.float32),
    )
    assert float(residual["field_l2"][0]) == pytest.approx(2.0, abs=1.0e-5)
    assert float(residual["rayleigh_frequency"][0]) == pytest.approx(frequency / 2.0, rel=2.0e-5)


def test_pino_training_chain_backpropagates_through_fno_field_and_frequency() -> None:
    torch.manual_seed(7)
    model = build_model("fno_field_v2", grid_channels=5, context_dim=3, field_channels=6, width=8, depth=1, modes=(4, 4))
    grid = torch.randn(2, 5, 13, 11)
    context = torch.tensor([[0.2, 0.0, 0.25], [0.0, 0.3, 0.75]], dtype=torch.float32)
    target = torch.randn(2, 6, 13, 11)
    predicted, predicted_frequency_z = model(grid, context)
    epsilon = torch.ones((2, 13, 11))
    data_loss, *_ = phase_invariant_field_loss(torch, predicted, target, epsilon=epsilon)
    moved = predicted.permute(0, 2, 3, 1)
    active = torch.complex(moved[..., :3], moved[..., 3:])
    predicted_frequency = torch.exp(predicted_frequency_z * 0.1 + np.log(0.3))
    residual = evaluate_te_bloch_eigen_residual_torch(
        active[..., :2],
        epsilon=epsilon,
        eigenfrequency=predicted_frequency,
        dx=torch.full((2,), 1.0 / 13),
        dy=torch.full((2,), 1.0 / 11),
        bloch_k_point=context[:, :2],
    )
    physics_total, physics_loss, normalization_loss = pino_te_physics_loss(residual)
    total = data_loss + 1.0e-3 * physics_total
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)
    assert torch.isfinite(physics_loss)
    assert torch.isfinite(normalization_loss)


def test_pino_loss_applies_positive_solver_floor_scale() -> None:
    residual = {
        "relative_residual_squared": torch.tensor([4.0]),
        "field_l2": torch.tensor([1.0]),
    }
    uncalibrated, _, _ = pino_te_physics_loss(residual)
    calibrated, _, _ = pino_te_physics_loss(residual, solver_floor_scale=2.0)
    assert calibrated < uncalibrated
    with pytest.raises(ValueError, match="positive"):
        pino_te_physics_loss(residual, solver_floor_scale=0.0)
