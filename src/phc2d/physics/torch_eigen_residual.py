"""Differentiable TE Bloch eigenfield residuals for adapted PINO training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Differentiable eigenfield residuals require optional torch.") from exc
    return torch


def _bloch_difference(values: Any, *, axis: int, spacing: Any, phase: Any) -> Any:
    torch = _torch()
    if axis == 1:
        forward = torch.cat((values[:, 1:, :], values[:, :1, :] * phase[:, None, None]), dim=1)
        backward = torch.cat((values[:, -1:, :] * phase.conj()[:, None, None], values[:, :-1, :]), dim=1)
    elif axis == 2:
        forward = torch.cat((values[:, :, 1:], values[:, :, :1] * phase[:, None, None]), dim=2)
        backward = torch.cat((values[:, :, -1:] * phase.conj()[:, None, None], values[:, :, :-1]), dim=2)
    else:
        raise ValueError("TE Bloch differences support only the two spatial axes.")
    return (forward - backward) / (2.0 * spacing[:, None, None])


def derive_epsilon_on_field_grid_torch(epsilon: Any, field_shape: tuple[int, int]) -> Any:
    """Lift a batched material grid to a batched field grid."""

    if epsilon.ndim != 3:
        raise ValueError("Batched epsilon must have shape (batch, nx, ny).")
    nx, ny = (int(value) for value in field_shape)
    if nx % int(epsilon.shape[1]) or ny % int(epsilon.shape[2]):
        raise ValueError("epsilon-to-field grid ratio must be integral.")
    return epsilon.repeat_interleave(nx // int(epsilon.shape[1]), dim=1).repeat_interleave(ny // int(epsilon.shape[2]), dim=2)


def evaluate_te_bloch_eigen_residual_torch(
    efield: Any,
    *,
    epsilon: Any,
    eigenfrequency: Any,
    dx: Any,
    dy: Any,
    bloch_k_point: Any,
) -> dict[str, Any]:
    """Return batched differentiable residual energies for complex ``Ex/Ey``."""

    torch = _torch()
    if efield.ndim != 4 or int(efield.shape[-1]) != 2 or not torch.is_complex(efield):
        raise ValueError("efield must be a complex tensor with shape (batch, nx, ny, 2).")
    batch, nx, ny, _ = efield.shape
    if epsilon.shape[0] != batch or bloch_k_point.shape != (batch, 2):
        raise ValueError("epsilon and Bloch k-point batches must align with efield.")
    frequency = eigenfrequency.reshape(batch).to(dtype=efield.real.dtype, device=efield.device)
    dx_values = dx.reshape(batch).to(dtype=efield.real.dtype, device=efield.device)
    dy_values = dy.reshape(batch).to(dtype=efield.real.dtype, device=efield.device)
    if torch.any(frequency <= 0) or torch.any(dx_values <= 0) or torch.any(dy_values <= 0):
        raise ValueError("frequency and field spacing must be positive.")
    k_point = bloch_k_point.to(dtype=efield.real.dtype, device=efield.device)
    eps = derive_epsilon_on_field_grid_torch(epsilon.to(dtype=efield.real.dtype, device=efield.device), (int(nx), int(ny)))
    phase_x = torch.exp(2j * torch.pi * k_point[:, 0])
    phase_y = torch.exp(2j * torch.pi * k_point[:, 1])
    ex, ey = efield[..., 0], efield[..., 1]
    curl_z = _bloch_difference(ey, axis=1, spacing=dx_values, phase=phase_x) - _bloch_difference(ex, axis=2, spacing=dy_values, phase=phase_y)
    omega_squared = (2.0 * torch.pi * frequency).square()
    # Keep the curl-curl term explicit: besides the ordinary residual this
    # gives a differentiable Rayleigh-style eigenfrequency for B4's
    # field--frequency consistency loss.  It is intentionally derived from
    # the same Bloch finite difference used by the residual, not a second
    # surrogate or an MPB target.
    curlcurl_x = _bloch_difference(curl_z, axis=2, spacing=dy_values, phase=phase_y)
    curlcurl_y = -_bloch_difference(curl_z, axis=1, spacing=dx_values, phase=phase_x)
    residual_x = curlcurl_x - omega_squared[:, None, None] * eps * ex
    residual_y = curlcurl_y - omega_squared[:, None, None] * eps * ey
    cell = dx_values * dy_values
    residual_density = residual_x.abs().square() + residual_y.abs().square()
    scale_density = (omega_squared[:, None, None] * eps * ex).abs().square() + (omega_squared[:, None, None] * eps * ey).abs().square()
    residual_energy = residual_density.sum(dim=(1, 2)) * cell
    scale_energy = scale_density.sum(dim=(1, 2)) * cell
    relative_squared = residual_energy / scale_energy.clamp_min(1.0e-12)
    boundary_mask = torch.zeros((int(nx), int(ny)), dtype=torch.bool, device=efield.device)
    boundary_mask[[0, -1], :] = True
    boundary_mask[:, [0, -1]] = True
    boundary_energy = residual_density[:, boundary_mask].sum(dim=1) * cell
    boundary_scale = scale_density[:, boundary_mask].sum(dim=1) * cell
    field_l2 = torch.sqrt(
        ((eps[..., None] * efield.abs().square()).sum(dim=(1, 2, 3)) * cell).clamp_min(0.0)
    )
    numerator = (
        (ex.conj() * curlcurl_x + ey.conj() * curlcurl_y).sum(dim=(1, 2)) * cell
    ).real
    denominator = (eps * (ex.abs().square() + ey.abs().square())).sum(dim=(1, 2)) * cell
    rayleigh_omega_squared = numerator / denominator.clamp_min(1.0e-12)
    rayleigh_valid = rayleigh_omega_squared > 0.0
    rayleigh_frequency = torch.sqrt(rayleigh_omega_squared.clamp_min(1.0e-12)) / (2.0 * torch.pi)
    return {
        "relative_residual_squared": relative_squared,
        "relative_residual": torch.sqrt(relative_squared.clamp_min(0.0)),
        "boundary_relative_residual": torch.sqrt((boundary_energy / boundary_scale.clamp_min(1.0e-12)).clamp_min(0.0)),
        "field_l2": field_l2,
        "residual_l2": torch.sqrt(residual_energy.clamp_min(0.0)),
        "rayleigh_frequency": rayleigh_frequency,
        "rayleigh_valid": rayleigh_valid,
    }


def pino_te_physics_loss(residual: dict[str, Any], *, solver_floor_scale: float = 1.0) -> tuple[Any, Any, Any]:
    """Return a resolution-calibrated PINO loss and non-trivial normalization loss.

    ``solver_floor_scale`` is a positive, validation-only real-MPB reference
    floor.  It makes the same finite-difference residual comparable across
    grids without changing the Maxwell equation or using test targets.
    """

    torch = _torch()
    if float(solver_floor_scale) <= 0.0:
        raise ValueError("solver_floor_scale must be positive.")
    residual_loss = torch.log1p(residual["relative_residual_squared"] / float(solver_floor_scale) ** 2).mean()
    normalization_loss = (residual["field_l2"] - 1.0).square().mean()
    return residual_loss + 0.1 * normalization_loss, residual_loss, normalization_loss


__all__ = ["derive_epsilon_on_field_grid_torch", "evaluate_te_bloch_eigen_residual_torch", "pino_te_physics_loss"]
