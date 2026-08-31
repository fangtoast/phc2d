"""Bounded theory-frequency weighting for Full-Wh surrogate objectives.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from phc2d.physics import effective_refractive_index


FREQUENCY_WEIGHTED_LOSS_NAME = "equal_lane_bounded_inverse_frequency_v1"


@dataclass(frozen=True)
class FrequencyWeightedLossConfig:
    """Validated frozen parameters for the bounded inverse-frequency arm."""

    name: str
    epsilon_air: float
    epsilon_material: float
    epsilon_f: float
    rho: float
    alpha: float
    center_kind: str = "heuristic_folded_center_x"
    polarization: str = "TE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epsilon_air": self.epsilon_air,
            "epsilon_material": self.epsilon_material,
            "epsilon_f": self.epsilon_f,
            "rho": self.rho,
            "alpha": self.alpha,
            "center_kind": self.center_kind,
            "polarization": self.polarization,
            "frequency_scale": "physical_Wh_train_normalizer_mean_std",
        }


@dataclass(frozen=True)
class FrequencyWeightedLossTerms:
    """Objective components and detached-ready weight summaries."""

    total: Any
    baseline_mse: Any
    bounded_inverse_weighted_mse: Any
    weight_mean: Any
    weight_min: Any
    weight_max: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "baseline_mse": self.baseline_mse,
            "bounded_inverse_weighted_mse": self.bounded_inverse_weighted_mse,
            "weight_mean": self.weight_mean,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
        }


def resolve_frequency_weighted_loss_config(raw: Any) -> FrequencyWeightedLossConfig | None:
    """Resolve this optional objective without importing Torch or HDF5."""

    if not isinstance(raw, Mapping) or str(raw.get("name", "")) != FREQUENCY_WEIGHTED_LOSS_NAME:
        return None
    allowed = {
        "name", "epsilon_air", "epsilon_material", "epsilon_f", "rho", "alpha",
        "center_kind", "polarization",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown frequency-weighted loss fields: {sorted(unknown)}")
    required = {"epsilon_air", "epsilon_material", "epsilon_f", "rho", "alpha"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"frequency-weighted loss missing required fields: {missing}")
    epsilon_air = float(raw["epsilon_air"])
    epsilon_material = float(raw["epsilon_material"])
    epsilon_f = float(raw["epsilon_f"])
    rho = float(raw["rho"])
    alpha = float(raw["alpha"])
    if not np.isfinite([epsilon_air, epsilon_material, epsilon_f, rho, alpha]).all():
        raise ValueError("frequency-weighted loss parameters must be finite")
    if epsilon_air <= 0.0 or epsilon_material <= epsilon_air:
        raise ValueError("frequency-weighted loss requires epsilon_material > epsilon_air > 0")
    if epsilon_f <= 0.0:
        raise ValueError("frequency-weighted loss epsilon_f must be positive")
    if rho < 0.0:
        raise ValueError("frequency-weighted loss rho must be nonnegative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("frequency-weighted loss alpha must lie in [0, 1]")
    center_kind = str(raw.get("center_kind", "heuristic_folded_center_x"))
    if center_kind != "heuristic_folded_center_x":
        raise ValueError("frequency-weighted loss only supports heuristic_folded_center_x")
    polarization = str(raw.get("polarization", "TE")).upper()
    if polarization != "TE":
        raise ValueError("frequency-weighted loss requires the TE effective-index contract")
    return FrequencyWeightedLossConfig(
        name=FREQUENCY_WEIGHTED_LOSS_NAME,
        epsilon_air=epsilon_air,
        epsilon_material=epsilon_material,
        epsilon_f=epsilon_f,
        rho=rho,
        alpha=alpha,
        center_kind=center_kind,
        polarization=polarization,
    )


def frequency_center_from_material_weight(
    material_weight: np.ndarray,
    *,
    config: FrequencyWeightedLossConfig,
) -> np.ndarray:
    """Compute $f_0=1/(2n_eff)$ from frozen material-weight rasters."""

    weight = np.asarray(material_weight, dtype=np.float64)
    if weight.ndim < 2:
        raise ValueError("material_weight must expose at least two spatial axes")
    if not np.isfinite(weight).all() or np.any(weight < -1.0e-6) or np.any(weight > 1.0 + 1.0e-6):
        raise ValueError("material_weight must be finite and lie in [0, 1]")
    epsilon = config.epsilon_air + (config.epsilon_material - config.epsilon_air) * np.clip(weight, 0.0, 1.0)
    index = effective_refractive_index(epsilon, config.polarization)
    center = 1.0 / (2.0 * np.asarray(index, dtype=np.float64))
    if not np.isfinite(center).all() or np.any(center <= 0.0):
        raise RuntimeError("effective-index frequency centers must be finite and positive")
    return center


def precompute_frequency_centers(
    path: str | Any,
    *,
    grid_field: str,
    indices: np.ndarray,
    config: FrequencyWeightedLossConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only requested raster rows once and return centers plus sample ids."""

    import h5py

    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("frequency-center indices must be a non-empty vector")
    with h5py.File(path, "r") as handle:
        if grid_field not in handle:
            raise KeyError(f"frequency-center grid field is missing: {grid_field}")
        material = np.asarray(handle[grid_field][selected], dtype=np.float64)
        sample_ids = (
            np.asarray(handle["sample_id"][selected]).astype("U")
            if "sample_id" in handle
            else selected.astype("U")
        )
    return frequency_center_from_material_weight(material, config=config).astype(np.float32), sample_ids


def bounded_inverse_frequency_loss(
    prediction: Any,
    target: Any,
    *,
    target_mean: Any,
    target_std: Any,
    frequency_center: Any,
    config: FrequencyWeightedLossConfig,
    torch_module: Any | None = None,
) -> FrequencyWeightedLossTerms:
    """Blend full normalized MSE with bounded physical-frequency weighting."""

    if torch_module is None:
        import torch as torch_module
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("frequency-weighted loss expects matching normalized [batch, k, band] tensors")
    mean = torch_module.as_tensor(target_mean, dtype=prediction.dtype, device=prediction.device)
    std = torch_module.as_tensor(target_std, dtype=prediction.dtype, device=prediction.device)
    if tuple(mean.shape) != tuple(prediction.shape[1:]) or tuple(std.shape) != tuple(prediction.shape[1:]):
        raise ValueError("target normalizers must match the [k, band] target shape")
    center = torch_module.as_tensor(frequency_center, dtype=prediction.dtype, device=prediction.device)
    if tuple(center.shape) != (int(prediction.shape[0]),):
        raise ValueError("frequency_center must have one value per batch sample")
    if not bool(torch_module.isfinite(center).all().item()) or bool((center <= 0.0).any().item()):
        raise ValueError("frequency_center must be finite and positive")
    physical_target = target * std + mean
    raw = config.epsilon_f / (torch_module.abs(physical_target - center[:, None, None]) + config.epsilon_f)
    weight = (1.0 + config.rho * raw)
    weight = weight / weight.mean(dim=(1, 2), keepdim=True)
    baseline = torch_module.mean((prediction - target) ** 2)
    weighted = torch_module.mean(weight * (prediction - target) ** 2)
    total = (1.0 - config.alpha) * baseline + config.alpha * weighted
    return FrequencyWeightedLossTerms(
        total=total,
        baseline_mse=baseline,
        bounded_inverse_weighted_mse=weighted,
        weight_mean=weight.mean(),
        weight_min=weight.amin(),
        weight_max=weight.amax(),
    )


__all__ = [
    "FREQUENCY_WEIGHTED_LOSS_NAME",
    "FrequencyWeightedLossConfig",
    "FrequencyWeightedLossTerms",
    "bounded_inverse_frequency_loss",
    "frequency_center_from_material_weight",
    "precompute_frequency_centers",
    "resolve_frequency_weighted_loss_config",
]
