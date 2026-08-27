"""Work-frequency priors and bounded weighting primitives for Full-Wh tasks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


BraggDirection = Literal["X", "M"]
WeightKind = Literal["inverse_distance", "gaussian"]


def mean_refractive_index(
    material_weight: np.ndarray,
    *,
    epsilon_air: float,
    epsilon_material: float,
) -> np.ndarray:
    """Return mean sqrt(epsilon) over the final two material-grid axes."""

    weight = np.asarray(material_weight, dtype=np.float64)
    if weight.ndim < 2:
        raise ValueError("material_weight must expose at least two spatial axes")
    if not np.isfinite(weight).all():
        raise ValueError("material_weight must be finite")
    if float(epsilon_air) <= 0.0 or float(epsilon_material) <= 0.0:
        raise ValueError("permittivities must be positive")
    if np.any(weight < -1.0e-6) or np.any(weight > 1.0 + 1.0e-6):
        raise ValueError("material_weight must lie in [0, 1]")
    clipped = np.clip(weight, 0.0, 1.0)
    epsilon = float(epsilon_air) + (float(epsilon_material) - float(epsilon_air)) * clipped
    return np.mean(np.sqrt(epsilon), axis=(-2, -1))


def bragg_frequency(
    material_weight: np.ndarray,
    *,
    epsilon_air: float,
    epsilon_material: float,
    direction: BraggDirection = "X",
) -> np.ndarray:
    """Return the first-shell square-lattice Bragg scale in MPB a/lambda units."""

    n_bar = mean_refractive_index(
        material_weight,
        epsilon_air=epsilon_air,
        epsilon_material=epsilon_material,
    )
    if direction == "X":
        factor = 1.0
    elif direction == "M":
        factor = np.sqrt(2.0)
    else:
        raise ValueError("direction must be 'X' or 'M'")
    return factor / (2.0 * n_bar)


def positive_adjacent_band_spacings(frequencies: np.ndarray) -> np.ndarray:
    """Return finite positive adjacent-band spacings at fixed k coordinates."""

    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("frequencies must end in a band axis of length at least two")
    if not np.isfinite(values).all():
        raise ValueError("frequencies must be finite")
    spacing = np.diff(values, axis=-1)
    return spacing[spacing > 0.0]


def nearest_frequency_distance(
    frequencies: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Return each sample's nearest Full-Wh coordinate distance to its center."""

    values = np.asarray(frequencies, dtype=np.float64)
    centers = np.asarray(center, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("frequencies must include sample and coordinate axes")
    if centers.shape != values.shape[:1]:
        raise ValueError(f"center shape {centers.shape} does not match sample axis {values.shape[:1]}")
    flat = values.reshape(values.shape[0], -1)
    return np.min(np.abs(flat - centers[:, None]), axis=1)


def normalized_frequency_weights(
    frequencies: np.ndarray,
    center: np.ndarray,
    *,
    kind: WeightKind,
    epsilon: float,
    sigma: float | None = None,
    rho: float = 3.0,
) -> np.ndarray:
    """Build finite per-sample weights normalized to coordinate mean one."""

    values = np.asarray(frequencies, dtype=np.float64)
    centers = np.asarray(center, dtype=np.float64)
    if values.ndim < 2 or centers.shape != values.shape[:1]:
        raise ValueError("frequencies and center must share the sample axis")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    distance = np.abs(values - centers.reshape((-1,) + (1,) * (values.ndim - 1)))
    if kind == "inverse_distance":
        raw = 1.0 / (distance + float(epsilon))
    elif kind == "gaussian":
        if sigma is None or float(sigma) <= 0.0:
            raise ValueError("positive sigma is required for gaussian weights")
        if float(rho) < 0.0:
            raise ValueError("rho must be nonnegative")
        raw = 1.0 + float(rho) * np.exp(-(distance * distance) / (2.0 * float(sigma) ** 2))
    else:
        raise ValueError(f"unsupported weight kind: {kind}")
    axes = tuple(range(1, raw.ndim))
    weights = raw / np.mean(raw, axis=axes, keepdims=True)
    if not np.isfinite(weights).all():
        raise RuntimeError("frequency weighting produced non-finite values")
    return weights


__all__ = [
    "BraggDirection",
    "WeightKind",
    "bragg_frequency",
    "mean_refractive_index",
    "nearest_frequency_distance",
    "normalized_frequency_weights",
    "positive_adjacent_band_spacings",
]
