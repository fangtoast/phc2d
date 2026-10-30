"""Sampled common band gaps across manufacturing perturbations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonGap:
    """Sampled common interval; negative width indicates no common gap."""

    lower_edge: float
    upper_edge: float
    signed_width: float
    target_margin: float
    contains_target: bool


def common_gap(bands: np.ndarray, *, lower_band: int, target_frequency: float) -> CommonGap:
    """Evaluate [perturbation, k, band] spectra in physical frequency units.

    ``lower_band`` is one-based. The result describes only the supplied
    perturbations and k points, with no continuous-domain certification.
    """
    values = np.asarray(bands, dtype=np.float64)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("bands must have nonempty [perturbation, k, band] axes")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("bands must contain finite nonnegative physical frequencies")
    if isinstance(lower_band, bool) or not isinstance(lower_band, (int, np.integer)):
        raise ValueError("lower_band must be a one-based integer")
    index = int(lower_band) - 1
    if index < 0 or index + 1 >= values.shape[2]:
        raise ValueError("lower_band must select an adjacent band pair")
    if np.any(values[..., 1:] < values[..., :-1]):
        raise ValueError("bands must be in ascending order at each k point")
    target = float(target_frequency)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_frequency must be finite and positive")
    lower = float(np.max(values[..., index]))
    upper = float(np.min(values[..., index + 1]))
    margin = 2.0 * min(target - lower, upper - target) / target
    return CommonGap(lower, upper, upper - lower, margin, bool(margin > 0.0))
