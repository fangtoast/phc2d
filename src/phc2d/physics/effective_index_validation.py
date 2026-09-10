"""Numerical validation primitives for effective-index physical audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from phc2d.forward.full_cell_metrics import enumerate_adjacent_gaps
from phc2d.physics.effective_medium import folded_square_lattice_frequencies


CANDIDATES = ("gamma_g1", "x", "m")


@dataclass(frozen=True)
class SlopeFit:
    """One through-origin near-Gamma slope fit and stability diagnostic."""

    slope: float
    effective_index: float
    r2_origin: float
    residual_rmse: float
    inner_slope: float
    window_drift: float


def build_near_gamma_k_points(
    kappa: Sequence[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build nine unique equal-norm Gamma-X and Gamma-M reduced coordinates."""

    values = np.asarray(kappa, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or values[0] != 0.0:
        raise ValueError("kappa must be a one-dimensional sequence beginning at zero")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("kappa must be strictly increasing after Gamma")
    x = np.stack((values, np.zeros_like(values)), axis=1)
    diagonal = values / np.sqrt(2.0)
    m = np.stack((diagonal, diagonal), axis=1)
    unique = np.concatenate((x, m[1:]), axis=0)
    return unique, {
        "x_indices": np.arange(len(values), dtype=np.int64),
        "m_indices": np.concatenate(
            (np.asarray([0], dtype=np.int64), np.arange(len(values), len(unique)))
        ),
    }


def fit_gamma_slope(kappa: Sequence[float], frequencies: Sequence[float]) -> SlopeFit:
    """Fit Band-1 Delta-frequency through the origin with an inner-window check."""

    x = np.asarray(kappa, dtype=np.float64)
    raw = np.asarray(frequencies, dtype=np.float64)
    if x.shape != raw.shape or len(x) < 4 or x[0] != 0.0:
        raise ValueError("kappa and frequencies must align and include Gamma")
    if not np.isfinite(raw).all():
        raise ValueError("slope frequencies must be finite")
    y = raw - raw[0]

    def slope_for(stop: int) -> float:
        selected_x = x[1:stop]
        selected_y = y[1:stop]
        slope = float(
            np.dot(selected_x, selected_y) / float(np.dot(selected_x, selected_x))
        )
        if slope <= 0.0:
            raise ValueError("near-Gamma Band-1 slope must be positive")
        return slope

    slope = slope_for(len(x))
    inner = slope_for(len(x) - 1)
    residual = y[1:] - slope * x[1:]
    denominator = float(np.dot(y[1:], y[1:]))
    r2 = 1.0 - float(np.dot(residual, residual)) / max(denominator, 1.0e-30)
    return SlopeFit(
        slope=slope,
        effective_index=1.0 / slope,
        r2_origin=r2,
        residual_rmse=float(np.sqrt(np.mean(residual * residual))),
        inner_slope=inner,
        window_drift=abs(inner - slope) / slope,
    )


def candidate_values(index: float) -> dict[str, float]:
    """Return scalar folded-frequency candidates as JSON-compatible values."""

    values = folded_square_lattice_frequencies(index)
    return {name: float(getattr(values, name)) for name in CANDIDATES}


def nearest_gap(Wh: np.ndarray, frequency: float, spacing_floor: float) -> dict[str, Any]:
    """Match a frequency to the closest adjacent-gap center."""

    gaps = enumerate_adjacent_gaps(Wh)
    selected = min(gaps, key=lambda row: (abs(row.center - frequency), row.lower_band_one_based))
    distance = abs(selected.center - frequency)
    edge_distance = min(
        abs(selected.lower_edge_frequency - frequency),
        abs(selected.upper_edge_frequency - frequency),
    )
    return {
        **asdict(selected),
        "center_distance": distance,
        "edge_distance": edge_distance,
        "normalized_center_distance": distance / max(abs(selected.signed_width), spacing_floor),
        "inside_complete_gap": bool(
            selected.status == "complete_gap"
            and selected.lower_edge_frequency <= frequency <= selected.upper_edge_frequency
        ),
        "out_of_frequency_range": bool(frequency < np.min(Wh) or frequency > np.max(Wh)),
    }


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int = 10_000
) -> tuple[float, float]:
    """Return a deterministic paired-bootstrap confidence interval for a mean."""

    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 500):
        stop = min(start + 500, int(replicates))
        draws = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def total_variation(left: Sequence[int], right: Sequence[int], band_count: int) -> float:
    """Return total-variation distance between empirical band-pair distributions."""

    a = np.bincount(np.asarray(left, dtype=np.int64), minlength=band_count) / max(len(left), 1)
    b = np.bincount(np.asarray(right, dtype=np.int64), minlength=band_count) / max(len(right), 1)
    return 0.5 * float(np.sum(np.abs(a - b)))


def analyze_gap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lanes: Sequence[str],
    band_count: int,
) -> dict[str, Any]:
    """Aggregate analytic old-versus-TE gap-alignment rows."""

    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        metrics[candidate] = {}
        for lane in lanes:
            train = [row for row in selected if row["lane"] == lane and row["split"] == "train"]
            validation = [
                row for row in selected if row["lane"] == lane and row["split"] == "validation"
            ]
            metrics[candidate][lane] = {
                "train_count": len(train),
                "validation_count": len(validation),
                "validation_new_normalized_distance_mean": float(
                    np.mean([row["new_normalized_center_distance"] for row in validation])
                ),
                "validation_old_normalized_distance_mean": float(
                    np.mean([row["old_normalized_center_distance"] for row in validation])
                ),
                "validation_distance_improvement_mean": float(
                    np.mean([row["distance_improvement"] for row in validation])
                ),
                "validation_out_of_range_rate": float(
                    np.mean([row["new_out_of_frequency_range"] for row in validation])
                ),
                "train_validation_pair_tv": total_variation(
                    [int(row["new_lower_band_one_based"]) for row in train],
                    [int(row["new_lower_band_one_based"]) for row in validation],
                    band_count,
                ),
                "train_pair_distribution": dict(
                    Counter(int(row["new_lower_band_one_based"]) for row in train)
                ),
                "validation_pair_distribution": dict(
                    Counter(int(row["new_lower_band_one_based"]) for row in validation)
                ),
            }
    return metrics


__all__ = [
    "CANDIDATES",
    "SlopeFit",
    "analyze_gap_rows",
    "bootstrap_mean_ci",
    "build_near_gamma_k_points",
    "candidate_values",
    "fit_gamma_slope",
    "nearest_gap",
    "total_variation",
]
