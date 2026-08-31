"""Validation metrics for bounded theory-frequency weighting experiments.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phc2d.forward.frequency_weighted_loss import FrequencyWeightedLossConfig


def evaluate_frequency_weighted_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    frequency_center: np.ndarray,
    target_std: np.ndarray,
    config: FrequencyWeightedLossConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return window, spectrum, per-band, and per-k validation diagnostics."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    center = np.asarray(frequency_center, dtype=np.float64)
    std = np.asarray(target_std, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("frequency metrics require matching [sample, k, band] arrays")
    if center.shape != (prediction.shape[0],) or std.shape != prediction.shape[1:]:
        raise ValueError("frequency metrics received incompatible centers or normalizer shape")
    if not all(np.isfinite(value).all() for value in (prediction, target, center, std)):
        raise ValueError("frequency metrics require finite inputs")
    diff = prediction - target
    normalized_abs = np.abs(diff) / np.maximum(np.abs(std)[None, :, :], 1.0e-8)
    relative = np.abs(diff) / np.maximum(np.abs(target), 1.0e-8)
    raw = config.epsilon_f / (np.abs(target - center[:, None, None]) + config.epsilon_f)
    weights = (1.0 + config.rho * raw)
    weights /= weights.mean(axis=(1, 2), keepdims=True)
    windows: dict[str, Any] = {}
    for fraction in (0.05, 0.10, 0.15):
        mask = np.abs(target - center[:, None, None]) <= fraction * center[:, None, None]
        count = int(mask.sum())
        key = f"pm_{int(fraction * 100)}pct"
        if count == 0:
            windows[key] = {"coordinate_count": 0, "nmae": None, "mre_pct": None, "mae": None, "band_distribution": {}}
            continue
        bands = np.where(mask)[2]
        b12 = mask[:, :, : min(2, target.shape[2])]
        windows[key] = {
            "coordinate_count": count,
            "nmae": float(normalized_abs[mask].mean()),
            "mre_pct": float(100.0 * relative[mask].mean()),
            "mae": float(np.abs(diff)[mask].mean()),
            "b1_b2_conditional_nmae": float(normalized_abs[:, :, : min(2, target.shape[2])][b12].mean()) if bool(b12.any()) else None,
            "band_distribution": {str(index + 1): int((bands == index).sum()) for index in range(target.shape[2])},
        }
    groups = {"band1": slice(0, 1), "bands_2_10": slice(1, None), "bands_1_10": slice(0, None)}
    spectrum = {
        name: {
            "mre_pct": float(100.0 * relative[:, :, band].mean()),
            "mae": float(np.abs(diff[:, :, band]).mean()),
            "rmse": float(np.sqrt(np.mean(diff[:, :, band] ** 2))),
        }
        for name, band in groups.items()
    }
    per_band = [
        {"band_one_based": band + 1, "mre_pct": float(100.0 * relative[:, :, band].mean()),
         "mae": float(np.abs(diff[:, :, band]).mean()), "nmae": float(normalized_abs[:, :, band].mean())}
        for band in range(target.shape[2])
    ]
    per_k = [
        {"k_index": k, "mre_pct": float(100.0 * relative[:, k, :].mean()),
         "mae": float(np.abs(diff[:, k, :]).mean()), "nmae": float(normalized_abs[:, k, :].mean())}
        for k in range(target.shape[1])
    ]
    pair_rows: list[dict[str, Any]] = []
    for lower in range(target.shape[2] - 1):
        target_lower = target[:, :, lower].max(axis=1)
        target_upper = target[:, :, lower + 1].min(axis=1)
        prediction_lower = prediction[:, :, lower].max(axis=1)
        prediction_upper = prediction[:, :, lower + 1].min(axis=1)
        target_width = target_upper - target_lower
        prediction_width = prediction_upper - prediction_lower
        pair_rows.append({
            "lower_band_one_based": lower + 1,
            "center_mae": float(np.abs((prediction_upper + prediction_lower - target_upper - target_lower) / 2.0).mean()),
            "signed_width_mae": float(np.abs(prediction_width - target_width).mean()),
            "lower_edge_mae": float(np.abs(prediction_lower - target_lower).mean()),
            "upper_edge_mae": float(np.abs(prediction_upper - target_upper).mean()),
            "target_complete_rate": float(np.mean(target_width > 0.0)),
            "target_pseudo_rate": float(np.mean(target_width <= 0.0)),
        })
    nearest_target = np.argmin(np.abs((target[:, :, :-1].max(axis=1) + target[:, :, 1:].min(axis=1)) / 2.0 - center[:, None]), axis=1)
    nearest_prediction = np.argmin(np.abs((prediction[:, :, :-1].max(axis=1) + prediction[:, :, 1:].min(axis=1)) / 2.0 - center[:, None]), axis=1)
    result = {
        "definition": {
            "center": config.center_kind,
            "window_coordinate_rule": "abs(target_frequency-f0)<=fraction*f0",
            "nmae": "absolute physical error divided by lane train-only target std at each coordinate",
            "bounded_inverse_weighted_mse": "weighted normalized-coordinate squared error",
        },
        "windows": windows,
        "spectrum": spectrum,
        "bounded_inverse_weighted_mse": float(np.mean(weights * (diff / np.maximum(std[None, :, :], 1.0e-8)) ** 2)),
        "weight_statistics": {"mean": float(weights.mean()), "min": float(weights.min()), "max": float(weights.max()), "finite": bool(np.isfinite(weights).all())},
        "nearest_pair_consistency_rate": float(np.mean(nearest_prediction == nearest_target)),
        "sample_count": int(target.shape[0]),
    }
    return result, per_band, per_k, pair_rows


__all__ = ["evaluate_frequency_weighted_predictions"]
