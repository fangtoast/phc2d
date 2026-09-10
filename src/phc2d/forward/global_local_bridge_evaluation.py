"""Pure metric and aggregation helpers for EXP-FWD-020 G3 evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


METRIC_GROUPS: dict[str, tuple[int, ...]] = {
    "band1": (0,),
    "bands_2_10": tuple(range(1, 10)),
    "bands_1_10": tuple(range(10)),
}


def _relative_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(prediction - target) / np.maximum(np.abs(target), 1.0e-8)


def _r2_summary(prediction: np.ndarray, target: np.ndarray) -> tuple[float, str, int]:
    """Return variance-weighted R2 with an explicit zero-variance state."""
    diff = prediction - target
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = (centered * centered).sum(axis=0)
    numerator = (diff * diff).sum(axis=0)
    valid = denominator > 1.0e-12
    zero_count = int((~valid).sum())
    if not np.any(valid):
        return 0.0, "undefined_all_zero_variance", zero_count
    r2 = 1.0 - numerator[valid] / denominator[valid]
    value = float(np.sum(r2 * denominator[valid]) / denominator[valid].sum())
    status = "defined" if zero_count == 0 else "partial_zero_variance"
    return value, status, zero_count


def _group_metrics(prediction: np.ndarray, target: np.ndarray, bands: tuple[int, ...]) -> dict[str, Any]:
    subset_prediction = prediction[:, :, bands]
    subset_target = target[:, :, bands]
    difference = subset_prediction - subset_target
    relative = _relative_error(subset_prediction, subset_target)
    r2, r2_status, zero_variance_count = _r2_summary(subset_prediction, subset_target)
    return {
        "mre_pct": float(relative.mean() * 100.0),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt((difference * difference).mean())),
        "r2_variance_weighted": r2,
        "r2_status": r2_status,
        "zero_variance_coordinate_count": zero_variance_count,
        "band_order_violation_rate": float(np.mean(np.any(np.diff(subset_prediction, axis=2) < 0.0, axis=2))),
    }


def gap_decisions(values: np.ndarray) -> dict[str, np.ndarray]:
    """Choose each sample's maximum adjacent B2--B10 gap from a Wh tensor.

    `complete_gap` is intentionally a label-level decision: it means the
    selected global adjacent-band gap is positive on the frozen 81-point grid.
    It is not a continuous-BZ or real-MPB certification.
    """
    if values.ndim != 3 or values.shape[2] < 10:
        raise ValueError("gap decisions require values shaped [N, K, >=10]")
    candidates: list[dict[str, np.ndarray]] = []
    for lower_band in range(1, 9):  # zero-based B2 through B9
        lower = values[:, :, lower_band]
        upper = values[:, :, lower_band + 1]
        lower_k = lower.argmax(axis=1)
        upper_k = upper.argmin(axis=1)
        lower_edge = lower[np.arange(len(values)), lower_k]
        upper_edge = upper[np.arange(len(values)), upper_k]
        candidates.append(
            {
                "lower_band": np.full(len(values), lower_band + 1, dtype=np.int16),
                "upper_band": np.full(len(values), lower_band + 2, dtype=np.int16),
                "lower_k": lower_k.astype(np.int16),
                "upper_k": upper_k.astype(np.int16),
                "lower_edge": lower_edge.astype(np.float64),
                "upper_edge": upper_edge.astype(np.float64),
                "gap_width": (upper_edge - lower_edge).astype(np.float64),
            }
        )
    widths = np.stack([item["gap_width"] for item in candidates], axis=1)
    selected = widths.argmax(axis=1)
    output = {key: np.stack([item[key] for item in candidates], axis=1)[np.arange(len(values)), selected] for key in candidates[0]}
    output["gap_center"] = (output["lower_edge"] + output["upper_edge"]) / 2.0
    output["complete_gap"] = output["gap_width"] > 0.0
    return output


def gap_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute frozen-label gap-decision agreement and retain sample decisions."""
    predicted = gap_decisions(prediction)
    truth = gap_decisions(target)
    selected_band_match = predicted["lower_band"] == truth["lower_band"]
    upper_k_match = predicted["upper_k"] == truth["upper_k"]
    status_match = predicted["complete_gap"] == truth["complete_gap"]
    target_complete = truth["complete_gap"]
    predicted_complete = predicted["complete_gap"]
    selected_frequency_relative = np.abs(predicted["gap_center"] - truth["gap_center"]) / np.maximum(np.abs(truth["gap_center"]), 1.0e-8)
    metrics: dict[str, Any] = {
        "selected_band_accuracy": float(selected_band_match.mean()),
        "upper_edge_k_accuracy": float(upper_k_match.mean()),
        "band_k_joint_accuracy": float((selected_band_match & upper_k_match).mean()),
        "complete_pseudo_status_accuracy": float(status_match.mean()),
        "target_complete_gap_rate": float(target_complete.mean()),
        "predicted_complete_gap_rate": float(predicted_complete.mean()),
        "complete_gap_recall": float(predicted_complete[target_complete].mean()) if np.any(target_complete) else 0.0,
        "complete_gap_recall_status": "defined" if np.any(target_complete) else "undefined_no_target_complete_gap",
        "selected_frequency_mre_pct": float(selected_frequency_relative.mean() * 100.0),
        "gap_width_mae": float(np.abs(predicted["gap_width"] - truth["gap_width"]).mean()),
        "gap_center_mae": float(np.abs(predicted["gap_center"] - truth["gap_center"]).mean()),
    }
    for label, mask in (("complete", target_complete), ("pseudo", ~target_complete)):
        metrics[f"{label}_subset_count"] = int(mask.sum())
        metrics[f"{label}_metrics_status"] = "defined" if np.any(mask) else f"undefined_no_target_{label}_gap"
        metrics[f"{label}_selected_band_accuracy"] = float(selected_band_match[mask].mean()) if np.any(mask) else 0.0
        metrics[f"{label}_gap_width_mae"] = float(np.abs(predicted["gap_width"][mask] - truth["gap_width"][mask]).mean()) if np.any(mask) else 0.0
    arrays = {
        "sample_mre_band1_pct": _relative_error(prediction[:, :, :1], target[:, :, :1]).mean(axis=(1, 2)) * 100.0,
        "sample_mre_bands_2_10_pct": _relative_error(prediction[:, :, 1:10], target[:, :, 1:10]).mean(axis=(1, 2)) * 100.0,
        "sample_mre_bands_1_10_pct": _relative_error(prediction[:, :, :10], target[:, :, :10]).mean(axis=(1, 2)) * 100.0,
        **{f"predicted_{key}": value for key, value in predicted.items()},
        **{f"target_{key}": value for key, value in truth.items()},
    }
    return metrics, arrays


def cell_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Return all G3 cell metrics without importing solver or training stacks."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 10:
        raise ValueError("G3 prediction and target must both be shaped [N, K, 10]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("nonfinite prediction or target")
    groups = {name: _group_metrics(prediction, target, bands) for name, bands in METRIC_GROUPS.items()}
    gap, arrays = gap_metrics(prediction, target)
    metrics = {
        **{f"{name}_{field}": value for name, values in groups.items() for field, value in values.items()},
        # Compatibility aliases used by the pre-existing controller and campaign checks.
        "band1_mre_pct": groups["band1"]["mre_pct"],
        "bands_2_10_mre_pct": groups["bands_2_10"]["mre_pct"],
        "bands_1_10_mre_pct": groups["bands_1_10"]["mre_pct"],
        **gap,
    }
    per_band = []
    for band in range(10):
        values = _group_metrics(prediction[:, :, band : band + 1], target[:, :, band : band + 1], (0,))
        per_band.append({"band_one_based": band + 1, **values})
    per_k = []
    for k_index in range(prediction.shape[1]):
        values = _group_metrics(prediction[:, k_index : k_index + 1, :], target[:, k_index : k_index + 1, :], tuple(range(10)))
        per_k.append({"k_index": k_index, **values})
    return metrics, per_band, per_k, arrays


def paired_bootstrap(candidate: np.ndarray, baseline: np.ndarray, *, key: str, replicates: int = 10_000) -> dict[str, Any]:
    """Bootstrap a same-target candidate-minus-baseline sample MRE comparison."""
    if candidate.shape != baseline.shape or candidate.ndim != 1 or len(candidate) == 0:
        raise ValueError("paired bootstrap requires equally sized non-empty vectors")
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16))
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        index = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start : start + size] = delta[index].mean(axis=1)
    low, high = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    direction = "candidate_improves" if high < 0.0 else "candidate_regresses" if low > 0.0 else "uncertain"
    return {
        "baseline_bands_2_10_mre_pct": float(baseline.mean()),
        "candidate_bands_2_10_mre_pct": float(candidate.mean()),
        "mre_delta_pp": float(delta.mean()),
        "relative_improvement_pct": float(-delta.mean() / max(float(baseline.mean()), 1.0e-12) * 100.0),
        "ci95_low_pp": low,
        "ci95_high_pp": high,
        "conclusion": direction,
        "bootstrap_replicates": replicates,
    }


def mean_std(rows: Iterable[Mapping[str, str]], metric: str = "bands_2_10_mre_pct") -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))
