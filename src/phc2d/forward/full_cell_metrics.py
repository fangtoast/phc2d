"""Band-sliced metrics for Full-Wh surrogate evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BAND_GROUPS: Mapping[str, slice] = {
    "band1": slice(0, 1),
    "bands2_10": slice(1, 10),
    "bands1_10": slice(0, 10),
}


@dataclass(frozen=True)
class GapEdgeSelection:
    """One complete- or pseudo-gap upper-edge mode selected from a full BZ."""

    lower_band_one_based: int
    upper_band_one_based: int
    gap: float
    status: str
    k_index: int
    frequency: float


def select_largest_gap_edge(
    Wh: np.ndarray,
    *,
    lower_band_min_one_based: int = 2,
    lower_band_max_one_based: int | None = None,
) -> GapEdgeSelection:
    """Select the upper edge of the largest configured full-BZ adjacent-band gap."""

    bands = np.asarray(Wh, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[1] < 2 or bands.shape[0] < 1:
        raise ValueError(f"Wh must have shape [n_k_points, n_bands>=2], got {bands.shape}")
    if np.any(~np.isfinite(bands)):
        raise ValueError("Wh contains non-finite frequencies")
    lower_min = int(lower_band_min_one_based)
    lower_limit = bands.shape[1] - 1
    lower_max = (
        int(lower_band_max_one_based)
        if lower_band_max_one_based is not None
        else lower_limit
    )
    if lower_min < 1 or lower_max < lower_min or lower_max > lower_limit:
        raise ValueError(
            f"invalid lower-band range {lower_min}..{lower_max} for {bands.shape[1]} retained bands"
        )
    candidates: list[GapEdgeSelection] = []
    for lower_band in range(lower_min, lower_max + 1):
        lower_index = lower_band - 1
        upper_index = lower_index + 1
        lower_edge = float(np.max(bands[:, lower_index]))
        k_index = int(np.argmin(bands[:, upper_index]))
        upper_edge = float(bands[k_index, upper_index])
        gap = upper_edge - lower_edge
        candidates.append(
            GapEdgeSelection(
                lower_band_one_based=lower_band,
                upper_band_one_based=lower_band + 1,
                gap=gap,
                status="complete_gap" if gap > 0.0 else "pseudo_gap",
                k_index=k_index,
                frequency=upper_edge,
            )
        )
    return max(candidates, key=lambda row: (row.gap, -row.lower_band_one_based))


def compute_gap_decision_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    lower_band_min_one_based: int = 2,
    lower_band_max_one_based: int | None = None,
) -> dict[str, Any]:
    """Compare predicted and target largest-gap decisions on one frozen split."""

    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 3:
        raise ValueError(
            f"expected matching [sample, k, band] arrays, got {predicted.shape} and {truth.shape}"
        )
    if predicted.shape[0] < 1:
        raise ValueError("gap decision metrics require at least one sample")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("gap decision inputs must be finite")
    rows: list[dict[str, Any]] = []
    for predicted_sample, target_sample in zip(predicted, truth, strict=True):
        predicted_gap = select_largest_gap_edge(
            predicted_sample,
            lower_band_min_one_based=lower_band_min_one_based,
            lower_band_max_one_based=lower_band_max_one_based,
        )
        target_gap = select_largest_gap_edge(
            target_sample,
            lower_band_min_one_based=lower_band_min_one_based,
            lower_band_max_one_based=lower_band_max_one_based,
        )
        predicted_center = predicted_gap.frequency - 0.5 * predicted_gap.gap
        target_center = target_gap.frequency - 0.5 * target_gap.gap
        band_match = predicted_gap.lower_band_one_based == target_gap.lower_band_one_based
        k_match = predicted_gap.k_index == target_gap.k_index
        rows.append(
            {
                "target_status": target_gap.status,
                "band_match": band_match,
                "upper_k_match": k_match,
                "joint_band_k_match": band_match and k_match,
                "status_match": predicted_gap.status == target_gap.status,
                "selected_frequency_relative_error": abs(
                    predicted_gap.frequency - target_gap.frequency
                )
                / max(abs(target_gap.frequency), 1.0e-8),
                "gap_width_absolute_error": abs(predicted_gap.gap - target_gap.gap),
                "gap_center_absolute_error": abs(predicted_center - target_center),
            }
        )

    def aggregate(selected_rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        count = len(selected_rows)
        if count == 0:
            return {
                f"{prefix}sample_count": 0,
                f"{prefix}band_accuracy": None,
                f"{prefix}upper_edge_k_accuracy": None,
                f"{prefix}joint_band_k_accuracy": None,
                f"{prefix}status_accuracy": None,
                f"{prefix}selected_frequency_mre_pct": None,
                f"{prefix}gap_width_mae": None,
                f"{prefix}gap_center_mae": None,
            }
        return {
            f"{prefix}sample_count": count,
            f"{prefix}band_accuracy": float(np.mean([row["band_match"] for row in selected_rows])),
            f"{prefix}upper_edge_k_accuracy": float(
                np.mean([row["upper_k_match"] for row in selected_rows])
            ),
            f"{prefix}joint_band_k_accuracy": float(
                np.mean([row["joint_band_k_match"] for row in selected_rows])
            ),
            f"{prefix}status_accuracy": float(
                np.mean([row["status_match"] for row in selected_rows])
            ),
            f"{prefix}selected_frequency_mre_pct": 100.0
            * float(np.mean([row["selected_frequency_relative_error"] for row in selected_rows])),
            f"{prefix}gap_width_mae": float(
                np.mean([row["gap_width_absolute_error"] for row in selected_rows])
            ),
            f"{prefix}gap_center_mae": float(
                np.mean([row["gap_center_absolute_error"] for row in selected_rows])
            ),
        }

    complete = [row for row in rows if row["target_status"] == "complete_gap"]
    pseudo = [row for row in rows if row["target_status"] == "pseudo_gap"]
    return {
        **aggregate(rows, ""),
        "target_complete_gap_rate": len(complete) / len(rows),
        **aggregate(complete, "complete_gap_"),
        **aggregate(pseudo, "pseudo_gap_"),
    }


def _validate_arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 10:
        raise ValueError(f"expected matching [N,K,10] arrays, got {prediction.shape} and {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    return prediction, target


def _r2(values: np.ndarray, target: np.ndarray) -> tuple[float | None, str, int]:
    denominator = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    defined = denominator > 1.0e-12
    if not np.any(defined):
        return None, "undefined_zero_target_variance", int(defined.size)
    squared_error = np.sum((values - target) ** 2, axis=0)
    weights = np.where(defined, denominator, 0.0)
    score = 1.0 - float(np.sum(squared_error[defined])) / float(np.sum(weights))
    return score, "defined_with_zero_variance_coordinates_excluded" if not np.all(defined) else "defined", int((~defined).sum())


def _sample_metrics(prediction: np.ndarray, target: np.ndarray, bands: slice) -> dict[str, np.ndarray]:
    pred = prediction[:, :, bands]
    truth = target[:, :, bands]
    diff = pred - truth
    relative = np.abs(diff) / np.maximum(np.abs(truth), 1.0e-8)
    violation = np.zeros(pred.shape[0], dtype=np.float64)
    if pred.shape[2] > 1:
        violation = np.any(np.diff(pred, axis=2) < 0.0, axis=2).mean(axis=1)
    return {
        "mre": relative.mean(axis=(1, 2)) * 100.0,
        "mae": np.abs(diff).mean(axis=(1, 2)),
        "rmse": np.sqrt(np.mean(diff * diff, axis=(1, 2))),
        "band_order_violation": violation,
    }


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    if replicates <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=np.float64)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(sample), size=(count, len(sample)))
        means[start : start + count] = sample[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_band_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return three band-group metric sets plus per-band and per-k evidence.

    MRE values are percentages. Band-order violation is a per-sample rate; the
    single-band group is explicitly marked not applicable because it has no
    adjacent-band pair to compare.
    """

    prediction, target = _validate_arrays(prediction, target)
    metrics: dict[str, Any] = {"sample_count": int(target.shape[0]), "band_count": 10}
    sample_values: dict[str, np.ndarray] = {}
    for group_index, (group, bands) in enumerate(BAND_GROUPS.items()):
        values = _sample_metrics(prediction, target, bands)
        sample_values[group] = values["mre"]
        r2, r2_status, undefined = _r2(prediction[:, :, bands], target[:, :, bands])
        prefix = group
        metrics[f"{prefix}_mre"] = float(values["mre"].mean())
        metrics[f"{prefix}_mae"] = float(values["mae"].mean())
        metrics[f"{prefix}_rmse"] = float(values["rmse"].mean())
        metrics[f"{prefix}_r2_variance_weighted"] = r2
        metrics[f"{prefix}_r2_status"] = r2_status
        metrics[f"{prefix}_undefined_r2_coordinates"] = undefined
        if group == "band1":
            metrics[f"{prefix}_band_order_violation"] = 0.0
            metrics[f"{prefix}_band_order_violation_status"] = "not_applicable_single_band"
        else:
            metrics[f"{prefix}_band_order_violation"] = float(values["band_order_violation"].mean())
            metrics[f"{prefix}_band_order_violation_status"] = "defined"
        low, high = _bootstrap_ci(values["mre"], seed=bootstrap_seed + group_index, replicates=bootstrap_replicates)
        metrics[f"{prefix}_mre_bootstrap_ci95_low"] = low
        metrics[f"{prefix}_mre_bootstrap_ci95_high"] = high
        for offset, name in enumerate(("mae", "rmse", "band_order_violation"), start=1):
            low, high = _bootstrap_ci(values[name], seed=bootstrap_seed + group_index * 10 + offset, replicates=bootstrap_replicates)
            metrics[f"{prefix}_{name}_bootstrap_ci95_low"] = low
            metrics[f"{prefix}_{name}_bootstrap_ci95_high"] = high
        metrics[f"{prefix}_bootstrap_replicates"] = int(bootstrap_replicates)

    per_band: list[dict[str, Any]] = []
    for band in range(target.shape[2]):
        values = _sample_metrics(prediction, target, slice(band, band + 1))
        r2, r2_status, undefined = _r2(prediction[:, :, band : band + 1], target[:, :, band : band + 1])
        per_band.append({
            "band_one_based": band + 1,
            "mre": float(values["mre"].mean()),
            "mae": float(values["mae"].mean()),
            "rmse": float(values["rmse"].mean()),
            "r2_variance_weighted": r2,
            "r2_status": r2_status,
            "undefined_r2_coordinates": undefined,
            "band_order_violation": 0.0,
            "band_order_violation_status": "not_applicable_single_band",
        })

    per_k: list[dict[str, Any]] = []
    for group, bands in BAND_GROUPS.items():
        for k_index in range(target.shape[1]):
            values = _sample_metrics(prediction[:, k_index : k_index + 1], target[:, k_index : k_index + 1], bands)
            r2, r2_status, undefined = _r2(prediction[:, k_index : k_index + 1, bands], target[:, k_index : k_index + 1, bands])
            per_k.append({
                "group": group,
                "k_index": k_index,
                "mre": float(values["mre"].mean()),
                "mae": float(values["mae"].mean()),
                "rmse": float(values["rmse"].mean()),
                "r2_variance_weighted": r2,
                "r2_status": r2_status,
                "undefined_r2_coordinates": undefined,
                "band_order_violation": 0.0 if group == "band1" else float(values["band_order_violation"].mean()),
            })
    return metrics, sample_values, per_band, per_k


def paired_bootstrap_mre(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Compute deterministic paired bootstrap deltas for the three band groups."""

    result: dict[str, Any] = {"bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed)}
    for group_index, group in enumerate(BAND_GROUPS):
        delta = np.asarray(left[group], dtype=np.float64) - np.asarray(right[group], dtype=np.float64)
        low, high = _bootstrap_ci(delta, seed=seed + group_index, replicates=replicates)
        result[f"{group}_mre_delta"] = float(delta.mean())
        result[f"{group}_mre_delta_ci95_low"] = low
        result[f"{group}_mre_delta_ci95_high"] = high
    return result


__all__ = [
    "BAND_GROUPS",
    "GapEdgeSelection",
    "compute_band_metrics",
    "compute_gap_decision_metrics",
    "paired_bootstrap_mre",
    "select_largest_gap_edge",
]
