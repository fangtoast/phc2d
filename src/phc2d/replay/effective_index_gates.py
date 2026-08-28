"""Preregistered evidence aggregation and gates for EXP-PHY-005.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from phc2d.physics.effective_index_validation import (
    CANDIDATES,
    analyze_gap_rows,
    bootstrap_mean_ci,
)


@dataclass(frozen=True)
class EffectiveIndexAuditDecision:
    """Aggregated evidence and terminal decision for one audit execution."""

    status: str
    slope_metrics: dict[str, Any]
    gap_payload: dict[str, Any]
    numeric_pass: bool
    anisotropy_pass: bool
    estimator_pass: bool
    gap_pass: bool
    candidate_gates: dict[str, Any]


def _oracle_metrics(
    rows: Sequence[Mapping[str, Any]], lanes: Sequence[str]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        metrics[candidate] = {}
        for lane in lanes:
            selected = [
                row
                for row in rows
                if row["candidate"] == candidate
                and row["lane"] == lane
                and row["split"] == "validation"
            ]
            complete = [row for row in selected if row["oracle_status"] == "complete_gap"]
            metrics[candidate][lane] = {
                "sample_count": len(selected),
                "pair_agreement": float(np.mean([row["pair_match"] for row in selected])),
                "complete_sample_count": len(complete),
                "complete_pair_agreement": (
                    float(np.mean([row["pair_match"] for row in complete])) if complete else None
                ),
                "oracle_normalized_distance_median": float(
                    np.median([row["oracle_normalized_center_distance"] for row in selected])
                ),
                "oracle_out_of_range_rate": float(
                    np.mean([row["oracle_out_of_range"] for row in selected])
                ),
            }
    return metrics


def evaluate_effective_index_gates(
    *,
    mode: str,
    sentinel_rows: Sequence[Mapping[str, Any]],
    slope_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    spacing_floor: float,
    gates: Mapping[str, Any],
    lanes: Sequence[str],
    band_count: int,
) -> EffectiveIndexAuditDecision:
    """Apply every preregistered numerical, estimator, anisotropy, and gap gate."""

    gap_metrics = analyze_gap_rows(gap_rows, lanes=lanes, band_count=band_count)
    oracle_metrics = _oracle_metrics(oracle_rows, lanes)
    slope_metrics: dict[str, Any] = {"sentinels": list(sentinel_rows), "lanes": {}}
    for lane in lanes:
        selected = [row for row in slope_rows if row["lane"] == lane]
        slope_metrics["lanes"][lane] = {
            "sample_count": len(selected),
            "x_r2_pass_rate": float(
                np.mean([row["x_r2"] >= float(gates["slope_r2_min"]) for row in selected])
            ),
            "m_r2_pass_rate": float(
                np.mean([row["m_r2"] >= float(gates["slope_r2_min"]) for row in selected])
            ),
            "window_drift_pass_rate": float(
                np.mean(
                    [
                        max(row["x_window_drift"], row["m_window_drift"])
                        <= float(gates["slope_window_drift_max"])
                        for row in selected
                    ]
                )
            ),
            "old_error_median": float(np.median([row["old_relative_error"] for row in selected])),
            "te_error_median": float(np.median([row["te_relative_error"] for row in selected])),
            "old_error_p90": float(np.quantile([row["old_relative_error"] for row in selected], 0.9)),
            "te_error_p90": float(np.quantile([row["te_relative_error"] for row in selected], 0.9)),
            "anisotropy_median": float(np.median([row["anisotropy"] for row in selected])),
            "anisotropy_p90": float(np.quantile([row["anisotropy"] for row in selected], 0.9)),
        }
    estimator_improvement = [
        row["old_relative_error"] - row["te_relative_error"] for row in slope_rows
    ]
    estimator_ci = bootstrap_mean_ci(estimator_improvement, seed=5005)
    slope_metrics["pooled_estimator_improvement_ci95"] = list(estimator_ci)
    numeric_pass = all(
        metrics[direction] >= float(gates["slope_r2_pass_rate_min"])
        for metrics in slope_metrics["lanes"].values()
        for direction in ("x_r2_pass_rate", "m_r2_pass_rate")
    ) and all(
        metrics["window_drift_pass_rate"]
        >= float(gates["slope_window_drift_pass_rate_min"])
        for metrics in slope_metrics["lanes"].values()
    )
    anisotropy_pass = all(
        metrics["anisotropy_median"] <= float(gates["anisotropy_median_max"])
        and metrics["anisotropy_p90"] <= float(gates["anisotropy_p90_max"])
        for metrics in slope_metrics["lanes"].values()
    )
    estimator_pass = estimator_ci[0] > 0.0 and all(
        metrics["te_error_median"] < metrics["old_error_median"]
        and metrics["te_error_p90"] <= metrics["old_error_p90"]
        for metrics in slope_metrics["lanes"].values()
    )
    candidate_gates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        improvements = [
            row["distance_improvement"]
            for row in gap_rows
            if row["candidate"] == candidate and row["split"] == "validation"
        ]
        improvement_ci = bootstrap_mean_ci(
            improvements, seed=5100 + CANDIDATES.index(candidate)
        )
        lane_pass = all(
            oracle_metrics[candidate][lane]["pair_agreement"]
            >= float(gates["oracle_pair_agreement_min"])
            and gap_metrics[candidate][lane]["validation_distance_improvement_mean"] > 0.0
            and gap_metrics[candidate][lane]["validation_out_of_range_rate"]
            <= float(gates["candidate_out_of_range_rate_max"])
            and gap_metrics[candidate][lane]["train_validation_pair_tv"]
            <= float(gates["train_validation_tv_max"])
            and (
                oracle_metrics[candidate][lane]["complete_sample_count"]
                < int(gates["complete_subset_min"])
                or oracle_metrics[candidate][lane]["complete_pair_agreement"]
                >= float(gates["complete_pair_agreement_min"])
            )
            for lane in lanes
        )
        candidate_gates[candidate] = {
            "pass": bool(lane_pass and improvement_ci[0] > 0.0),
            "pooled_distance_improvement_ci95": list(improvement_ci),
        }
    gap_pass = any(value["pass"] for value in candidate_gates.values())
    oracle_viable = any(
        all(
            oracle_metrics[candidate][lane]["oracle_normalized_distance_median"] <= 1.0
            and oracle_metrics[candidate][lane]["oracle_out_of_range_rate"]
            <= float(gates["candidate_out_of_range_rate_max"])
            for lane in lanes
        )
        for candidate in CANDIDATES
    )
    if mode == "engineering_smoke":
        status = "smoke_pass"
    elif not numeric_pass:
        status = "blocked"
    elif not anisotropy_pass:
        status = "scalar_anisotropy_fail"
    elif estimator_pass and gap_pass:
        status = "prior_physical_gate_pass"
    elif oracle_viable:
        status = "effective_medium_fail"
    else:
        status = "folding_hypothesis_fail"
    return EffectiveIndexAuditDecision(
        status=status,
        slope_metrics=slope_metrics,
        gap_payload={
            "spacing_floor": spacing_floor,
            "analytic": gap_metrics,
            "oracle": oracle_metrics,
            "candidate_gates": candidate_gates,
            "oracle_viable": oracle_viable,
        },
        numeric_pass=numeric_pass,
        anisotropy_pass=anisotropy_pass,
        estimator_pass=estimator_pass,
        gap_pass=gap_pass,
        candidate_gates=candidate_gates,
    )


__all__ = ["EffectiveIndexAuditDecision", "evaluate_effective_index_gates"]
