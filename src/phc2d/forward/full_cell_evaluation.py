"""Shared validation engine for normalized Full-Wh surrogate outputs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from phc2d.forward.full_cell_metrics import compute_gap_decision_metrics


def evaluate_full_cell(
    model: Any,
    loader: Any,
    *,
    device: Any,
    mean: Any,
    std: Any,
    model_spec: Any,
    k_points: Any,
    lane_index: int | None,
    torch_module: Any,
    forward_model: Callable[..., Any],
    gap_lower_band_min_one_based: int | None = None,
    gap_lower_band_max_one_based: int | None = None,
) -> dict[str, Any]:
    """Evaluate full-spectrum errors and optional largest-gap decisions."""

    model.eval()
    squared = 0.0
    absolute = 0.0
    relative = 0.0
    relative_band1 = 0.0
    relative_all = 0.0
    normalized_squared = 0.0
    count = 0
    gap_predictions: list[np.ndarray] = []
    gap_targets: list[np.ndarray] = []
    inference_start = time.perf_counter()
    with torch_module.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction_norm = forward_model(
                model,
                xb,
                model_spec=model_spec,
                k_points=k_points,
                lane_index=lane_index,
                torch=torch_module,
            )
            if not bool(torch_module.isfinite(prediction_norm).all().item()):
                raise RuntimeError("non-finite validation prediction")
            normalized_diff = prediction_norm - yb
            normalized_squared += float(
                torch_module.sum(normalized_diff * normalized_diff).detach().cpu()
            )
            prediction = prediction_norm * std + mean
            target = yb * std + mean
            if gap_lower_band_min_one_based is not None:
                gap_predictions.append(prediction.detach().cpu().numpy())
                gap_targets.append(target.detach().cpu().numpy())
            diff = prediction - target
            squared += float(torch_module.sum(diff * diff).detach().cpu())
            absolute += float(torch_module.sum(torch_module.abs(diff)).detach().cpu())
            relative += float(
                torch_module.sum(
                    torch_module.abs(diff[..., 1:])
                    / torch_module.clamp(torch_module.abs(target[..., 1:]), min=1.0e-8)
                )
                .detach()
                .cpu()
            )
            relative_band1 += float(
                torch_module.sum(
                    torch_module.abs(diff[..., :1])
                    / torch_module.clamp(torch_module.abs(target[..., :1]), min=1.0e-8)
                )
                .detach()
                .cpu()
            )
            relative_all += float(
                torch_module.sum(
                    torch_module.abs(diff)
                    / torch_module.clamp(torch_module.abs(target), min=1.0e-8)
                )
                .detach()
                .cpu()
            )
            count += int(target.numel())
    elapsed = time.perf_counter() - inference_start
    band_count = count * max(int(mean.shape[-1]) - 1, 0) / max(int(mean.shape[-1]), 1)
    sample_count = max(count // int(mean.shape[-1]), 1)
    metrics: dict[str, Any] = {
        "mse": squared / max(count, 1),
        "normalized_mse": normalized_squared / max(count, 1),
        "mae": absolute / max(count, 1),
        "rmse": float(np.sqrt(squared / max(count, 1))),
        "bands_2_10_mre_pct": 100.0 * relative / max(band_count, 1),
        "band1_mre_pct": 100.0 * relative_band1 / sample_count,
        "bands_1_10_mre_pct": 100.0 * relative_all / max(count, 1),
        "eval_time_s": elapsed,
    }
    if gap_lower_band_min_one_based is not None:
        metrics.update(
            compute_gap_decision_metrics(
                np.concatenate(gap_predictions, axis=0),
                np.concatenate(gap_targets, axis=0),
                lower_band_min_one_based=gap_lower_band_min_one_based,
                lower_band_max_one_based=gap_lower_band_max_one_based,
            )
        )
    return metrics


__all__ = ["evaluate_full_cell"]
