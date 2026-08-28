"""Differentiable gap-aware objectives for Full-Wh surrogate training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BASE_LOSS_NAME = "equal_lane_normalized_full_matrix_mse"
GAP_AWARE_LOSS_NAME = "equal_lane_gap_aware_v1"


@dataclass(frozen=True)
class GapAwareLossConfig:
    """Validated weights and band range for one gap-aware loss contract."""

    lower_band_min_one_based: int
    lower_band_max_one_based: int | None
    full_spectrum_weight: float
    gap_profile_weight: float
    selected_edge_weight: float
    selected_width_weight: float
    selected_center_weight: float
    selected_upper_profile_weight: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable contract for run evidence."""

        return {
            "name": GAP_AWARE_LOSS_NAME,
            "lower_band_min_one_based": self.lower_band_min_one_based,
            "lower_band_max_one_based": self.lower_band_max_one_based,
            "full_spectrum_weight": self.full_spectrum_weight,
            "gap_profile_weight": self.gap_profile_weight,
            "selected_edge_weight": self.selected_edge_weight,
            "selected_width_weight": self.selected_width_weight,
            "selected_center_weight": self.selected_center_weight,
            "selected_upper_profile_weight": self.selected_upper_profile_weight,
            "frequency_scale": "train_normalizer_mean_std",
            "selection_source": "train_target_largest_gap_complete_or_pseudo",
        }


@dataclass(frozen=True)
class GapAwareLossTerms:
    """Total objective and its unweighted, dimensionless components."""

    total: Any
    full_spectrum: Any
    gap_profile: Any
    selected_edge: Any
    selected_width: Any
    selected_center: Any
    selected_upper_profile: Any

    def as_dict(self) -> dict[str, Any]:
        """Expose stable component names for training history."""

        return {
            "total": self.total,
            "full_spectrum": self.full_spectrum,
            "gap_profile": self.gap_profile,
            "selected_edge": self.selected_edge,
            "selected_width": self.selected_width,
            "selected_center": self.selected_center,
            "selected_upper_profile": self.selected_upper_profile,
        }


def resolve_gap_aware_loss_config(raw: Any) -> GapAwareLossConfig | None:
    """Resolve a legacy base loss or the explicit gap-aware mapping."""

    if raw is None or raw == BASE_LOSS_NAME:
        return None
    if isinstance(raw, str):
        raise ValueError(
            f"unsupported dual-lane loss {raw!r}; {GAP_AWARE_LOSS_NAME} requires a mapping"
        )
    if not isinstance(raw, Mapping):
        raise TypeError("training.loss must be a string or mapping")
    name = str(raw.get("name", ""))
    if name == BASE_LOSS_NAME:
        extra = set(raw) - {"name"}
        if extra:
            raise ValueError(f"base dual-lane loss does not accept parameters: {sorted(extra)}")
        return None
    if name != GAP_AWARE_LOSS_NAME:
        raise ValueError(f"unsupported dual-lane loss {name!r}")
    allowed = {
        "name",
        "lower_band_min_one_based",
        "lower_band_max_one_based",
        "full_spectrum_weight",
        "gap_profile_weight",
        "selected_edge_weight",
        "selected_width_weight",
        "selected_center_weight",
        "selected_upper_profile_weight",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown gap-aware loss fields: {sorted(unknown)}")
    lower_min = int(raw.get("lower_band_min_one_based", 2))
    lower_max_raw = raw.get("lower_band_max_one_based")
    lower_max = int(lower_max_raw) if lower_max_raw is not None else None
    if lower_min < 1 or (lower_max is not None and lower_max < lower_min):
        raise ValueError("gap-aware lower-band range is invalid")
    weights = {
        "full_spectrum_weight": float(raw.get("full_spectrum_weight", 1.0)),
        "gap_profile_weight": float(raw.get("gap_profile_weight", 0.0)),
        "selected_edge_weight": float(raw.get("selected_edge_weight", 0.0)),
        "selected_width_weight": float(raw.get("selected_width_weight", 0.0)),
        "selected_center_weight": float(raw.get("selected_center_weight", 0.0)),
        "selected_upper_profile_weight": float(
            raw.get("selected_upper_profile_weight", 0.0)
        ),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("gap-aware loss weights must be non-negative")
    if weights["full_spectrum_weight"] <= 0.0:
        raise ValueError("gap-aware loss must retain positive full-spectrum supervision")
    if not any(value > 0.0 for key, value in weights.items() if key != "full_spectrum_weight"):
        raise ValueError("gap-aware loss requires at least one positive auxiliary weight")
    return GapAwareLossConfig(
        lower_band_min_one_based=lower_min,
        lower_band_max_one_based=lower_max,
        **weights,
    )


def _gap_edges(values: Any, config: GapAwareLossConfig, *, torch_module: Any) -> tuple[Any, Any]:
    """Return lower maxima and upper minima for every configured adjacent pair."""

    band_count = int(values.shape[-1])
    lower_limit = band_count - 1
    lower_max = config.lower_band_max_one_based or lower_limit
    if lower_max > lower_limit:
        raise ValueError(
            f"gap-aware lower-band maximum {lower_max} exceeds {lower_limit} for {band_count} bands"
        )
    indices = torch_module.arange(
        config.lower_band_min_one_based - 1,
        lower_max,
        device=values.device,
    )
    lower_edges = values.index_select(2, indices).amax(dim=1)
    upper_edges = values.index_select(2, indices + 1).amin(dim=1)
    return lower_edges, upper_edges


def gap_aware_spectrum_loss(
    prediction: Any,
    target: Any,
    *,
    target_mean: Any,
    target_std: Any,
    config: GapAwareLossConfig,
    torch_module: Any | None = None,
) -> GapAwareLossTerms:
    """Combine normalized spectrum MSE with physical-frequency gap errors.

    ``prediction`` and ``target`` are normalized ``[batch, k, band]`` tensors.
    Gap features are computed after de-normalization with train-only statistics.
    The target chooses the largest complete or pseudo gap; no test or inference
    target is needed by the model.
    """

    if torch_module is None:
        import torch as torch_module

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "gap-aware loss expects matching normalized [batch, k, band] tensors, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    mean = torch_module.as_tensor(
        target_mean,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    std = torch_module.as_tensor(
        target_std,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    if tuple(mean.shape) != tuple(prediction.shape[1:]) or tuple(std.shape) != tuple(
        prediction.shape[1:]
    ):
        raise ValueError("target normalizers must match the [k, band] target shape")
    scale = std.detach().abs().mean().clamp_min(1.0e-8)
    physical_prediction = prediction * std + mean
    physical_target = target * std + mean
    pred_lower, pred_upper = _gap_edges(
        physical_prediction,
        config,
        torch_module=torch_module,
    )
    true_lower, true_upper = _gap_edges(physical_target, config, torch_module=torch_module)
    pred_gaps = pred_upper - pred_lower
    true_gaps = true_upper - true_lower
    selected = true_gaps.argmax(dim=1, keepdim=True)

    def selected_value(values: Any) -> Any:
        return values.gather(1, selected).squeeze(1)

    pred_selected_lower = selected_value(pred_lower)
    pred_selected_upper = selected_value(pred_upper)
    true_selected_lower = selected_value(true_lower)
    true_selected_upper = selected_value(true_upper)
    pred_selected_width = pred_selected_upper - pred_selected_lower
    true_selected_width = true_selected_upper - true_selected_lower
    pred_selected_center = 0.5 * (pred_selected_upper + pred_selected_lower)
    true_selected_center = 0.5 * (true_selected_upper + true_selected_lower)
    lower_indices = torch_module.arange(
        config.lower_band_min_one_based - 1,
        (config.lower_band_max_one_based or prediction.shape[-1] - 1),
        device=prediction.device,
    )
    selected_upper_bands = (lower_indices[selected.squeeze(1)] + 1).to(
        dtype=torch_module.long
    )
    batch_indices = torch_module.arange(prediction.shape[0], device=prediction.device)
    pred_upper_profile = physical_prediction[batch_indices, :, selected_upper_bands]
    true_upper_profile = physical_target[batch_indices, :, selected_upper_bands]
    true_upper_k = true_upper_profile.argmin(dim=1, keepdim=True)
    pred_relative_profile = pred_upper_profile - pred_upper_profile.gather(1, true_upper_k)
    true_relative_profile = true_upper_profile - true_upper_profile.gather(1, true_upper_k)

    full_spectrum = torch_module.nn.functional.mse_loss(prediction, target)
    gap_profile = torch_module.mean(((pred_gaps - true_gaps) / scale) ** 2)
    selected_edge = 0.5 * torch_module.mean(
        ((pred_selected_lower - true_selected_lower) / scale) ** 2
        + ((pred_selected_upper - true_selected_upper) / scale) ** 2
    )
    selected_width = torch_module.mean(
        ((pred_selected_width - true_selected_width) / scale) ** 2
    )
    selected_center = torch_module.mean(
        ((pred_selected_center - true_selected_center) / scale) ** 2
    )
    selected_upper_profile = torch_module.mean(
        ((pred_relative_profile - true_relative_profile) / scale) ** 2
    )
    total = (
        config.full_spectrum_weight * full_spectrum
        + config.gap_profile_weight * gap_profile
        + config.selected_edge_weight * selected_edge
        + config.selected_width_weight * selected_width
        + config.selected_center_weight * selected_center
        + config.selected_upper_profile_weight * selected_upper_profile
    )
    return GapAwareLossTerms(
        total=total,
        full_spectrum=full_spectrum,
        gap_profile=gap_profile,
        selected_edge=selected_edge,
        selected_width=selected_width,
        selected_center=selected_center,
        selected_upper_profile=selected_upper_profile,
    )


__all__ = [
    "BASE_LOSS_NAME",
    "GAP_AWARE_LOSS_NAME",
    "GapAwareLossConfig",
    "GapAwareLossTerms",
    "gap_aware_spectrum_loss",
    "resolve_gap_aware_loss_config",
]
