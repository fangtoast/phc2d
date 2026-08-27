"""Tests for differentiable Full-Wh gap-aware training objectives.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util

import pytest

from phc2d.forward.gap_aware_loss import (
    BASE_LOSS_NAME,
    GAP_AWARE_LOSS_NAME,
    gap_aware_spectrum_loss,
    resolve_gap_aware_loss_config,
)
from phc2d.forward.full_cell_metrics import compute_gap_decision_metrics


def _config(**overrides: float | int) -> object:
    raw = {
        "name": GAP_AWARE_LOSS_NAME,
        "lower_band_min_one_based": 1,
        "lower_band_max_one_based": 3,
        "full_spectrum_weight": 1.0,
        "gap_profile_weight": 0.25,
        "selected_edge_weight": 0.25,
        "selected_width_weight": 0.25,
        "selected_center_weight": 0.25,
        **overrides,
    }
    return resolve_gap_aware_loss_config(raw)


def test_gap_aware_config_preserves_legacy_and_rejects_unsafe_contracts() -> None:
    assert resolve_gap_aware_loss_config(BASE_LOSS_NAME) is None
    with pytest.raises(ValueError, match="positive full-spectrum"):
        _config(full_spectrum_weight=0.0)
    with pytest.raises(ValueError, match="at least one positive auxiliary"):
        _config(
            gap_profile_weight=0.0,
            selected_edge_weight=0.0,
            selected_width_weight=0.0,
            selected_center_weight=0.0,
        )
    with pytest.raises(ValueError, match="unknown gap-aware"):
        resolve_gap_aware_loss_config({"name": GAP_AWARE_LOSS_NAME, "test_center": 0.5})


def test_gap_aware_loss_is_zero_for_exact_spectrum() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    import torch

    target = torch.tensor(
        [[[0.10, 0.20, 0.50, 0.70], [0.12, 0.22, 0.52, 0.72], [0.11, 0.21, 0.51, 0.71]]]
    )
    terms = gap_aware_spectrum_loss(
        target.clone(),
        target,
        target_mean=torch.zeros((3, 4)),
        target_std=torch.ones((3, 4)),
        config=_config(),
        torch_module=torch,
    )
    assert all(float(value) == pytest.approx(0.0) for value in terms.as_dict().values())


def test_gap_aware_loss_penalizes_gap_edges_and_has_finite_gradients() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    import torch

    target = torch.tensor(
        [[[0.10, 0.20, 0.50, 0.70], [0.12, 0.22, 0.52, 0.72], [0.11, 0.21, 0.51, 0.71]]]
    )
    prediction = target.clone()
    prediction[0, 0, 2] -= 0.05
    prediction.requires_grad_(True)
    terms = gap_aware_spectrum_loss(
        prediction,
        target,
        target_mean=torch.zeros((3, 4)),
        target_std=torch.ones((3, 4)),
        config=_config(),
        torch_module=torch,
    )
    assert float(terms.gap_profile.detach()) > 0.0
    assert float(terms.selected_edge.detach()) > 0.0
    assert float(terms.selected_width.detach()) > 0.0
    assert float(terms.selected_center.detach()) > 0.0
    terms.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert float(prediction.grad.abs().sum()) > 0.0


def test_gap_aware_loss_uses_metadata_driven_band_count() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    import torch

    target = torch.zeros((2, 5, 3))
    config = _config(lower_band_max_one_based=3)
    with pytest.raises(ValueError, match="exceeds"):
        gap_aware_spectrum_loss(
            target,
            target,
            target_mean=torch.zeros((5, 3)),
            target_std=torch.ones((5, 3)),
            config=config,
            torch_module=torch,
        )


def test_gap_decision_metrics_separate_complete_and_pseudo_targets() -> None:
    import numpy as np

    target = np.asarray(
        [
            [[0.10, 0.20, 0.50], [0.12, 0.22, 0.52]],
            [[0.10, 0.20, 0.21], [0.30, 0.31, 0.32]],
        ]
    )
    prediction = target.copy()
    metrics = compute_gap_decision_metrics(
        prediction,
        target,
        lower_band_min_one_based=1,
        lower_band_max_one_based=2,
    )
    assert metrics["band_accuracy"] == pytest.approx(1.0)
    assert metrics["upper_edge_k_accuracy"] == pytest.approx(1.0)
    assert metrics["joint_band_k_accuracy"] == pytest.approx(1.0)
    assert metrics["selected_frequency_mre_pct"] == pytest.approx(0.0)
    assert metrics["complete_gap_sample_count"] == 1
    assert metrics["pseudo_gap_sample_count"] == 1
