"""Tests for inverse uniqueness-repair objective terms.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from phc2d.inverse import CandidateBatch, candidate_batch_metrics
from phc2d.inverse.objective import objective_loss_components, quality_weighted_collision_loss


def test_quality_weighted_collision_loss_penalizes_exact_duplicates() -> None:
    torch = pytest.importorskip("torch")
    candidates = torch.tensor(
        [[0.9, 0.9, 0.1], [0.9, 0.9, 0.1], [0.1, 0.9, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.ones((3, 6), dtype=torch.float32)
    predictions = target.clone()

    loss, _, nn_min, nn_p10, _ = quality_weighted_collision_loss(
        candidates,
        predictions=predictions,
        target=target,
        torch=torch,
        n_bands=3,
        target_nn_hamming=0.1,
        group_size=3,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert nn_min.item() == pytest.approx(0.0)
    assert nn_p10.item() == pytest.approx(0.0)
    assert candidates.grad is not None


def test_quality_weighted_collision_loss_is_zero_for_separated_candidates() -> None:
    torch = pytest.importorskip("torch")
    candidates = torch.tensor(
        [[0.9, 0.9, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.ones((3, 6), dtype=torch.float32)
    predictions = target.clone()

    loss, _, nn_min, _, _ = quality_weighted_collision_loss(
        candidates,
        predictions=predictions,
        target=target,
        torch=torch,
        n_bands=3,
        target_nn_hamming=0.1,
        group_size=3,
    )

    assert nn_min.item() > 0.1
    assert loss.item() == pytest.approx(0.0)


def test_objective_collision_aliases_are_differentiable_and_shape_free() -> None:
    torch = pytest.importorskip("torch")
    predictions = torch.ones((4, 8), dtype=torch.float32, requires_grad=True)
    target = torch.ones((4, 8), dtype=torch.float32)
    candidates = torch.tensor(
        [[0.9, 0.9, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9, 0.9], [0.9, 0.1, 0.9, 0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )

    total, metrics = objective_loss_components(
        predictions=predictions,
        target=target,
        candidates=candidates,
        reconstruction=None,
        reference=None,
        torch=torch,
        n_bands=4,
        objective={
            "surrogate_weight": 1.0,
            "collision_weight": 0.6,
            "target_nearest_neighbor_hamming": 0.1,
            "collision_soft_binary_temperature": 0.08,
            "collision_quality_temperature": 0.03,
            "collision_quality_detach": True,
            "n_objective_candidates": 4,
        },
    )
    total.backward()

    assert metrics["quality_weighted_collision_loss"] > 0.0
    assert metrics["nearest_neighbor_hamming_min"] == pytest.approx(0.0)
    assert metrics["nearest_neighbor_hamming_p10"] == pytest.approx(0.0)
    assert predictions.grad is not None
    assert candidates.grad is not None


def test_hard_duplicate_metrics_are_stable_for_edge_cases() -> None:
    duplicate_batch = CandidateBatch(
        "dup",
        {"Wh": [1.0]},
        np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "rho_wedge",
    )
    duplicate_metrics = candidate_batch_metrics(duplicate_batch)
    assert duplicate_metrics["duplicate_cluster_count"] == pytest.approx(1.0)
    assert duplicate_metrics["largest_duplicate_cluster_fraction"] == pytest.approx(2 / 3)

    unique_batch = CandidateBatch("unique", {"Wh": [1.0]}, np.eye(3, dtype=np.float32), "rho_wedge")
    unique_metrics = candidate_batch_metrics(unique_batch)
    assert unique_metrics["duplicate_cluster_count"] == pytest.approx(0.0)
    assert unique_metrics["largest_duplicate_cluster_fraction"] == pytest.approx(0.0)

    single_batch = CandidateBatch("single", {"Wh": [1.0]}, np.ones((1, 3), dtype=np.float32), "rho_wedge")
    single_metrics = candidate_batch_metrics(single_batch)
    assert single_metrics["nearest_neighbor_hamming_mean"] == pytest.approx(0.0)
    assert single_metrics["duplicate_cluster_count"] == pytest.approx(0.0)
