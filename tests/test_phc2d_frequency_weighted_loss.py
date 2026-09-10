"""Contract tests for EXP-FWD-018 bounded theory-frequency weighting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from phc2d.forward.frequency_weighted_loss import (
    FREQUENCY_WEIGHTED_LOSS_NAME,
    bounded_inverse_frequency_loss,
    frequency_center_from_material_weight,
    precompute_frequency_centers,
    resolve_frequency_weighted_loss_config,
)


def _config(**changes: object):
    raw = {
        "name": FREQUENCY_WEIGHTED_LOSS_NAME,
        "epsilon_air": 1.0,
        "epsilon_material": 9.0,
        "epsilon_f": 0.01,
        "rho": 4.0,
        "alpha": 0.25,
    }
    raw.update(changes)
    return resolve_frequency_weighted_loss_config(raw)


def test_frequency_center_matches_uniform_and_harmonic_te_contract() -> None:
    config = _config()
    assert config is not None
    uniform = frequency_center_from_material_weight(np.ones((2, 4, 4)), config=config)
    assert uniform.tolist() == pytest.approx([1.0 / 6.0, 1.0 / 6.0])
    mixed = frequency_center_from_material_weight(np.asarray([[[0.0, 0.0], [1.0, 1.0]]]), config=config)
    expected_index = 1.0 / np.sqrt(np.mean([1.0, 1.0, 1.0 / 9.0, 1.0 / 9.0]))
    assert mixed.tolist() == pytest.approx([1.0 / (2.0 * expected_index)])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_bounded_weights_are_finite_monotone_and_mean_one() -> None:
    import torch

    config = _config(alpha=0.25)
    assert config is not None
    target = torch.tensor([[[0.10, 0.20, 0.50], [0.11, 0.30, 0.60]]])
    prediction = target.clone().requires_grad_(True)
    terms = bounded_inverse_frequency_loss(
        prediction, target, target_mean=torch.zeros((2, 3)), target_std=torch.ones((2, 3)),
        frequency_center=torch.tensor([0.20]), config=config, torch_module=torch,
    )
    assert float(terms.weight_mean) == pytest.approx(1.0)
    assert float(terms.weight_max) > float(terms.weight_min)
    assert bool(torch.isfinite(terms.total))
    terms.total.backward()
    assert bool(torch.isfinite(prediction.grad).all())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_alpha_zero_is_exact_baseline() -> None:
    import torch

    config = _config(alpha=0.0)
    assert config is not None
    prediction = torch.tensor([[[0.0, 0.2], [0.3, 0.4]]])
    target = torch.tensor([[[0.1, 0.1], [0.2, 0.6]]])
    terms = bounded_inverse_frequency_loss(
        prediction, target, target_mean=torch.zeros((2, 2)), target_std=torch.ones((2, 2)),
        frequency_center=torch.tensor([0.2]), config=config, torch_module=torch,
    )
    assert float(terms.total) == pytest.approx(float(torch.mean((prediction - target) ** 2)))


@pytest.mark.parametrize("changes", (
    {"epsilon_air": 0.0}, {"epsilon_material": 1.0}, {"epsilon_f": 0.0},
    {"rho": -1.0}, {"alpha": 1.1},
))
def test_invalid_frequency_weight_parameters_fail(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _config(**changes)


@pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py is optional")
def test_frequency_center_precompute_reads_only_requested_rows(tmp_path) -> None:
    import h5py

    path = tmp_path / "lane.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("material_weight_grid_4", data=np.asarray([
            np.zeros((4, 4)), np.ones((4, 4)), np.full((4, 4), np.nan),
        ], dtype=np.float32))
        handle.create_dataset("sample_id", data=np.asarray([b"train0", b"validation0", b"test0"]))
    config = _config()
    assert config is not None
    centers, ids = precompute_frequency_centers(path, grid_field="material_weight_grid_4", indices=np.asarray([0, 1]), config=config)
    assert centers.shape == (2,)
    assert ids.tolist() == ["train0", "validation0"]
