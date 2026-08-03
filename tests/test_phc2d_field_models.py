"""Tests for trainable TE eigenfield baseline model registration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util

import pytest

from phc2d.models import build_model, get_model_spec


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
@pytest.mark.parametrize("name", ["unet_field", "fno_field"])
def test_field_models_are_registered_and_preserve_schema_grid_shape(name: str) -> None:
    import torch

    model = build_model(name, input_channels=4, field_channels=12, width=8, depth=2, modes=(4, 4))
    field, frequency = model(torch.randn(2, 4, 19, 23))
    assert get_model_spec(name).supports_field is True
    assert tuple(field.shape) == (2, 12, 19, 23)
    assert tuple(frequency.shape) == (2,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
@pytest.mark.parametrize("name", ["unet_field_v2", "fno_field_v2"])
@pytest.mark.parametrize("shape", [(19, 23), (32, 28)])
def test_conditioned_field_v2_models_preserve_variable_grid(name: str, shape: tuple[int, int]) -> None:
    import torch

    model = build_model(name, grid_channels=5, context_dim=3, field_channels=6, width=8, depth=2, modes=(4, 4))
    field, frequency = model(torch.randn(2, 5, *shape), torch.randn(2, 3))
    assert get_model_spec(name).supports_variable_grid is True
    assert tuple(field.shape) == (2, 6, *shape)
    assert tuple(frequency.shape) == (2,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_fno_v2_registers_negative_spectral_modes() -> None:
    model = build_model("fno_field_v2", grid_channels=5, context_dim=3, field_channels=6, width=8, depth=1, modes=(4, 4))
    names = {name for name, _ in model.named_parameters()}
    assert any("weight_positive" in name for name in names)
    assert any("weight_negative" in name for name in names)
    positive_only = build_model("fno_field_v2", grid_channels=5, context_dim=3, field_channels=6, width=8, depth=1, modes=(4, 4), corrected_spectral_modes=False)
    assert not any("weight_negative" in name for name, _ in positive_only.named_parameters())
