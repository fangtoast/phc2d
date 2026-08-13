"""Built-in catalog entries for TE Bloch eigenfield models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from phc2d.models.registry import (
    ModelFactory,
    ModelSpec,
    _build_fno_field,
    _build_fno_field_v2,
    _build_unet_field,
    _build_unet_field_v2,
)


def model_specs() -> tuple[tuple[str, ModelFactory, ModelSpec], ...]:
    """Return the built-in specs owned by this model family."""

    return (
        (
            "unet_field_v2",
            _build_unet_field_v2,
            ModelSpec(
                name="unet_field_v2",
                family="unet",
                input_repr="epsilon_periodic_grid+field_context",
                output_target="active_te_field_eigen",
                supports_variable_grid=True,
                supports_field=True,
                requires=("torch",),
                defaults={
                    "grid_channels": 5,
                    "context_dim": 3,
                    "field_channels": 6,
                    "width": 48,
                    "depth": 3,
                    "use_film": True,
                },
                source="phc2d B3 conditioned supervised TE eigenfield baseline",
            ),
        ),
        (
            "fno_field_v2",
            _build_fno_field_v2,
            ModelSpec(
                name="fno_field_v2",
                family="neural_operator",
                input_repr="epsilon_periodic_grid+field_context",
                output_target="active_te_field_eigen",
                supports_variable_grid=True,
                supports_field=True,
                requires=("torch",),
                defaults={
                    "grid_channels": 5,
                    "context_dim": 3,
                    "field_channels": 6,
                    "width": 48,
                    "depth": 4,
                    "modes": (16, 16),
                    "use_film": True,
                    "corrected_spectral_modes": True,
                },
                source="phc2d B3 data-only FNO and adapted PINO shared architecture",
            ),
        ),
        (
            "unet_field",
            _build_unet_field,
            ModelSpec(
                name="unet_field",
                family="unet",
                input_repr="epsilon_grid+field_context",
                output_target="field_eigen_te",
                supports_variable_grid=True,
                supports_field=True,
                requires=("torch",),
                defaults={"input_channels": 4, "field_channels": 12, "width": 32, "depth": 2},
                source="phc2d supervised TE eigenfield baseline",
            ),
        ),
        (
            "fno_field",
            _build_fno_field,
            ModelSpec(
                name="fno_field",
                family="neural_operator",
                input_repr="epsilon_grid+field_context",
                output_target="field_eigen_te",
                supports_variable_grid=True,
                supports_field=True,
                requires=("torch",),
                defaults={
                    "input_channels": 4,
                    "field_channels": 12,
                    "width": 32,
                    "depth": 4,
                    "modes": (12, 12),
                },
                source="phc2d data-only FNO-style TE eigenfield baseline",
            ),
        ),
    )
