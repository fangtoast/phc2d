"""Catalog entries for Full-Wh bridge and component-attribution models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models.registry import ModelFactory, ModelSpec


def _build_global_local_band_operator_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_operators import build_global_local_band_operator_v1

    return build_global_local_band_operator_v1(**kwargs)


def _build_fno2d_band_component_v1(**kwargs: Any) -> Any:
    from phc2d.models.band_operators import build_fno2d_band

    return build_fno2d_band(**kwargs)


def _build_tfno2d_band_component_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_operators import build_tfno2d_band_v3

    return build_tfno2d_band_v3(**kwargs)


def _build_convnext_band_component_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_convnext import build_convnext_band_v1

    return build_convnext_band_v1(**kwargs)


def _build_dual_lane_band_component_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_operators import build_dual_lane_band_operator_v1

    return build_dual_lane_band_operator_v1(**kwargs)


for _factory in (
    _build_global_local_band_operator_v1,
    _build_fno2d_band_component_v1,
    _build_tfno2d_band_component_v1,
    _build_convnext_band_component_v1,
    _build_dual_lane_band_component_v1,
):
    _factory.__phc2d_factory_identity__ = f"phc2d.models.registry.{_factory.__name__}"


def _operator_defaults(*, channel_expansion: int, local_kernel: int) -> dict[str, Any]:
    return {
        "input_shape": (1, 64, 64),
        "output_shape": (81, 10),
        "width": 64,
        "depth": 6,
        "modes": (12, 12),
        "factorization_rank": 0.75,
        "norm_groups": 8,
        "channel_expansion": channel_expansion,
        "local_kernel": local_kernel,
        "drop_path": 0.1,
        "head_kind": "dense",
        "k_points": None,
        "head_hidden_dim": 256,
        "band_embedding_dim": 32,
        "fourier_frequencies": (1, 2, 4, 8),
        "dropout": 0.05,
    }


def model_specs() -> tuple[tuple[str, ModelFactory, ModelSpec], ...]:
    """Return experimental bridge and attribution-only model specs."""

    operator_defaults = _operator_defaults(channel_expansion=2, local_kernel=3)
    return (
        (
            "global_local_band_operator_v1",
            _build_global_local_band_operator_v1,
            ModelSpec(
                name="global_local_band_operator_v1",
                family="global_local_neural_operator",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    **_operator_defaults(channel_expansion=4, local_kernel=7),
                    "enabled_branches": ("spectral", "local"),
                    "dense_head_hidden_dim": None,
                },
                source=(
                    "phc2d EXP-FWD-019-derived fixed-fusion spectral and "
                    "periodic large-kernel bridge"
                ),
            ),
        ),
        (
            "fno2d_band_component_v1",
            _build_fno2d_band_component_v1,
            ModelSpec(
                name="fno2d_band_component_v1",
                family="neural_operator_component_ablation",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64),
                    "output_shape": (81, 10),
                    "width": 32,
                    "depth": 4,
                    "modes": (8, 8),
                    "band_modes": (5, 5),
                    "fourier_rank": 8,
                    "activation": "gelu",
                    "use_primary_spectral": True,
                    "use_band_spectral": True,
                    "use_band_gate": True,
                    "use_low_rank": True,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                    "dropout": 0.05,
                },
                source="phc2d component-attribution FNO with shared dense head",
            ),
        ),
        (
            "tfno2d_band_component_v1",
            _build_tfno2d_band_component_v1,
            ModelSpec(
                name="tfno2d_band_component_v1",
                family="neural_operator_component_ablation",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    **operator_defaults,
                    "enabled_branches": ("spectral", "local", "channel"),
                },
                source="phc2d TFNO branch component ablation",
            ),
        ),
        (
            "convnext_band_component_v1",
            _build_convnext_band_component_v1,
            ModelSpec(
                name="convnext_band_component_v1",
                family="convnext_component_ablation",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64),
                    "output_shape": (81, 10),
                    "stage_dims": (48, 96, 192),
                    "stage_depths": (2, 3, 6),
                    "stem_stride": 4,
                    "kernel_size": 7,
                    "channel_expansion": 4,
                    "drop_path": 0.1,
                    "layer_scale_init": 1.0e-6,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                    "dropout": 0.05,
                    "padding_mode": "circular",
                    "hierarchical_downsample": True,
                },
                source="phc2d ConvNeXt component ablation",
            ),
        ),
        (
            "dual_lane_band_component_v1",
            _build_dual_lane_band_component_v1,
            ModelSpec(
                name="dual_lane_band_component_v1",
                family="conditional_operator_component_ablation",
                input_repr="material_weight_grid+lane",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                accepts_lane_id=True,
                requires=("torch",),
                defaults={
                    **operator_defaults,
                    "num_lanes": 2,
                    "lane_embedding_dim": 16,
                    "enabled_branches": ("spectral", "local", "channel"),
                },
                source="phc2d joint branch component ablation",
            ),
        ),
    )
