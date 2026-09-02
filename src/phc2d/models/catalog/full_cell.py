"""Built-in catalog entries for project-native Full-Wh upgrades.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models.registry import ModelFactory, ModelSpec


def _build_mlp_mixer_band_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_mixer import build_mlp_mixer_band_v1

    return build_mlp_mixer_band_v1(**kwargs)


def _build_res_mlp_gated_v2(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_resmlp import build_res_mlp_gated_v2

    return build_res_mlp_gated_v2(**kwargs)


def _build_convnext_band_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_convnext import build_convnext_band_v1

    return build_convnext_band_v1(**kwargs)


def _build_tfno2d_band_v3(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_operators import build_tfno2d_band_v3

    return build_tfno2d_band_v3(**kwargs)


def _build_dual_lane_band_operator_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_operators import build_dual_lane_band_operator_v1

    return build_dual_lane_band_operator_v1(**kwargs)


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
    _build_mlp_mixer_band_v1,
    _build_res_mlp_gated_v2,
    _build_convnext_band_v1,
    _build_tfno2d_band_v3,
    _build_dual_lane_band_operator_v1,
    _build_tfno2d_band_component_v1,
    _build_convnext_band_component_v1,
    _build_dual_lane_band_component_v1,
):
    _factory.__phc2d_factory_identity__ = f"phc2d.models.registry.{_factory.__name__}"


def model_specs() -> tuple[tuple[str, ModelFactory, ModelSpec], ...]:
    """Return the built-in specs owned by this model family."""

    return (
        (
            "mlp_mixer_band_v1",
            _build_mlp_mixer_band_v1,
            ModelSpec(
                name="mlp_mixer_band_v1",
                family="mixer",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64),
                    "output_shape": (81, 10),
                    "token_grid": (16, 16),
                    "embed_dim": 256,
                    "depth": 8,
                    "token_mlp_dim": 128,
                    "channel_mlp_dim": 512,
                    "dropout": 0.05,
                    "drop_path": 0.1,
                    "layer_scale_init": 1.0e-4,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                },
                source="phc2d project-native MLP-Mixer-style full-cell upgrade",
            ),
        ),
        (
            "res_mlp_gated_v2",
            _build_res_mlp_gated_v2,
            ModelSpec(
                name="res_mlp_gated_v2",
                family="resmlp",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=False,
                input_layout="flat",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    "input_shape": (4096,),
                    "output_shape": (81, 10),
                    "hidden_dim": 320,
                    "depth": 8,
                    "expansion": 2,
                    "dropout": 0.05,
                    "drop_path": 0.1,
                    "layer_scale_init": 1.0e-4,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                },
                source="phc2d project-native gated deep residual MLP upgrade",
            ),
        ),
        (
            "convnext_band_v1",
            _build_convnext_band_v1,
            ModelSpec(
                name="convnext_band_v1",
                family="convnext",
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
                },
                source="phc2d project-native periodic ConvNeXt-style full-cell upgrade",
            ),
        ),
        (
            "tfno2d_band_v3",
            _build_tfno2d_band_v3,
            ModelSpec(
                name="tfno2d_band_v3",
                family="neural_operator",
                input_repr="material_weight_grid",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64),
                    "output_shape": (81, 10),
                    "width": 64,
                    "depth": 6,
                    "modes": (12, 12),
                    "factorization_rank": 0.75,
                    "norm_groups": 8,
                    "channel_expansion": 2,
                    "local_kernel": 3,
                    "drop_path": 0.1,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                    "dropout": 0.05,
                },
                source="phc2d project-native Tucker-factorized two-sided FNO-style upgrade",
            ),
        ),
        (
            "dual_lane_band_operator_v1",
            _build_dual_lane_band_operator_v1,
            ModelSpec(
                name="dual_lane_band_operator_v1",
                family="conditional_neural_operator",
                input_repr="material_weight_grid+lane",
                output_target="Wh",
                supports_variable_grid=True,
                input_layout="image",
                accepts_k_points=True,
                accepts_lane_id=True,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64),
                    "output_shape": (81, 10),
                    "width": 64,
                    "depth": 6,
                    "modes": (12, 12),
                    "factorization_rank": 0.75,
                    "norm_groups": 8,
                    "channel_expansion": 2,
                    "local_kernel": 3,
                    "drop_path": 0.1,
                    "num_lanes": 2,
                    "lane_embedding_dim": 16,
                    "head_kind": "dense",
                    "k_points": None,
                    "head_hidden_dim": 256,
                    "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8),
                    "dropout": 0.05,
                },
                source="phc2d project-native binary/graded FiLM-conditioned band operator",
            ),
        ),
        (
            "tfno2d_band_component_v1",
            _build_tfno2d_band_component_v1,
            ModelSpec(
                name="tfno2d_band_component_v1", family="neural_operator_component_ablation",
                input_repr="material_weight_grid", output_target="Wh", supports_variable_grid=True,
                input_layout="image", accepts_k_points=True, requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64), "output_shape": (81, 10), "width": 64,
                    "depth": 6, "modes": (12, 12), "factorization_rank": 0.75,
                    "norm_groups": 8, "channel_expansion": 2, "local_kernel": 3,
                    "drop_path": 0.1, "head_kind": "dense", "k_points": None,
                    "head_hidden_dim": 256, "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8), "dropout": 0.05,
                    "enabled_branches": ("spectral", "local", "channel"),
                },
                source="phc2d TFNO branch component ablation",
            ),
        ),
        (
            "convnext_band_component_v1",
            _build_convnext_band_component_v1,
            ModelSpec(
                name="convnext_band_component_v1", family="convnext_component_ablation",
                input_repr="material_weight_grid", output_target="Wh", supports_variable_grid=True,
                input_layout="image", accepts_k_points=True, requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64), "output_shape": (81, 10),
                    "stage_dims": (48, 96, 192), "stage_depths": (2, 3, 6), "stem_stride": 4,
                    "kernel_size": 7, "channel_expansion": 4, "drop_path": 0.1,
                    "layer_scale_init": 1.0e-6, "head_kind": "dense", "k_points": None,
                    "head_hidden_dim": 256, "band_embedding_dim": 32,
                    "fourier_frequencies": (1, 2, 4, 8), "dropout": 0.05,
                    "padding_mode": "circular", "hierarchical_downsample": True,
                },
                source="phc2d ConvNeXt component ablation",
            ),
        ),
        (
            "dual_lane_band_component_v1",
            _build_dual_lane_band_component_v1,
            ModelSpec(
                name="dual_lane_band_component_v1", family="conditional_operator_component_ablation",
                input_repr="material_weight_grid+lane", output_target="Wh", supports_variable_grid=True,
                input_layout="image", accepts_k_points=True, accepts_lane_id=True, requires=("torch",),
                defaults={
                    "input_shape": (1, 64, 64), "output_shape": (81, 10), "width": 64,
                    "depth": 6, "modes": (12, 12), "factorization_rank": 0.75,
                    "norm_groups": 8, "channel_expansion": 2, "local_kernel": 3,
                    "drop_path": 0.1, "num_lanes": 2, "lane_embedding_dim": 16,
                    "head_kind": "dense", "k_points": None, "head_hidden_dim": 256,
                    "band_embedding_dim": 32, "fourier_frequencies": (1, 2, 4, 8),
                    "dropout": 0.05, "enabled_branches": ("spectral", "local", "channel"),
                },
                source="phc2d joint branch component ablation",
            ),
        ),
    )
