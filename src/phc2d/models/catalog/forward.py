"""Built-in catalog entries for Full-Wh forward baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models.registry import ModelFactory, ModelSpec


def _build_res_mlp(**kwargs: Any) -> Any:
    from phc2d.models.band_mlp import build_res_mlp

    return build_res_mlp(**kwargs)


def _build_mlp(**kwargs: Any) -> Any:
    from phc2d.models.band_mlp import build_mlp

    return build_mlp(**kwargs)


def _build_pod_res_mlp(**kwargs: Any) -> Any:
    from phc2d.models.band_mlp import build_pod_res_mlp

    return build_pod_res_mlp(**kwargs)


def _build_lifted_cnn(**kwargs: Any) -> Any:
    from phc2d.models.band_cnn import build_lifted_cnn

    return build_lifted_cnn(**kwargs)


def _build_fno_lite(**kwargs: Any) -> Any:
    from phc2d.models.band_operators import build_fno_lite

    return build_fno_lite(**kwargs)


def _build_fno2d_band(**kwargs: Any) -> Any:
    from phc2d.models.band_operators import build_fno2d_band

    return build_fno2d_band(**kwargs)


for _factory in (
    _build_res_mlp,
    _build_mlp,
    _build_pod_res_mlp,
    _build_lifted_cnn,
    _build_fno_lite,
    _build_fno2d_band,
):
    _factory.__phc2d_factory_identity__ = f"phc2d.models.registry.{_factory.__name__}"


def model_specs() -> tuple[tuple[str, ModelFactory, ModelSpec], ...]:
    """Return the built-in specs owned by this model family."""

    return (
        (
            "mlp",
            _build_mlp,
            ModelSpec(
                name="mlp",
                family="mlp",
                input_repr="rho_wedge",
                output_target="Wh",
                supports_variable_grid=False,
                supports_query=False,
                supports_field=False,
                input_layout="flat",
                requires=("torch",),
                defaults={
                    "input_shape": (36,),
                    "output_shape": (31, 10),
                    "hidden_dim": 256,
                    "depth": 6,
                    "dropout": 0.05,
                    "activation": "gelu",
                },
                source="phc2d A1.1 plain-MLP residual-connection ablation",
            ),
        ),
        (
            "res_mlp",
            _build_res_mlp,
            ModelSpec(
                name="res_mlp",
                family="mlp",
                input_repr="rho_wedge",
                output_target="Wh",
                supports_variable_grid=False,
                supports_query=False,
                supports_field=False,
                input_layout="flat",
                requires=("torch",),
                defaults={
                    "input_shape": (8,),
                    "output_shape": (5, 3),
                    "hidden_dim": 32,
                    "depth": 2,
                },
                source="phc2d Gate 2 minimal baseline",
            ),
        ),
        (
            "pod_res_mlp",
            _build_pod_res_mlp,
            ModelSpec(
                name="pod_res_mlp",
                family="pod_mlp",
                input_repr="rho_wedge",
                output_target="Wh",
                supports_variable_grid=False,
                supports_query=False,
                supports_field=False,
                input_layout="flat",
                requires=("torch", "pod_basis"),
                defaults={
                    "input_shape": (8,),
                    "output_shape": (5, 3),
                    "hidden_dim": 32,
                    "depth": 2,
                },
                source="phc2d Gate 2 placeholder; POD basis contract not implemented",
            ),
        ),
        (
            "lifted_cnn",
            _build_lifted_cnn,
            ModelSpec(
                name="lifted_cnn",
                family="cnn",
                input_repr="epsilon_grid",
                output_target="Wh",
                supports_variable_grid=True,
                supports_query=False,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (5, 3),
                    "channels": 12,
                },
                source="phc2d Gate 2 minimal image baseline",
            ),
        ),
        (
            "fno_lite",
            _build_fno_lite,
            ModelSpec(
                name="fno_lite",
                family="neural_operator",
                input_repr="epsilon_grid",
                output_target="Wh",
                supports_variable_grid=True,
                supports_query=False,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (5, 3),
                    "width": 12,
                    "depth": 2,
                    "modes": (4, 5),
                },
                source="phc2d Gate 2 minimal FNO-style baseline",
            ),
        ),
        (
            "fno2d_band",
            _build_fno2d_band,
            ModelSpec(
                name="fno2d_band",
                canonical_name="fno2d_band",
                family="neural_operator",
                input_repr="epsilon_grid",
                output_target="Wh",
                supports_variable_grid=True,
                supports_query=False,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (5, 3),
                    "width": 12,
                    "depth": 2,
                    "modes": (6, 7),
                    "band_modes": (4, 5),
                    "fourier_rank": 4,
                },
                source="phc2d P1 band-matrix FNO pilot; dedicated fno2d_band architecture",
            ),
        ),
        (
            "res_mlp_wide",
            _build_res_mlp,
            ModelSpec(
                name="res_mlp_wide",
                family="mlp",
                input_repr="rho_wedge",
                output_target="Wh",
                supports_variable_grid=False,
                supports_query=False,
                supports_field=False,
                input_layout="flat",
                requires=("torch",),
                defaults={
                    "input_shape": (36,),
                    "output_shape": (31, 10),
                    "hidden_dim": 256,
                    "depth": 6,
                    "dropout": 0.05,
                    "activation": "gelu",
                },
                source="phc2d formal V2 10k wide residual MLP",
            ),
        ),
        (
            "fno2d_band_v2",
            _build_fno2d_band,
            ModelSpec(
                name="fno2d_band_v2",
                family="neural_operator",
                input_repr="epsilon_grid",
                output_target="Wh",
                supports_variable_grid=True,
                supports_query=False,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 16, 16),
                    "output_shape": (31, 10),
                    "width": 32,
                    "depth": 4,
                    "modes": (8, 8),
                    "band_modes": (5, 5),
                    "fourier_rank": 8,
                    "activation": "gelu",
                },
                source="phc2d formal V2 10k band-aware FNO",
            ),
        ),
    )
