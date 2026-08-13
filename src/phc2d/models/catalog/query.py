"""Built-in catalog entries for query-conditioned models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

from phc2d.models.registry import ModelFactory, ModelSpec


def _build_mlp_query(**kwargs: Any) -> Any:
    from phc2d.models.query import build_mlp_query

    return build_mlp_query(**kwargs)


def _build_deep_onet_query(**kwargs: Any) -> Any:
    from phc2d.models.query import build_deep_onet_query

    return build_deep_onet_query(**kwargs)


def _build_fno2d_query(**kwargs: Any) -> Any:
    from phc2d.models.query import build_fno2d_query

    return build_fno2d_query(**kwargs)


for _factory in (_build_mlp_query, _build_deep_onet_query, _build_fno2d_query):
    _factory.__phc2d_factory_identity__ = f"phc2d.models.registry.{_factory.__name__}"


def model_specs() -> tuple[tuple[str, ModelFactory, ModelSpec], ...]:
    """Return the built-in specs owned by this model family."""

    return (
        (
            "mlp_query",
            _build_mlp_query,
            ModelSpec(
                name="mlp_query",
                family="mlp_query",
                input_repr="epsilon_grid+query",
                output_target="band_query",
                supports_variable_grid=True,
                supports_query=True,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (3,),
                    "k_dim": 1,
                    "hidden_dim": 64,
                    "depth": 2,
                },
                source="phc2d Full-2A query MLP baseline",
            ),
        ),
        (
            "deep_onet_query",
            _build_deep_onet_query,
            ModelSpec(
                name="deep_onet_query",
                family="deeponet",
                input_repr="epsilon_grid+query",
                output_target="band_query",
                supports_variable_grid=True,
                supports_query=True,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (3,),
                    "k_dim": 1,
                    "branch_width": 64,
                    "trunk_width": 64,
                    "latent_dim": 32,
                    "depth": 2,
                },
                source="phc2d Full-2A DeepONet-style query model",
            ),
        ),
        (
            "fno2d_query",
            _build_fno2d_query,
            ModelSpec(
                name="fno2d_query",
                family="neural_operator_query",
                input_repr="epsilon_grid+query",
                output_target="band_query",
                supports_variable_grid=True,
                supports_query=True,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 12, 14),
                    "output_shape": (3,),
                    "k_dim": 1,
                    "width": 16,
                    "depth": 2,
                    "modes": (6, 7),
                    "query_hidden_dim": 32,
                },
                source="phc2d Full-2A FNO query-conditioning model",
            ),
        ),
        (
            "deep_onet_query_v2",
            _build_deep_onet_query,
            ModelSpec(
                name="deep_onet_query_v2",
                family="deeponet",
                input_repr="epsilon_grid+query",
                output_target="band_query",
                supports_variable_grid=True,
                supports_query=True,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 16, 16),
                    "output_shape": (10,),
                    "k_dim": 1,
                    "branch_width": 128,
                    "trunk_width": 128,
                    "latent_dim": 128,
                    "depth": 4,
                    "activation": "gelu",
                },
                source="phc2d formal V2 10k DeepONet query model",
            ),
        ),
        (
            "fno2d_query_v2",
            _build_fno2d_query,
            ModelSpec(
                name="fno2d_query_v2",
                family="neural_operator_query",
                input_repr="epsilon_grid+query",
                output_target="band_query",
                supports_variable_grid=True,
                supports_query=True,
                supports_field=False,
                requires=("torch",),
                defaults={
                    "input_shape": (1, 16, 16),
                    "output_shape": (10,),
                    "k_dim": 1,
                    "width": 32,
                    "depth": 4,
                    "modes": (8, 8),
                    "query_hidden_dim": 64,
                    "activation": "gelu",
                },
                source="phc2d formal V2 10k gated FNO query model",
            ),
        ),
    )
