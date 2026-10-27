"""Lazy registry entries for official baselines and channel-gated Global-Local.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations
from typing import Any
from phc2d.models.registry import ModelSpec


def official_fno(**kwargs: Any) -> Any:
    from phc2d.models.official_band_operators import build_official_fno_band_v1
    return build_official_fno_band_v1(**kwargs)


def official_tfno(**kwargs: Any) -> Any:
    from phc2d.models.official_band_operators import build_official_tucker_tfno_band_v1
    return build_official_tucker_tfno_band_v1(**kwargs)


def channel_gate(**kwargs: Any) -> Any:
    from phc2d.models.global_local_channel_gate import build_global_local_channel_gate_v1
    return build_global_local_channel_gate_v1(**kwargs)


def model_specs() -> tuple:
    shapes = dict(input_shape=(1, 64, 64), output_shape=(81, 10))
    official = dict(**shapes, width=64, depth=4, modes=(12, 12))
    gated = dict(**shapes, width=64, depth=6, modes=(12, 12), factorization_rank=.75,
                 norm_groups=8, channel_expansion=4, local_kernel=3, drop_path=.1, dropout=.05)
    return tuple((name, factory, ModelSpec(
        name=name, family="neural_operator", input_repr="material_weight_grid",
        output_target="Wh", input_layout="image", supports_variable_grid=True,
        defaults=defaults, requires=requires, source=source,
    )) for name, factory, defaults, requires, source in (
        ("official_fno_band_v1", official_fno, official, ("torch", "neuralop"), "NeuralOperator 2.0.0 FNO; band adaptation"),
        ("official_tucker_tfno_band_v1", official_tfno, {**official, "rank": .1}, ("torch", "neuralop"), "NeuralOperator 2.0.0 Tucker-TFNO; band adaptation"),
        ("global_local_channel_gate_v1", channel_gate, gated, ("torch",), "native GL with channel-wise softmax fusion"),
    ))
