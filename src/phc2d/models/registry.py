"""Lightweight model registry used by the unified workflow.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any

ModelFactory = Callable[..., Any]

_REGISTRY: dict[str, ModelFactory] = {}
_SPECS: dict[str, "ModelSpec"] = {}


@dataclass(frozen=True)
class ModelSpec:
    """Experiment-facing metadata for one model family member."""

    name: str
    family: str
    input_repr: str
    output_target: str
    canonical_name: str | None = None
    alias_of: str | None = None
    supports_variable_grid: bool = False
    supports_query: bool = False
    supports_field: bool = False
    input_layout: str = "image"
    accepts_k_points: bool = False
    accepts_lane_id: bool = False
    requires: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


def register_model(
    name: str,
    factory: ModelFactory,
    *,
    replace: bool = False,
    spec: ModelSpec | None = None,
) -> None:
    """Register one model factory by stable experiment-facing name."""
    key = str(name)
    if key in _REGISTRY and not replace:
        raise KeyError(f"Model already registered: {key}")
    _REGISTRY[key] = factory
    if spec is not None:
        if spec.name != key:
            raise ValueError(f"ModelSpec.name must match registration key: {spec.name!r} != {key!r}")
        if spec.canonical_name is None:
            spec = dataclass_replace(spec, canonical_name=spec.alias_of or key)
        if spec.alias_of is not None and spec.canonical_name != spec.alias_of:
            raise ValueError("Alias ModelSpec must set canonical_name equal to alias_of")
        if spec.input_layout not in {"flat", "image"}:
            raise ValueError("ModelSpec.input_layout must be 'flat' or 'image'")
        _SPECS[key] = spec
    elif replace:
        _SPECS.pop(key, None)


def build_model(name: str, **kwargs: Any) -> Any:
    """Build a registered model."""
    key = str(name)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown model '{key}'. Registered models: {known}")
    params = dict(_SPECS[key].defaults) if key in _SPECS else {}
    params.update(kwargs)
    if key in _SPECS and _SPECS[key].defaults:
        params = {name: value for name, value in params.items() if name in _SPECS[key].defaults}
    return _REGISTRY[key](**params)


def list_models() -> tuple[str, ...]:
    """Return registered model names."""
    return tuple(sorted(_REGISTRY))


def get_model_spec(name: str) -> ModelSpec:
    """Return registered model metadata."""
    key = str(name)
    if key not in _SPECS:
        known = ", ".join(sorted(_SPECS)) or "<empty>"
        raise KeyError(f"Unknown model spec '{key}'. Registered specs: {known}")
    return _SPECS[key]


def get_model_factory(name: str) -> ModelFactory:
    """Return the registered factory callable for a model name."""
    key = str(name)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown model '{key}'. Registered models: {known}")
    return _REGISTRY[key]


def get_canonical_model_name(name: str) -> str:
    """Return the canonical model name for bookkeeping/audits."""
    try:
        spec = get_model_spec(name)
    except KeyError:
        return str(name)
    return str(spec.canonical_name or spec.name)


def get_model_factory_identity(name: str) -> str:
    """Return a stable identity string for the factory behind a model name."""
    factory = get_model_factory(name)
    return f"{factory.__module__}.{factory.__name__}"


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return registered model specs sorted by name."""
    return tuple(_SPECS[name] for name in sorted(_SPECS))


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    """Serialize one ModelSpec for CLI or report output."""
    return {
        "name": spec.name,
        "canonical_name": spec.canonical_name,
        "alias_of": spec.alias_of,
        "family": spec.family,
        "input_repr": spec.input_repr,
        "output_target": spec.output_target,
        "supports_variable_grid": spec.supports_variable_grid,
        "supports_query": spec.supports_query,
        "supports_field": spec.supports_field,
        "input_layout": spec.input_layout,
        "accepts_k_points": spec.accepts_k_points,
        "accepts_lane_id": spec.accepts_lane_id,
        "requires": list(spec.requires),
        "defaults": dict(spec.defaults),
        "source": spec.source,
    }


def _build_res_mlp(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_res_mlp

    return build_res_mlp(**kwargs)


def _build_mlp(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_mlp

    return build_mlp(**kwargs)


def _build_pod_res_mlp(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_pod_res_mlp

    return build_pod_res_mlp(**kwargs)


def _build_lifted_cnn(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_lifted_cnn

    return build_lifted_cnn(**kwargs)


def _build_fno_lite(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_fno_lite

    return build_fno_lite(**kwargs)


def _build_fno2d_band(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_fno2d_band

    return build_fno2d_band(**kwargs)


def _build_mlp_query(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_mlp_query

    return build_mlp_query(**kwargs)


def _build_deep_onet_query(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_deep_onet_query

    return build_deep_onet_query(**kwargs)


def _build_fno2d_query(**kwargs: Any) -> Any:
    from phc2d.models.baselines import build_fno2d_query

    return build_fno2d_query(**kwargs)


def _build_mlp_mixer_band_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_upgrades import build_mlp_mixer_band_v1

    return build_mlp_mixer_band_v1(**kwargs)


def _build_res_mlp_gated_v2(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_upgrades import build_res_mlp_gated_v2

    return build_res_mlp_gated_v2(**kwargs)


def _build_convnext_band_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_upgrades import build_convnext_band_v1

    return build_convnext_band_v1(**kwargs)


def _build_tfno2d_band_v3(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_upgrades import build_tfno2d_band_v3

    return build_tfno2d_band_v3(**kwargs)


def _build_dual_lane_band_operator_v1(**kwargs: Any) -> Any:
    from phc2d.models.full_cell_upgrades import build_dual_lane_band_operator_v1

    return build_dual_lane_band_operator_v1(**kwargs)


def _build_unet_field(**kwargs: Any) -> Any:
    from phc2d.models.field import build_unet_field
    return build_unet_field(**kwargs)


def _build_fno_field(**kwargs: Any) -> Any:
    from phc2d.models.field import build_fno_field
    return build_fno_field(**kwargs)


def _build_unet_field_v2(**kwargs: Any) -> Any:
    from phc2d.models.field import build_unet_field_v2
    return build_unet_field_v2(**kwargs)


def _build_fno_field_v2(**kwargs: Any) -> Any:
    from phc2d.models.field import build_fno_field_v2
    return build_fno_field_v2(**kwargs)


def _register_builtin_models() -> None:
    """Register Gate 2 baseline specs without importing optional torch."""

    builtins: tuple[tuple[str, ModelFactory, ModelSpec], ...] = (
        ("unet_field_v2", _build_unet_field_v2, ModelSpec(name="unet_field_v2", family="unet", input_repr="epsilon_periodic_grid+field_context", output_target="active_te_field_eigen", supports_variable_grid=True, supports_field=True, requires=("torch",), defaults={"grid_channels":5,"context_dim":3,"field_channels":6,"width":48,"depth":3,"use_film":True}, source="phc2d B3 conditioned supervised TE eigenfield baseline")),
        ("fno_field_v2", _build_fno_field_v2, ModelSpec(name="fno_field_v2", family="neural_operator", input_repr="epsilon_periodic_grid+field_context", output_target="active_te_field_eigen", supports_variable_grid=True, supports_field=True, requires=("torch",), defaults={"grid_channels":5,"context_dim":3,"field_channels":6,"width":48,"depth":4,"modes":(16,16),"use_film":True,"corrected_spectral_modes":True}, source="phc2d B3 data-only FNO and adapted PINO shared architecture")),
        ("unet_field", _build_unet_field, ModelSpec(name="unet_field", family="unet", input_repr="epsilon_grid+field_context", output_target="field_eigen_te", supports_variable_grid=True, supports_field=True, requires=("torch",), defaults={"input_channels":4,"field_channels":12,"width":32,"depth":2}, source="phc2d supervised TE eigenfield baseline")),
        ("fno_field", _build_fno_field, ModelSpec(name="fno_field", family="neural_operator", input_repr="epsilon_grid+field_context", output_target="field_eigen_te", supports_variable_grid=True, supports_field=True, requires=("torch",), defaults={"input_channels":4,"field_channels":12,"width":32,"depth":4,"modes":(12,12)}, source="phc2d data-only FNO-style TE eigenfield baseline")),
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
    for name, factory, spec in builtins:
        if name not in _REGISTRY:
            register_model(name, factory, spec=spec)


_register_builtin_models()
