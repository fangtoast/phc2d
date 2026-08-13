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
            raise ValueError(
                f"ModelSpec.name must match registration key: {spec.name!r} != {key!r}"
            )
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


def _load_builtin_catalog() -> None:
    # Import after the registry API and lazy factory adapters are defined.  The
    # catalog imports these symbols to avoid importing optional torch at module load.
    from phc2d.models._builtin_models import register_builtin_models

    register_builtin_models()


_load_builtin_catalog()
