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

__all__ = [
    "ModelFactory",
    "ModelSpec",
    "build_model",
    "get_canonical_model_name",
    "get_model_factory",
    "get_model_factory_identity",
    "get_model_spec",
    "is_registered",
    "list_model_specs",
    "list_models",
    "model_spec_to_dict",
    "register_model",
]


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


def is_registered(name: str) -> bool:
    """Return whether a model name is present in the registry."""

    return str(name) in _REGISTRY


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
    return str(
        getattr(
            factory,
            "__phc2d_factory_identity__",
            f"{factory.__module__}.{factory.__name__}",
        )
    )


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
