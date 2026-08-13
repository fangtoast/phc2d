"""Model identity fingerprints for benchmark and audit reports.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from typing import Any, Mapping

from phc2d.models.registry import (
    ModelSpec,
    get_canonical_model_name,
    get_model_factory,
    get_model_factory_identity,
    get_model_spec,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _parameter_count(model: Any, *, trainable_only: bool = False) -> int:
    if not hasattr(model, "parameters"):
        return 0
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if bool(getattr(param, "requires_grad", False)))
    return int(sum(int(param.numel()) for param in params))


def state_dict_key_names(model: Any) -> list[str]:
    """Return state-dict keys in stable order."""

    if not hasattr(model, "state_dict"):
        return []
    return sorted(str(name) for name in model.state_dict().keys())


def state_dict_schema(model: Any) -> dict[str, list[int]]:
    """Return state-dict tensor shapes without recording tensor values."""

    if not hasattr(model, "state_dict"):
        return {}
    return {
        str(name): [int(dim) for dim in tensor.shape]
        for name, tensor in sorted(model.state_dict().items())
    }


def state_dict_schema_hash(model: Any) -> str:
    """Hash state-dict key names, tensor shapes, and tensor dtypes."""

    if not hasattr(model, "state_dict"):
        return _sha256("{}")
    payload = {
        str(name): {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(getattr(tensor, "dtype", "")),
        }
        for name, tensor in sorted(model.state_dict().items())
    }
    return _sha256(_canonical_json(payload))


def architecture_fingerprint(model: Any) -> str:
    """Hash architecture-level features without depending on trained values."""

    implementation = type(model)
    identity_module = getattr(
        implementation,
        "__phc2d_architecture_module__",
        implementation.__module__,
    )
    identity_qualname = getattr(
        implementation,
        "__phc2d_architecture_qualname__",
        implementation.__qualname__,
    )
    payload = {
        # Stable identity hooks let maintenance-only module moves preserve
        # fingerprints already recorded in checkpoints and run artifacts.
        "implementation_module": identity_module,
        "implementation_class": identity_qualname,
        "state_dict_schema": state_dict_schema(model),
        "parameter_count": _parameter_count(model),
        "trainable_parameter_count": _parameter_count(model, trainable_only=True),
    }
    return _sha256(_canonical_json(payload))


def model_spec_to_identity_dict(spec: ModelSpec) -> dict[str, Any]:
    """Serialize ModelSpec fields used by identity reports."""

    payload = asdict(spec)
    payload["defaults"] = dict(spec.defaults)
    payload["requires"] = list(spec.requires)
    return payload


def describe_model_identity(
    model: Any,
    requested_name: str,
    spec: ModelSpec | None = None,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the registry, implementation, and state schema for one model."""

    requested = str(requested_name)
    if spec is None:
        spec = get_model_spec(requested)
    factory = get_model_factory(requested)
    implementation = type(model)
    source_file = inspect.getsourcefile(implementation) or inspect.getfile(implementation)
    return {
        "requested_model_name": requested,
        "canonical_model_name": get_canonical_model_name(requested),
        "alias_of": spec.alias_of,
        "model_spec": model_spec_to_identity_dict(spec),
        "registry_factory": get_model_factory_identity(requested),
        "implementation_module": implementation.__module__,
        "implementation_class": implementation.__qualname__,
        "source_file": source_file,
        "model_kwargs": dict(model_kwargs or {}),
        "parameter_count": _parameter_count(model),
        "trainable_parameter_count": _parameter_count(model, trainable_only=True),
        "state_dict_key_names": state_dict_key_names(model),
        "state_dict_schema_hash": state_dict_schema_hash(model),
        "architecture_fingerprint": architecture_fingerprint(model),
        "factory_function_identity": f"{factory.__module__}.{getattr(factory, '__qualname__', factory.__name__)}",
    }


def compare_model_identities(identity_a: Mapping[str, Any], identity_b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two identity payloads without inspecting model tensors again."""

    keys = (
        "canonical_model_name",
        "registry_factory",
        "implementation_module",
        "implementation_class",
        "source_file",
        "model_kwargs",
        "parameter_count",
        "trainable_parameter_count",
        "state_dict_key_names",
        "state_dict_schema_hash",
        "architecture_fingerprint",
        "factory_function_identity",
    )
    matches = {key: identity_a.get(key) == identity_b.get(key) for key in keys}
    return {
        "model_a": identity_a.get("requested_model_name"),
        "model_b": identity_b.get("requested_model_name"),
        "matches": matches,
        "all_identity_fields_match": all(matches.values()),
        "same_architecture_fingerprint": bool(matches["architecture_fingerprint"]),
        "same_state_dict_schema_hash": bool(matches["state_dict_schema_hash"]),
        "same_factory_function_identity": bool(matches["factory_function_identity"]),
    }


__all__ = [
    "architecture_fingerprint",
    "compare_model_identities",
    "describe_model_identity",
    "model_spec_to_identity_dict",
    "state_dict_key_names",
    "state_dict_schema",
    "state_dict_schema_hash",
]
