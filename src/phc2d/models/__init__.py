"""Model registry facade for forward surrogate families.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.models.registry import (
    ModelFactory,
    ModelSpec,
    build_model,
    get_canonical_model_name,
    get_model_factory,
    get_model_factory_identity,
    get_model_spec,
    is_registered,
    list_model_specs,
    list_models,
    model_spec_to_dict,
    register_model,
)
from phc2d.models._builtin_models import register_builtin_models as _register_builtin_models
from phc2d.models.identity import (
    architecture_fingerprint,
    compare_model_identities,
    describe_model_identity,
    state_dict_key_names,
    state_dict_schema_hash,
)

_register_builtin_models()

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
    "architecture_fingerprint",
    "compare_model_identities",
    "describe_model_identity",
    "state_dict_key_names",
    "state_dict_schema_hash",
]
