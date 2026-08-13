"""Model-inventory command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    models = sub.add_parser("models", help="Registered forward surrogate model families.")
    models_sub = models.add_subparsers(dest="action", required=True)
    models_list = models_sub.add_parser(
        "list",
        help="List registered ModelSpec entries (read-only registry view).",
        description="List registered ModelSpec entries without importing torch.",
    )
    models_list.set_defaults(command="models list")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "models list":
        from phc2d.models import list_model_specs, model_spec_to_dict

        payload = [model_spec_to_dict(spec) for spec in list_model_specs()]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return True
    return False
