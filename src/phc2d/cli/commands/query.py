"""Query command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    query = sub.add_parser("query", help="Query-form band surrogate audit and benchmarks.")
    query_sub = query.add_subparsers(dest="action", required=True)
    query_audit = query_sub.add_parser(
        "audit",
        help="Audit query dataset schema, split lineage, and k-path metadata.",
        description="Audit query dataset schema, split lineage, and k-path metadata.",
    )
    query_audit.add_argument("--config", type=Path, required=True)
    query_audit.set_defaults(command="query audit")
    query_preflight = query_sub.add_parser(
        "preflight",
        help="Run GPU query model matrix preflight.",
        description="Run GPU query model matrix preflight.",
    )
    query_preflight.add_argument("--config", type=Path, required=True)
    query_preflight.set_defaults(command="query preflight")
    query_benchmark = query_sub.add_parser(
        "benchmark",
        help="Run formal query-form benchmark when gates allow it.",
        description="Run formal query-form benchmark when gates allow it.",
    )
    query_benchmark.add_argument("--config", type=Path, required=True)
    query_benchmark.set_defaults(command="query benchmark")
    query_data_scaling = query_sub.add_parser(
        "data-scaling",
        help="Run guarded query-form data scaling over configured V2 datasets.",
        description="Run guarded query-form data scaling over configured V2 datasets.",
    )
    query_data_scaling.add_argument("--config", type=Path, required=True)
    query_data_scaling.set_defaults(command="query data-scaling")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "query audit":
        from phc2d.query import run_query_audit

        print(run_query_audit(ns.config))
        return True
    if ns.command == "query preflight":
        from phc2d.query import run_query_preflight

        print(run_query_preflight(ns.config))
        return True
    if ns.command == "query benchmark":
        from phc2d.query import run_query_benchmark

        print(run_query_benchmark(ns.config))
        return True
    if ns.command == "query data-scaling":
        from phc2d.query import run_query_data_scaling

        print(run_query_data_scaling(ns.config))
        return True
    return False
