"""Report command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    report = sub.add_parser("report", help="Result-report helpers.")
    report_sub = report.add_subparsers(dest="action")
    report_build = report_sub.add_parser(
        "build", help="Build a curated report from one run directory."
    )
    report_build.add_argument("--run", type=Path, required=True)
    report_build.set_defaults(command="report build")
    report_model_audit = report_sub.add_parser(
        "model-audit",
        help="Audit model inventory, master-report coverage, links, and source hashes.",
        description=(
            "Read a model inventory and master report without importing torch, training models, "
            "or running MPB/Meep."
        ),
    )
    report_model_audit.add_argument("--inventory", type=Path, required=True)
    report_model_audit.add_argument("--report", type=Path, required=True)
    report_model_audit.add_argument(
        "--require-source-artifacts",
        action="store_true",
        help="Fail when ignored runs/source artifacts are unavailable.",
    )
    report_model_audit.set_defaults(command="report model-audit")
    report_docs_audit = report_sub.add_parser(
        "docs-audit",
        help="Audit documentation links, reachability, anchors, and entry metadata.",
        description="Audit Git-tracked project documentation without importing training backends.",
    )
    report_docs_audit.add_argument("--root", type=Path, default=Path("README.md"))
    report_docs_audit.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when documentation contract failures are found.",
    )
    report_docs_audit.set_defaults(command="report docs-audit")
    report.set_defaults(command="report")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command in {"report", "report build"}:
        print("Curated reports live under docs/results; raw outputs live under runs/.")
        return True
    if ns.command == "report model-audit":
        from phc2d.reporting import audit_model_report

        payload = audit_model_report(
            inventory_path=ns.inventory,
            report_path=ns.report,
            require_source_artifacts=ns.require_source_artifacts,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload["status"] != "pass":
            raise SystemExit(1)
        return True
    if ns.command == "report docs-audit":
        from phc2d.reporting.docs_audit import audit_documentation

        payload = audit_documentation(repo_root=Path.cwd(), root_path=ns.root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if ns.strict and payload["status"] != "pass":
            raise SystemExit(1)
        return True
    return False
