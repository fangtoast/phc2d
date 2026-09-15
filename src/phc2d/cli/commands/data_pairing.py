"""CLI surface for full-cell Fourier pairing-plan audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def configure_pairing_audit(data_sub: Any) -> None:
    """Add the pairing audit action to the data command domain."""

    parser = data_sub.add_parser(
        "audit-fourier-pairing",
        help="Audit paired full-cell Fourier sample plans across cutoffs.",
        description=(
            "Stream frozen Binary/Graded plans and verify sample, split, target, "
            "temperature, and shared-mode coefficient identity."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Plan run directory; repeat once for every lane/cutoff member.",
    )
    parser.add_argument(
        "--expected-cutoff",
        type=int,
        action="append",
        default=None,
        help="Required cutoff; repeat to freeze the expected cohort membership.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(command="data audit-fourier-pairing")


def run_pairing_audit(ns: Any) -> bool:
    """Run the pairing audit when selected, keeping data imports lazy."""

    if ns.command != "data audit-fourier-pairing":
        return False
    from phc2d.data.full_cell_pairing import audit_fourier_pairing_plans

    payload = audit_fourier_pairing_plans(
        ns.run_dir,
        expected_cutoffs=ns.expected_cutoff,
        output_path=ns.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return True


__all__ = ["configure_pairing_audit", "run_pairing_audit"]
