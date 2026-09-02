"""Replay command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    replay = sub.add_parser("replay", help="Physics replay and formal validation helpers.")
    replay_sub = replay.add_subparsers(dest="action", required=True)
    replay_mpb = replay_sub.add_parser(
        "mpb",
        help="Run MPB replay through WSL Ubuntu conda meep_env.",
        description="Run MPB replay through WSL Ubuntu conda meep_env.",
    )
    replay_mpb.add_argument("--config", type=Path, default=None)
    replay_mpb.set_defaults(command="replay mpb")
    effective_index = replay_sub.add_parser(
        "effective-index",
        help="Audit TE effective index and folded-frequency gap alignment.",
        description="Run the registered EXP-PHY-005 preflight or physical audit.",
    )
    effective_index.add_argument("--config", type=Path, required=True)
    effective_index.add_argument(
        "--preflight",
        action="store_true",
        help="Validate identities and sampling without importing or running MPB.",
    )
    effective_index.add_argument("--run-dir", type=Path, default=None)
    effective_index.set_defaults(command="replay effective-index")
    gap_refinement = replay_sub.add_parser(
        "gap-refinement",
        help="Refine full-BZ adjacent-band edges with validation-only real MPB.",
        description="Run a registered validation-only local or adaptive gap-refinement study.",
    )
    gap_refinement.add_argument("--config", type=Path, required=True)
    gap_refinement.add_argument(
        "--preflight",
        action="store_true",
        help="Validate frozen validation identities without importing or running MPB.",
    )
    gap_refinement.add_argument("--run-dir", type=Path, default=None)
    gap_refinement.set_defaults(command="replay gap-refinement")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "replay mpb":
        from phc2d.replay import run_replay_mpb

        if ns.config is None:
            raise SystemExit("`phc2d replay mpb` requires --config <yaml-path>.")
        print(run_replay_mpb(ns.config))
        return True
    if ns.command == "replay effective-index":
        from phc2d.replay.effective_index import run_effective_index_audit

        print(
            run_effective_index_audit(
                ns.config,
                run_dir=ns.run_dir,
                preflight=bool(ns.preflight),
            )
        )
        return True
    if ns.command == "replay gap-refinement":
        from phc2d.replay.gap_refinement import run_gap_refinement_study

        print(
            run_gap_refinement_study(
                ns.config,
                run_dir=ns.run_dir,
                preflight=bool(ns.preflight),
            )
        )
        return True
    return False
